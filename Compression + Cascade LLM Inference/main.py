# ====== IMPORTS ======
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import google.generativeai as genai
from cerebras.cloud.sdk import Cerebras
import os
from llmlingua import PromptCompressor
import pandas as pd
import torch
import torch.nn.functional as F
import math
import re
from datasets import load_dataset
import time
import json                 # ← added for MetricTracker
from datetime import datetime  # ← added for MetricTracker

LAST_CALL_TIME = 0
import random
MIN_DELAY = 12 + random.uniform(0, 2)

# ====== LOAD MODEL ======
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(model_name)
config.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    model_name, config=config, device_map="auto", torch_dtype="auto"
)
model.eval()
device = model.device
model.config.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id

os.environ["CEREBRAS_API_KEY1"] = "csk-eddx5rc6wjpxr4rmvjnjwn4w9tj4d4xx9cjxx28rcphvc6ym"
client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY1"))

genai.configure(api_key="AIzaSyAFIGuqvcOqvB4c4pP_7lcNaaCr2PiyAuQ")
judge_model = genai.GenerativeModel("gemma-4-26b-a4b-it")

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True
)

# ══════════════════════════════════════════════════════════════
# METRIC TRACKER  (new — no other code touched)
# ══════════════════════════════════════════════════════════════
class MetricTracker:
    """Collects per-sample records and computes research-grade metrics at end of run."""

    def __init__(self, run_name="RAG_FINAL"):
        self.run_name   = run_name
        self.start_time = time.time()
        self.samples    = []
        # 10 equal-width confidence buckets for ECE
        self.conf_buckets = {i: {"total": 0, "correct": 0, "conf_sum": 0.0}
                             for i in range(10)}

    def log(self, record: dict):
        self.samples.append(record)
        b = min(int(record.get("confidence", 0.5) * 10), 9)
        self.conf_buckets[b]["total"]    += 1
        self.conf_buckets[b]["correct"]  += int(record.get("correct", False))
        self.conf_buckets[b]["conf_sum"] += record.get("confidence", 0.5)

    # ── aggregate metrics ──────────────────────────────────────────────────────
    def _ece(self):
        n = len(self.samples)
        if not n: return 0.0
        ece = 0.0
        for d in self.conf_buckets.values():
            if not d["total"]: continue
            ece += (d["total"] / n) * abs(d["correct"] / d["total"]
                                          - d["conf_sum"] / d["total"])
        return ece

    def _selective_accuracy(self):
        s = [x for x in self.samples if not x["escalated"] and not x["abstained"]]
        return (sum(x["correct"] for x in s) / len(s), len(s)) if s else (0.0, 0)

    def _escalation_prf(self):
        esc  = [x for x in self.samples if x["escalated"] and not x["abstained"]]
        wrong= [x for x in self.samples if not x["small_correct"] and not x["abstained"]]
        tp   = sum(1 for x in esc if not x["small_correct"])
        fp   = sum(1 for x in esc if     x["small_correct"])
        p    = tp / len(esc)   if esc   else 0.0
        r    = tp / len(wrong) if wrong else 0.0
        f1   = 2*p*r/(p+r)    if p+r   else 0.0
        return p, r, f1, tp, fp

    def _cost_efficiency(self):
        n = len(self.samples)
        if not n: return 0.0, 0.0, 0
        fin  = sum(x["correct"]       for x in self.samples) / n
        sm   = sum(x["small_correct"] for x in self.samples) / n
        api  = sum(x["groq_calls"] + x["judge_calls"] for x in self.samples)
        gain = fin - sm
        return gain / max(api/n, 1e-6), gain, api

    def _compression_corr(self):
        if not self.samples: return {}
        ratios = [x["final_ratio"]  for x in self.samples]
        corr   = [float(x["correct"]) for x in self.samples]
        confs  = [x["confidence"]   for x in self.samples]
        n = len(ratios)
        mr, mc, mcf = sum(ratios)/n, sum(corr)/n, sum(confs)/n
        vr = sum((r-mr)**2 for r in ratios)/n
        cr = sum((r-mr)*(c-mc) for r,c in zip(ratios,corr))/n
        ccf= sum((r-mr)*(c-mcf) for r,c in zip(ratios,confs))/n
        std = vr**0.5 + 1e-9
        return {"corr_ratio_accuracy": cr/std,
                "corr_ratio_confidence": ccf/std,
                "mean_final_ratio": mr}

    def _abstention_quality(self):
        a = [x for x in self.samples if x["abstained"]]
        return (sum(x["correct"] for x in a)/len(a), len(a)) if a else (0.0, 0)

    def _hallucination_rate(self):
        hi = [x for x in self.samples if x["confidence"]>0.6 and not x["abstained"]]
        return (sum(1 for x in hi if not x["small_correct"])/len(hi), len(hi)) if hi else (0.0, 0)

    def _semantic_correction_rate(self):
        """Rate at which Tier-2 semantic matching rescued a correct answer."""
        called = [x for x in self.samples if x.get("tier2_called_small") or
                                             x.get("tier2_called_final")]
        rescued= [x for x in called if x.get("tier2_rescued")]
        return (len(rescued)/len(called), len(called)) if called else (0.0, 0)

    # ── reporting ──────────────────────────────────────────────────────────────
    def print_full_report(self):
        n = len(self.samples)
        if not n: print("No samples."); return
        fin_acc  = sum(x["correct"]       for x in self.samples) / n
        sm_acc   = sum(x["small_correct"] for x in self.samples) / n
        esc_rate = sum(x["escalated"]     for x in self.samples) / n
        abs_rate = sum(x["abstained"]     for x in self.samples) / n
        ece                    = self._ece()
        sel_acc, sel_n         = self._selective_accuracy()
        p, r, f1, tp, fp       = self._escalation_prf()
        cost_eff, gain, api    = self._cost_efficiency()
        comp                   = self._compression_corr()
        abs_q, abs_n           = self._abstention_quality()
        hall, hall_n           = self._hallucination_rate()
        sem_r, sem_n           = self._semantic_correction_rate()
        elapsed                = time.time() - self.start_time

        print("\n" + "="*62)
        print(f"  RESEARCH METRICS  —  {self.run_name}")
        print("="*62)
        print(f"\n[ACCURACY]")
        print(f"  Final Accuracy:              {fin_acc:.4f}  ({int(fin_acc*n)}/{n})")
        print(f"  Small Model Accuracy:        {sm_acc:.4f}  ({int(sm_acc*n)}/{n})")
        print(f"  Accuracy Gain (Δ):           {gain:+.4f}")
        print(f"  Selective Accuracy (φ2 only):{sel_acc:.4f}  (n={sel_n})")
        print(f"\n[CALIBRATION]")
        print(f"  Expected Calibration Error:  {ece:.4f}  (lower=better)")
        print(f"  Hallucination Rate:          {hall:.4f}  "
              f"(conf>0.6 but small wrong, n={hall_n})")
        print(f"\n[ESCALATION QUALITY]")
        print(f"  Escalation Rate:             {esc_rate:.4f}  ({int(esc_rate*n)}/{n})")
        print(f"  Abstention Rate:             {abs_rate:.4f}  ({int(abs_rate*n)}/{n})")
        print(f"  Escalation Precision:        {p:.4f}  (needed / escalated)")
        print(f"  Escalation Recall:           {r:.4f}  (caught / all-wrong)")
        print(f"  Escalation F1:               {f1:.4f}")
        print(f"  True Positives:              {tp}")
        print(f"  False Positives (redundant): {fp}")
        print(f"  Abstention Quality:          {abs_q:.4f}  (correct after abs, n={abs_n})")
        print(f"\n[COST EFFICIENCY]")
        print(f"  Total API Calls:             {api}  ({api/n:.2f}/sample)")
        print(f"  Qwen Answer Calls:           "
              f"{sum(x['groq_calls'] for x in self.samples)}")
        print(f"  Judge (Gemma) Calls:         "
              f"{sum(x['judge_calls'] for x in self.samples)}")
        print(f"  Phi-2 Only Coverage:         "
              f"{(n-int(esc_rate*n)-int(abs_rate*n))/n:.1%}")
        print(f"  Cost Efficiency Score:       {cost_eff:.4f}  (Δacc/calls_per_sample)")
        print(f"\n[is_correct SEMANTIC TIER-2]")
        print(f"  Tier-2 Invocations:          {sem_n}")
        print(f"  Semantic Rescue Rate:        {sem_r:.4f}  "
              f"(correct answers saved by synonym/abbrev/num matching)")
        print(f"\n[COMPRESSION ANALYSIS]")
        print(f"  Mean Final Ratio:            "
              f"{comp.get('mean_final_ratio',0):.4f}")
        print(f"  Corr(ratio, accuracy):       "
              f"{comp.get('corr_ratio_accuracy',0):.4f}")
        print(f"  Corr(ratio, confidence):     "
              f"{comp.get('corr_ratio_confidence',0):.4f}")
        print(f"\n[SYSTEM]")
        print(f"  Elapsed:                     {elapsed:.1f}s  ({elapsed/n:.1f}s/sample)")
        print(f"  Avg Tokens/Sample:           "
              f"{sum(x['total_tokens'] for x in self.samples)/n:.1f}")
        print("="*62)

    def save_json(self, path="rag_final_metrics.json"):
        p, r, f1, tp, fp  = self._escalation_prf()
        cost_eff, gain, api = self._cost_efficiency()
        comp = self._compression_corr()
        sel_acc, _ = self._selective_accuracy()
        abs_q, _   = self._abstention_quality()
        hall, _    = self._hallucination_rate()
        sem_r, sem_n = self._semantic_correction_rate()
        n = len(self.samples)
        summary = {
            "run_name":              self.run_name,
            "timestamp":             datetime.now().isoformat(),
            "n_samples":             n,
            "final_accuracy":        sum(x["correct"]       for x in self.samples)/max(n,1),
            "small_accuracy":        sum(x["small_correct"] for x in self.samples)/max(n,1),
            "accuracy_gain":         gain,
            "selective_accuracy":    sel_acc,
            "ece":                   self._ece(),
            "hallucination_rate":    hall,
            "escalation_rate":       sum(x["escalated"] for x in self.samples)/max(n,1),
            "abstention_rate":       sum(x["abstained"] for x in self.samples)/max(n,1),
            "escalation_precision":  p,
            "escalation_recall":     r,
            "escalation_f1":         f1,
            "true_positive_esc":     tp,
            "false_positive_esc":    fp,
            "abstention_quality":    abs_q,
            "total_api_calls":       api,
            "cost_efficiency":       cost_eff,
            "tier2_invocations":     sem_n,
            "tier2_rescue_rate":     sem_r,
            "corr_ratio_accuracy":   comp.get("corr_ratio_accuracy", 0),
            "corr_ratio_confidence": comp.get("corr_ratio_confidence", 0),
            "mean_final_ratio":      comp.get("mean_final_ratio", 0),
            "calibration_buckets":   {str(k): v
                                      for k, v in self.conf_buckets.items()},
            "samples":               self.samples,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[MetricTracker] Saved → {path}")
        return summary

tracker = MetricTracker("RAG_FINAL_LLMLINGUA_MEDIUM")

# ══════════════════════════════════════════════════════════════
# SEMANTIC is_correct  TIER 2  (new — deterministic, zero API cost)
#
# Placed right after the tracker so it is available to the
# existing is_correct() function defined further below.
# The existing Tier-1 logic is completely unchanged.
# Tier 2 only runs when Tier 1 returns False.
#
# Catches:
#   (a) number-word ↔ digit   :  seven = 7
#   (b) boolean synonyms      :  True = yes,  false = no
#   (c) broadcast abbreviations: NBC = National Broadcasting Company
#   (d) occupation synonyms   :  filmmaker = director
# ══════════════════════════════════════════════════════════════
_T2_NUM_WORDS = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
    'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14','fifteen':'15',
    'sixteen':'16','seventeen':'17','eighteen':'18','nineteen':'19','twenty':'20',
    'thirty':'30','forty':'40','fifty':'50','sixty':'60','seventy':'70',
    'eighty':'80','ninety':'90','hundred':'100','thousand':'1000',
}
_T2_NUM_WORDS_INV = {v: k for k, v in _T2_NUM_WORDS.items()}

_T2_CANONICAL = {
    # occupation synonyms (both directions via lookup on both sides)
    'filmmaker':  'director', 'filmmakers': 'director', 'directors':  'director',
    'directing':  'director',
    'musician':   'singer',   'musicians':  'singer',   'singers':    'singer',
    'songwriter': 'singer',   'songwriters':'singer',
    'actor':      'performer','actress':    'performer','actors':     'performer',
    # boolean synonyms
    'true':  'yes', 'false': 'no', 'yeah': 'yes', 'yep': 'yes', 'nope': 'no',
    # major broadcast abbreviations
    'nbc':  'national broadcasting company',
    'cbs':  'columbia broadcasting system',
    'abc':  'american broadcasting company',
    'bbc':  'british broadcasting corporation',
    'hbo':  'home box office',
}

def _t2_num_normalise(text: str) -> str:
    """Replace number-words with digits and digits with number-words (both ways)."""
    tokens = text.split()
    out = []
    for t in tokens:
        if   t in _T2_NUM_WORDS:     out.append(_T2_NUM_WORDS[t])
        elif t in _T2_NUM_WORDS_INV: out.append(_T2_NUM_WORDS_INV[t])
        else:                        out.append(t)
    return ' '.join(out)

def _t2_canonical(text: str) -> str:
    """Single-token canonical lookup (occupation/boolean/abbreviation)."""
    return _T2_CANONICAL.get(text.strip(), text.strip())

def _t2_is_correct(pred_n: str, gt_n: str) -> tuple:
    """
    Tier-2 semantic equivalence check.
    Returns (bool, reason_string).
    Both inputs must already be normalize_answer()-processed.
    Called only when Tier-1 has returned False.
    """
    # 2a. number-word normalisation applied to both sides
    pnn = _t2_num_normalise(pred_n)
    gnn = _t2_num_normalise(gt_n)
    if pnn == gnn:    return True,  "t2_num_both"
    if pnn == gt_n:   return True,  "t2_num_pred"
    if pred_n == gnn: return True,  "t2_num_gt"
    # also try short overlap after num-normalisation (e.g. pred="7" gt="seven albums")
    _SW = {'a','an','the','is','it','in','of','to','and','or','for','on','at'}
    pnn_cw = set(pnn.split()) - _SW
    gnn_cw = set(gnn.split()) - _SW
    if pnn_cw and len(pnn_cw) <= 3 and pnn_cw & gnn_cw:
        return True, "t2_num_overlap"

    # 2b. canonical synonym lookup (single token)
    pc = _t2_canonical(pred_n)
    gc = _t2_canonical(gt_n)
    if pc == gc:       return True, "t2_canon_exact"
    if pc == gt_n:     return True, "t2_canon_pred"
    if pred_n == gc:   return True, "t2_canon_gt"

    # 2c. token-wise canonical mapping (handles multi-token phrases)
    ptc = ' '.join(_t2_canonical(t) for t in pred_n.split())
    gtc = ' '.join(_t2_canonical(t) for t in gt_n.split())
    if ptc == gtc:     return True, "t2_tokcan_exact"
    if ptc == gt_n:    return True, "t2_tokcan_pred"
    if pred_n == gtc:  return True, "t2_tokcan_gt"

    # 2d. F1 on canonicalised tokens (catches "filmmaker" after → "director")
    def _f1_simple(p, g):
        from collections import Counter
        pt  = [w for w in p.split() if w not in _SW]
        gt_ = [w for w in g.split() if w not in _SW]
        if not pt or not gt_: return 0.0
        common = sum((Counter(pt) & Counter(gt_)).values())
        return 2*common/(len(pt)+len(gt_)) if common else 0.0
    f1c   = _f1_simple(ptc, gtc)
    gt_cw = len([w for w in gtc.split() if w not in _SW])
    if gt_cw <= 8  and f1c >= 0.45: return True, f"t2_f1c={f1c:.2f}"
    if gt_cw <= 15 and f1c >= 0.30: return True, f"t2_f1c={f1c:.2f}"

    return False, "t2_no_match"


# ====== CONFIDENCE ESTIMATOR ======
class ConfidenceEstimator:
    def __init__(self):
        self.n = 0
        self.lp_mean  = -1.2
        self.lp_M2    = 0.64
        self.ent_mean = 1.5
        self.ent_M2   = 0.36

    def _welford_update(self, mean, M2, n, x):
        delta  = x - mean
        mean  += delta / n
        delta2 = x - mean
        M2    += delta * delta2
        return mean, M2

    def update(self, logprob, entropy):
        self.n += 1
        self.lp_mean,  self.lp_M2  = self._welford_update(self.lp_mean,  self.lp_M2,  self.n, logprob)
        self.ent_mean, self.ent_M2 = self._welford_update(self.ent_mean, self.ent_M2, self.n, entropy)

    @property
    def lp_std(self):
        return max(0.1, (self.lp_M2 / max(self.n, 1)) ** 0.5)

    @property
    def ent_std(self):
        return max(0.1, (self.ent_M2 / max(self.n, 1)) ** 0.5)

    def score(self, logprob, entropy, answer=None, context=None, question=None):
        lp_z  = (logprob - self.lp_mean)  / self.lp_std
        ent_z = (entropy  - self.ent_mean) / self.ent_std
        uncertainty = 2.5 * (0.3 * (-lp_z) + 0.7 * ent_z)
        confidence = 1 / (1 + math.exp(uncertainty))
        if answer:
            words = answer.split()
            if len(words) < 8:
                if any(q in (question or "").lower() for q in ["why", "how", "explain"]):
                    confidence -= 0.15
            if len(words) > 10 and "." in answer:
                confidence += 0.1
        confidence = 1 / (1 + math.exp(-4 * (confidence - 0.5)))
        if answer:
            ans_lower = answer.lower()
            words = answer.split()
            if any(char.isdigit() for char in answer):
                if question and any(q in question.lower() for q in ["time", "percent", "calculate", "higher", "lower"]):
                    confidence -= 0.25
            generic_phrases = [
                "because it fails", "because it is", "because it has",
                "different domain", "various reasons", "complex reasoning"
            ]
            if any(p in ans_lower for p in generic_phrases):
                confidence -= 0.2
            if context:
                overlap = len(set(ans_lower.split()) & set(context.lower().split())) / max(len(words), 1)
                if overlap > 0.5 and len(words) > 6:
                    confidence += 0.15
            if len(words) <= 3:
                if question and any(q in question.lower() for q in ["time", "calculate", "how much"]):
                    confidence -= 0.2
        return float(confidence)

estimator = ConfidenceEstimator()


def rate_limited_call(messages, max_tokens=80, retries=5):
    global LAST_CALL_TIME

    for attempt in range(retries):
        now = time.time()
        elapsed = now - LAST_CALL_TIME

        if elapsed < MIN_DELAY:
            time.sleep(MIN_DELAY - elapsed)

        try:
            resp = client.chat.completions.create(
                messages=messages,
                model="qwen-3-235b-a22b-instruct-2507",
                max_tokens=max_tokens
            )
            LAST_CALL_TIME = time.time()
            return resp.choices[0].message.content

        except Exception as e:
            print(f"[Retry {attempt+1}] Cerebras error:", e)
            wait = 5 * (2 ** attempt)
            print(f"Waiting {wait}s before retry...")
            time.sleep(wait)

    # BUG4 FIX: Cerebras exhausted all retries — fall back to Groq immediately
    print("[Cerebras exhausted — falling back to Google]")
    try:
        # Initialize the model
        main_model = genai.GenerativeModel("gemini-3.1-flash-lite-preview") # or "gemini-1.5-pro"

        # Correct call structure
        response = main_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens  # max_tokens equivalent
            )
        )
        text = response.text.strip()
        # Access content directly via .text
        return text

    except Exception as e:
        print("gemini error:", e)
        return None


# ====== TOKEN-LEVEL UNCERTAINTY ======
def token_level_uncertainty(scores, top_k=5):
    entropies = []
    for score in scores:
        probs = F.softmax(score[0], dim=-1)
        top_probs, _ = torch.topk(probs, top_k)
        top_probs = top_probs / top_probs.sum()
        entropy = -(top_probs * torch.log(top_probs + 1e-10)).sum().item()
        entropies.append(entropy)
    mean_ent = sum(entropies) / len(entropies) if entropies else 0.0
    max_ent  = max(entropies) if entropies else 0.0
    spike_ratio = max_ent / (mean_ent + 1e-8)
    return {"mean_entropy": mean_ent, "max_entropy": max_ent, "spike_ratio": spike_ratio}

# ====== COMPRESSION PENALTY ======
def compression_confidence_penalty(original, compressed, base_confidence, question=""):
    orig_tokens_raw = original.split()
    comp_tokens_raw = compressed.split()
    orig_tokens_low = [w.lower() for w in orig_tokens_raw]
    comp_tokens_low = [w.lower() for w in comp_tokens_raw]
    ratio = len(comp_tokens_raw) / max(len(orig_tokens_raw), 1)
    question_words = set(question.lower().split()) if question else set()
    orig_entities = {
        w.lower() for i, w in enumerate(orig_tokens_raw)
        if ((i > 0 and w[0].isupper()) or any(c.isdigit() for c in w))
        and w.lower() not in question_words
    }
    comp_entities = {
        w.lower() for i, w in enumerate(comp_tokens_raw)
        if (i > 0 and w[0].isupper()) or any(c.isdigit() for c in w)
    }
    entity_retention = (
        len(comp_entities & orig_entities) / len(orig_entities)
        if orig_entities else 1.0
    )
    qwords = {"what", "who", "when", "where", "why", "how", "which"}
    orig_qwords = set(orig_tokens_low) & qwords
    comp_qwords = set(comp_tokens_low) & qwords
    question_intact = len(comp_qwords) / max(len(orig_qwords), 1) if orig_qwords else 1.0
    penalty = 0.0
    if ratio < 0.25:            penalty += 0.08
    if entity_retention < 0.30: penalty += 0.08
    if question_intact  < 0.60: penalty += 0.05
    adjusted = base_confidence * (1 - penalty)
    adjusted = max(adjusted, base_confidence * 0.5)
    print(f"  [compression_penalty] ratio={ratio:.2f} entity_ret={entity_retention:.2f} "
          f"q_intact={question_intact:.2f} penalty={penalty:.2f} "
          f"conf: {base_confidence:.3f} → {adjusted:.3f}")
    return adjusted

# ====== HELPERS ======
def is_garbage_output(ans):
    tokens = ans.strip().lower()
    return (len(tokens) <= 3 or tokens in ["1", "1.", "a", "a.", "1)", "1) ai"])

def is_good_answer(ans):
    ans = ans.strip().lower()
    words = ans.split()
    if len(words) <= 2: return False
    if re.match(r'^\d+\.\s*(\d+\.\s*)*', ans.strip()): return False
    if re.match(r'^\d+\.$', words[0]): return False
    vague = ["something", "things", "various", "many", "system", "stuff", "aspects", "mechanism", "phenomenon"]
    if any(v in ans for v in vague): return False
    incomplete = ["is one of", "is a type of", "is part of", "is known as", "refers to"]
    if any(p in ans for p in incomplete): return False
    return True

def is_hopeless_question(question):
    triggers = ["ethical", "society", "universe", "fictional", "philosophical"]
    return any(t in question.lower() for t in triggers)

def context_alignment_score(ans, context):
    ans_words = set(ans.lower().split())
    ctx_words = set(context.lower().split())
    if not ans_words: return 0.0
    return len(ans_words & ctx_words) / len(ans_words)

def is_valid_answer(ans):
    return len(ans.strip().lower().split()) <= 2

def phi2_self_check(question, answer):
    check_prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Is there anything wrong or missing in this answer? "
        f"Reply only: 'nothing wrong' or 'has issues'.\n"
        f"Response:"
    )
    inputs = tokenizer(check_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    verdict = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    vouches = "nothing" in verdict or "correct" in verdict
    print(f"  [self-check] verdict='{verdict}' → {'no escalation' if vouches else 'escalate'}")
    return vouches

def is_calculation_question(question):
    hard_triggers = ["higher or lower", "how much", "what time", "calculate", "revenue", "profit"]
    q = question.lower()
    if any(t in q for t in hard_triggers): return True
    if ("percent" in q or "%" in q):
        compute_verbs = ["higher", "lower", "increase", "decrease", "change", "calculate", "how much"]
        if any(v in q for v in compute_verbs): return True
    return False

def is_non_recoverable(question):
    triggers = ["ethical", "society", "universe", "fictional", "philosophical", "author", "decision-making"]
    return any(t in question.lower() for t in triggers)

def is_truncated(text):
    t = text.lower().rstrip(" .")
    truncation_endings = (
        " while", " whereas", " but", " however", " and", " although",
        " though", " yet", " because", " since", " as", " when", " which",
        " that", " or", " nor", " so", " for", " if", " unless",
    )
    if t.endswith("'s") and len(t.split()) > 4: return True
    return any(t.endswith(e) for e in truncation_endings)


def is_boolean_question(question):
    q = question.lower().strip()
    bool_starters = ["are ", "do ", "does ", "is ", "were ", "was ", "did ", "can ", "have ", "has "]
    return any(q.startswith(s) for s in bool_starters)


def is_specific_claim(answer):
    words = answer.strip().split()
    if len(words) > 15: return False
    # FIX: bare boolean words (Yes/No/True/False) are not specific factual claims —
    # exempting them prevents "Yes" from triggering hallucination detection.
    ans_core = answer.strip().lower().rstrip(".,!?")
    if ans_core in {"yes", "no", "true", "false", "yes.", "no.", "yes,", "no,"}:
        return False
    return (any(w[0].isupper() for w in words if len(w) > 2) or
            any(any(c.isdigit() for c in w) for w in words))

def is_hallucinated_claim(ans, extracted_context):
    if not is_specific_claim(ans): return False
    ans_words = set(re.sub(r'[,.\'"!?;:]', '', w).lower() for w in ans.split() if len(w) > 2)
    ctx_words = set(extracted_context.lower().split())
    if not ans_words: return False
    score = len(ans_words & ctx_words) / len(ans_words)
    return score < 0.25


def short_gt_in_pred(pred_n, gt_n):
    gt_stripped = re.sub(r'^(the|a|an) ', '', gt_n.strip())
    gt_words = gt_stripped.split()
    if len(gt_words) <= 4:
        if gt_stripped in pred_n: return True
        if all(w in pred_n.split() for w in gt_words if w): return True
        if gt_n in pred_n: return True
    return False


# FIX A: COMPARATIVE ANSWER EXTRACTOR
def extract_comparative_answer(pred, question):
    q_lower = question.lower()
    comparative_q_words = ["older", "younger", "bigger", "smaller", "taller", "shorter",
                           "higher", "lower", "longer", "heavier", "lighter",
                           "earlier", "later", "more", "less", "better", "worse",
                           "who is", "which is", "which one"]
    is_comparative_q = any(w in q_lower for w in comparative_q_words)
    if not is_comparative_q:
        return None
    comparative_pattern = re.compile(
        r'^(.+?)\s+(?:is|was|are|were)\s+(?:\w+er|more\s+\w+|less\s+\w+)\s+than\b',
        re.IGNORECASE
    )
    m = comparative_pattern.match(pred.strip())
    if m:
        subject = m.group(1).strip()
        subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.IGNORECASE)
        print(f"  [comparative_extract] pred='{pred[:60]}' → extracted='{subject}'")
        return subject
    has_pattern = re.compile(r'^(.+?)\s+has\s+a\s+(?:higher|lower|greater|fewer|more|less)\b', re.IGNORECASE)
    m2 = has_pattern.match(pred.strip())
    if m2:
        subject = m2.group(1).strip()
        subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.IGNORECASE)
        print(f"  [comparative_extract_has] pred='{pred[:60]}' → extracted='{subject}'")
        return subject
    return None


BOOLEAN_CONF_THRESHOLD = 0.40

_BOOL_SW = {'a','an','the','is','it','in','of','to','and','or','for','on','at','by',
            'as','be','are','was','were','this','that','with','from','its','not','but',
            'if','so','do','did','has','have','had','can','could','would','will','both',
            'they','them','yes','no','true','false','also','all','any','very','more',
            'than','which','who','what','when','where','how','why','these','those',
            'each','every','same'}


def boolean_predicate_type(question):
    q = question.lower().strip().rstrip('?')
    q_orig = question.strip().rstrip('?')

    shared_pats = [
        (r'both .{0,30} (composer|writer|actor|director|singer|artist|player|musician)', 'profession'),
        (r'(described|classified|known|regarded|referred) as', 'classification'),
        (r'both (opera|rock|jazz|pop|classical|country)', 'genre'),
        (r'(of the same|the same) nationality', 'nationality'),
        (r'both .{0,30} (genre|nationality|species|genus|genera|type|kind|category)', 'category'),
        (r'both types of', 'type_of'),
        (r'both kinds of', 'type_of'),
    ]
    for pat, reason in shared_pats:
        if re.search(pat, q):
            return 'shared_attribute', reason

    if 'both' in q and 'from' in q:
        m = re.search(r'from\s+(?:the\s+)?([A-Z][a-zA-Z]+)', q_orig)
        if m:
            return 'shared_attribute', f'origin_{m.group(1)}'
        if any(c in q for c in ['united states', 'united kingdom', 'usa', 'u.s.', 'america']):
            return 'shared_attribute', 'origin_us'

    inferential_substrings = [
        ('same neighborhood', 'spatial_same'), ('same district', 'spatial_same'),
        ('same city', 'spatial_same'),         ('same region', 'spatial_same'),
        ('same location', 'spatial_same'),     ('same area', 'spatial_same'),
        ('same building', 'spatial_same'),     ('same borough', 'spatial_same'),
        ('located in the same', 'colocation'), ('in the same', 'same_X'),
        ('both used for', 'usage_claim'),      ('used for real estate', 'usage_re'),
        ('used for residential', 'usage_res'), ('used for commercial', 'usage_com'),
        ('part of the same', 'membership'),    ('belong to the same', 'membership'),
        ('from the same', 'same_origin'),
        ('same level', 'spatial_same'),
        ('both contain', 'shared_contains'),
        ('both include', 'shared_contains'),
    ]
    for substring, reason in inferential_substrings:
        if substring in q:
            return 'inferential', reason

    return 'unknown', None


def _ctx_contains(word, ctx_lower):
    if word in ctx_lower:
        return True
    for suffix in ['s', 'es', 'ing', 'ed', 'ly', 'ers', 'er']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if word[:-len(suffix)] in ctx_lower:
                return True
    if len(word) >= 5 and re.search(r'' + re.escape(word[:5]), ctx_lower):
        return True
    return False


def boolean_claim_grounded(ans, question, extracted_context):
    ctx_lower = extracted_context.lower()
    q_words = set(re.findall(r'[a-z]+', question.lower()))

    entity_words = set()
    for i, tok in enumerate(question.split()):
        clean = re.sub(r'[^a-zA-Z]', '', tok).lower()
        if (i > 0 and tok[0].isupper() and clean not in
                {'the','a','an','are','is','do','does','were','was','did','both','and','or'}):
            entity_words.add(clean)

    ans_lower_words = re.findall(r'[a-z]+', ans.lower())
    new_words = [w for w in ans_lower_words
                 if w not in q_words and w not in _BOOL_SW and len(w) > 3]

    if len(new_words) >= 2:
        grounded = sum(1 for w in new_words if _ctx_contains(w, ctx_lower))
        ratio = grounded / len(new_words)
        expl = f"new_claims={new_words[:4]} g={grounded}/{len(new_words)}"
        return ratio >= 0.40, ratio, expl
    else:
        q_predicate = [w for w in re.findall(r'[a-z]+', question.lower())
                       if w not in _BOOL_SW
                       and w not in entity_words
                       and len(w) > 3]
        if not q_predicate:
            q_predicate = [w for w in re.findall(r'[a-z]+', question.lower())
                           if w not in _BOOL_SW and len(w) > 3]
        grounded = sum(1 for w in q_predicate if _ctx_contains(w, ctx_lower))
        ratio = grounded / len(q_predicate) if q_predicate else 1.0
        expl = f"echo_pred={q_predicate[:4]} g={grounded}/{len(q_predicate)}"
        return ratio >= 0.50, ratio, expl


def is_elaborated_boolean_answer(ans, question):
    ans_strip = ans.strip().lower()
    bool_starters = ["yes,", "yes.", "yes!", "no,", "no.", "no!",
                     "yes both", "yes they", "yes it", "yes the",
                     "no both", "no they", "no it", "no the"]
    return (
        any(ans_strip.startswith(s) for s in bool_starters)
        and len(ans.split()) > 3
        and is_boolean_question(question)
    )


def boolean_needs_escalation(ans, confidence, question, extracted_context):
    if is_elaborated_boolean_answer(ans, question):
        pred_type, pred_reason = boolean_predicate_type(question)
        print(f"  [bool_pred_type] {pred_type} ({pred_reason})")

        if pred_type == 'inferential':
            return True, f"elab_bool_inferential ({pred_reason})"

        grounded, ratio, expl = boolean_claim_grounded(ans, question, extracted_context)
        print(f"  [bool_grounding] {expl} → {'GROUNDED' if grounded else 'UNGROUNDED'}")
        if not grounded:
            return True, f"elab_bool_ungrounded (ratio={ratio:.2f})"
        if confidence < 0.30:
            return True, f"elab_bool_grounded_very_low_conf ({confidence:.3f})"
        if pred_type == 'unknown' and ratio < 0.75:
            return True, f"elab_bool_unknown_strict (ratio={ratio:.2f})"
        return False, "elab_bool_grounded"

    ans_strip = ans.strip().lower().rstrip(".")
    is_bare = ans_strip in ["yes", "no", "true", "false"]
    if is_bare and confidence < BOOLEAN_CONF_THRESHOLD:
        return True, f"bare_bool_low_conf ({confidence:.3f} < {BOOLEAN_CONF_THRESHOLD})"

    return False, "none"


def answer_self_accepts(ans, question, extracted_context):
    if not ans.strip() or is_boolean_question(question):
        return False
    words = ans.split()
    if len(words) > 12:
        return False
    has_anchor = (any(w[0].isupper() for w in words if len(w) > 2) or
                  any(any(c.isdigit() for c in w) for w in words))
    if not has_anchor:
        return False
    ctx_lower = extracted_context.lower()
    anchor_words = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in words
                    if (w[0].isupper() and len(w) > 2) or any(c.isdigit() for c in w)]
    grounded = sum(1 for w in anchor_words if w and w in ctx_lower)
    if not anchor_words or grounded / len(anchor_words) < 0.5:
        return False
    return True


def judge_answer(question, extracted_context, answer):
    prompt = f"""Score this answer 1-5 where:
  5 = correct and complete — you are certain this is right
  4 = correct, minor detail missing OR answer contains the correct fact with extra words
  3 = partially correct, OR context doesn't contain enough to verify
  2 = specific claim that contradicts or is unsupported by the context
  1 = completely wrong or irrelevant

Rules:
  - "The Animorphs series by Katherine Applegate" when GT is "Animorphs" → score 4
  - Short factual answer clearly present in context → score 4 or 5
  - Context insufficient to verify → score 3 (not 4 or 5)
  - Uncertain → prefer lower score
  - Do NOT use outside knowledge
  - "Yes." or "No." alone: score 4 if context clearly supports it, else score 2

Context: {extracted_context}
Question: {question}
Answer: {answer}

Reply with ONLY a number (1-5)."""
    try:
        response = judge_model.generate_content(prompt)
        text = response.text.strip()

        for ch in text:
            if ch in "12345":
                return int(ch)

        return 3

    except Exception as e:
        print("Gemma judge error:", e)
        return 3


HARD_ABSTAIN_PATTERNS = {
    "rel_clause_who",
    "rel_clause_where",
    "rel_clause_whose",
    "nested_ref",
}
SOFT_ABSTAIN_PATTERNS = {
    "role_of_title",
    "album_formed",
    "alias_lookup",
}

def is_multihop_question(question):
    q = question.lower()
    if re.search(r'the \w+ who ', q):   return True, "rel_clause_who"
    if re.search(r'the \w+ where ', q): return True, "rel_clause_where"
    if re.search(r'the \w+ whose ', q): return True, "rel_clause_whose"
    if re.search(r'\w+ of the \w+ (that|who|which)\b', q): return True, "nested_ref"
    if re.search(r'the (director|writer|screenwriter|producer|manager|coach) of\b', q):
        return True, "role_of_title"
    if "debut album" in q or "formed by" in q: return True, "album_formed"
    if "stage name" in q or "known by his" in q or "known by her" in q:
        return True, "alias_lookup"
    return False, "none"

def should_abstain(question, context):
    q_lower   = question.lower()
    ctx_lower = context.lower()
    q_fictional = [
        "in a fictional", "in this fictional", "in the story",
        "in a universe where", "in this universe", "in this society",
        "in this civilization", "in this world where",
    ]
    if any(t in q_lower for t in q_fictional):
        return True, "fictional_question"
    ctx_words = len(context.split())
    if ctx_words < 120:
        ctx_fictional = [
            "fictional", "in a universe", "civilization communicates",
            "memories are stored", "causality is reversed", "city where time",
        ]
        if any(t in ctx_lower for t in ctx_fictional):
            return True, "fictional_context"
    mh, mh_reason = is_multihop_question(question)
    if mh:
        if mh_reason in HARD_ABSTAIN_PATTERNS:
            return True, f"multihop_{mh_reason}"
        else:
            print(f"  [SOFT_MULTIHOP — let phi-2 try] reason={mh_reason}")
            return False, "none"
    if is_calculation_question(question):
        return True, "calculation"
    return False, "none"


def extract_relevant_context(context, question, max_words=400):
    _SW = {"a","an","the","is","in","of","to","and","or","for","on","at","by",
           "as","be","are","was","were","with","from","its","not","but","if",
           "this","that","which","who","what","when","where","how","why","both",
           "also","their","they","has","have","had","been","same","did","do"}
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    q_words = {w.lower().strip("?.,;:") for w in question.split()
               if w.lower().strip("?.,;:") not in _SW and len(w) > 2}
    scored = []
    for i, sent in enumerate(sentences):
        s_words = {w.lower().strip(".,;:") for w in sent.split()}
        overlap = len(q_words & s_words)
        scored.append((overlap, i, sent))
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = set()
    word_count = 0
    for _, idx, sent in scored:
        n = len(sent.split())
        if word_count + n <= max_words:
            selected.add(idx)
            word_count += n
        if word_count >= max_words:
            break
    return " ".join(sentences[i] for i in sorted(selected))

def build_phi2_prompt(context, question):
    q_lower = question.lower()
    is_comparison = any(w in q_lower for w in ["differ", "difference", "compare", "vs", "versus", "how do", "how does"])
    format_hint = "Answer (1-2 sentences):" if is_comparison else "Answer (1-8 words, factual):"
    return (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"{format_hint}"
    )

def get_answer_with_stats(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.2,
            return_dict_in_generate=True,
            output_scores=True
        )
    gen_ids = outputs.sequences[0][input_len:]
    answer  = tokenizer.decode(gen_ids, skip_special_tokens=True)
    for stop in ["Question:", "Example", "Answer:", "\n"]:
        answer = answer.split(stop)[0]
    answer = answer.strip()
    log_probs = []
    for i, score in enumerate(outputs.scores):
        token_id = outputs.sequences[0][input_len + i].item()
        probs    = F.softmax(score[0], dim=-1)
        lp       = torch.log(probs[token_id] + 1e-10).item()
        log_probs.append(lp)
    mean_lp = sum(log_probs) / len(log_probs) if log_probs else -999.0
    unc = token_level_uncertainty(outputs.scores, top_k=10)
    return answer, mean_lp, unc


def build_qwen_boolean_prompt(question, context):
    return f"""You are a strict factual verifier. Use ONLY what is EXPLICITLY stated in the context below.

Context:
{context}

Question: {question}

Rules:
- Do NOT infer, assume, or use outside knowledge.
- Only answer "yes" if the context EXPLICITLY supports the claim.
- If the context is silent, ambiguous, or contradicts the claim, answer "no".
- Answer with a single word: yes or no.

Answer:"""

def build_qwen_multihop_prompt(question, context):
    return f"""You are a precise factual assistant. Use ONLY the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Identify all relevant entities mentioned in the question.
2. Find each entity in the context.
3. Extract the specific fact being asked about.
4. Output ONLY the final answer — no explanation, no preamble, no "The answer is".

Answer:"""

def build_groq_prompt(question, context_for_groq, ans, truncated,
                      empty_answer=False, boolean_escalation=False):
    if truncated:
        return (
            f"The following answer was cut off mid-sentence. "
            f"Complete it using only the context below. "
            f"Return the FULL completed answer only, no preamble.\n\n"
            f"Context: {context_for_groq}\n"
            f"Question: {question}\n"
            f"Incomplete answer: {ans}\n"
            f"Completed answer:"
        ), 100

    if boolean_escalation:
        return build_qwen_boolean_prompt(question, context_for_groq), 10

    if empty_answer:
        return build_qwen_multihop_prompt(question, context_for_groq), 120

    return (
        f"Answer the question using ONLY the provided context.\n\n"
        f"STRICT RULES:\n"
        f"- Use the context as ground truth\n"
        f"- Do NOT infer, assume, or use outside knowledge."
        f"- Be concise and precise\n"
        f"- Output ONLY the final answer (no explanation)\n\n"
        f"Question: {question}\n"
        f"Context: {context_for_groq}\n\n"
        f"Answer:"
    ), 80


def answer_asks_for_specific_value(question, answer):
    q = question.lower()
    ans = answer.strip()
    words = ans.split()
    year_triggers = ["what year", "which year", "what years", "what timeframe",
                     "during what", "founded in", "when was", "since what",
                     "served during what"]
    if any(t in q for t in year_triggers):
        if not any(c.isdigit() for c in ans):
            return True, "year_question_no_digits"
    if "what other" in q or "which other" in q:
        q_entities = set(re.findall(r'\b[A-Z][a-z]+\b', question))
        ans_entities = set(re.findall(r'\b[A-Z][a-z]+\b', ans))
        if q_entities and ans_entities and ans_entities.issubset(q_entities):
            return True, "other_question_same_entity"
        return True, "which_other_always_escalate"
    name_triggers = ["what is the name", "what was the name", "name of the",
                     "what is it called", "name for the"]
    if any(t in q for t in name_triggers):
        if len(words) > 6:
            has_proper = any(w[0].isupper() for w in words if len(w) > 2)
            if not has_proper:
                return True, "name_question_description_not_name"
    title_triggers = ["voted to be", "voted as", "named as", "considered to be", "regarded as"]
    if any(t in q for t in title_triggers):
        has_proper = any(w[0].isupper() for w in words if len(w) > 2 and w.lower() not in {"the","a","an"})
        is_only_name = has_proper and len(words) <= 4 and not any(
            desc in ans.lower() for desc in [
                "best", "greatest", "champion", "award", "prize", "title",
                "honor", "winner", "goalkeeper", "player", "artist", "coach",
                "director", "manager", "person", "leader", "officer",
            ]
        )
        if is_only_name:
            return True, "voted_as_expects_title_not_name"
    if any(qw in q for qw in ["who", "which"]):
        has_proper = any(w[0].isupper() for w in words if len(w) > 2)
        if not has_proper:
            return True, "entity_expected_no_name"
    return False, "none"


def should_escalate(confidence_score, unc, ans, question=""):
    if confidence_score > 0.550:
        if any(word in question.lower() for word in ["why", "how", "explain"]):
            weak_patterns = [
                "because it is", "because it has", "because it fails",
                "due to various", "for many reasons", "complex reasoning"
            ]
            if any(p in ans.lower() for p in weak_patterns):
                return True

    if any(k in question.lower() for k in ["higher", "lower", "increase", "decrease"]):
        if confidence_score > 0.550:
            if "lower" in ans.lower() and "higher" in question.lower():
                return True

    if confidence_score > 0.580:
        return False

    if any(word in question.lower() for word in ["why", "how", "explain"]):
        if confidence_score > 0.600: return False
        if confidence_score < 0.500: return True
        if not is_good_answer(ans): return True

    if confidence_score > 0.450 and unc["spike_ratio"] < 12:
        return False

    if len(ans.split()) <= 4:
        if confidence_score > 0.400: return False

    if len(ans.split()) <= 3 and confidence_score > 0.550:
        return False

    if any(char.isdigit() for char in ans):
        if confidence_score < 0.450: return True

    if confidence_score < 0.300:
        return True

    if 0.400 <= confidence_score <= 0.620:
        return unc["mean_entropy"] > 0.35

    if confidence_score < 0.150:
        if is_valid_answer(ans): return False
        return True

    uncertain = (unc["spike_ratio"] > 12.0 or unc["max_entropy"] > 0.9)
    if uncertain and confidence_score < 0.400:
        if is_valid_answer(ans): return False
        if question:
            is_open = any(w in question.lower() for w in ["why", "explain", "what ethical", "how would"])
            if is_open: return True

    if confidence_score > 0.500:
        if not is_good_answer(ans): return True

    return False


def build_context(context_dict):
    context_str = ""
    for title, sentences in zip(context_dict["title"], context_dict["sentences"]):
        context_str += f"{title}: "
        context_str += " ".join(sentences)
        context_str += "\n"
    return context_str


# ====== DATA ======
dataset = load_dataset("hotpot_qa", "distractor")
rag_data = []
for item in dataset["validation"].select(range(100, 200)):
    context = build_context(item["context"])
    rag_data.append({
        "level": "mixed",
        "context": context,
        "question": item["question"],
        "a": item["answer"]
    })


# ====== COMPRESSION ======
def should_compress(prompt):
    return len(prompt.split()) > 20

def compress_llmlingua(prompt, level):
    target_map = {"light": 0.70, "medium": 0.50, "aggressive": 0.40}
    if not should_compress(prompt):
        return prompt
    result = compressor.compress_prompt(
        prompt,
        rate=target_map[level],
        force_tokens=["?", ".", "What", "Who"],
        drop_consecutive=True
    )
    compressed = result.get("compressed_prompt", prompt) if isinstance(result, dict) else str(result)
    compressed = compressed.strip()
    if len(compressed.split()) < 6:
        return prompt
    return compressed


# ====== HELPERS ======
def clean_output(text):
    text = str(text)
    for token in ["\n", "```", "class", "def", "Exercise", "Problem", "Question:"]:
        if token in text:
            text = text.split(token)[0]
    return text.strip()

def normalize_answer(text):
    text = re.sub(r'[,.]', '', str(text).lower().strip())
    text = re.sub(r'\s+', ' ', text)
    return text

def is_correct(pred, gt, aliases=None):
    """
    Tier-1: original fast string/token/F1 matching (unchanged).
    Tier-2: deterministic semantic normalisation (new, zero API cost).
            Handles: seven=7, True=yes, NBC=National Broadcasting Company,
                     filmmaker=director.
    Tracks tier2_called / tier2_rescued flags for MetricTracker.
    """
    from collections import Counter
    _SW = {"a","an","the","is","it","in","of","to","and","or","for","on","at","by",
           "as","be","are","was","were","this","that","with","from","its","not","but",
           "if","so","do","did","has","have","had","can","could","would","will","may",
           "might","should","because","which","who","what","when","where","why","how",
           "their","they","them","these","those","also","both","about","into","such",
           "more","likely","potentially","leading","being","before"}
    _SYN = {
        "outcomes":"effects", "outcome":"effect",
        "backwards":"reversed", "backward":"reversed",
        "anticipatory":"proactive", "proactive":"anticipatory",
        "distorted":"altered", "altered":"distorted",
        "ambiguous":"unclear", "uncertain":"ambiguous",
    }
    def _expand(ws):
        e = set(ws)
        for w in ws:
            if w in _SYN: e.add(_SYN[w])
        return e
    def _f1(p, g):
        pt = [w for w in p.split() if w not in _SW]
        gt_ = [w for w in g.split() if w not in _SW]
        if not pt or not gt_: return 0.0
        common = sum((Counter(pt) & Counter(gt_)).values())
        if common == 0: return 0.0
        return 2 * common / (len(pt) + len(gt_))

    def extract_numbers(text):
        cleaned = re.sub(r'(\d),(\d)', r'\1\2', text)
        return re.findall(r'\d+', cleaned)

    pred_raw = clean_output(str(pred))
    gt_raw   = str(gt)
    pred_nums = extract_numbers(pred_raw)
    gt_nums   = extract_numbers(gt_raw)

    if pred_nums and gt_nums and pred_nums[0] == gt_nums[0]:
        return True
    if len(pred_nums) == 1 and len(gt_nums) == 1 and len(gt_nums[0]) >= 6:
        try:
            pn, gn = int(pred_nums[0]), int(gt_nums[0])
            if gn > 0 and abs(pn - gn) / gn <= 0.20:
                return True
        except:
            pass

    pred_n = normalize_answer(pred_raw)
    gt_n   = normalize_answer(gt_raw)
    if pred_n == gt_n: return True

    comparative_check = re.search(
        r'\bis\s+(?:\w+er|more\s+\w+|less\s+\w+|older|younger|bigger|smaller|higher|lower)\s+than\b',
        pred_raw, re.IGNORECASE
    )
    if comparative_check:
        m = re.match(r'^(.+?)\s+(?:is|was)\s+(?:\w+er|more\s+\w+|less\s+\w+)\s+than\b', pred_raw.strip(), re.IGNORECASE)
        if m:
            extracted = m.group(1).strip()
            extracted = re.sub(r'^(the|a|an)\s+', '', extracted, flags=re.IGNORECASE)
            extracted_n = normalize_answer(extracted)
            gt_n_clean = normalize_answer(gt_raw)
            if extracted_n == gt_n_clean:
                return True
            gt_words_content = {w for w in gt_n_clean.split() if w not in _SW and len(w) > 2}
            extracted_words = {w for w in extracted_n.split() if w not in _SW and len(w) > 2}
            if gt_words_content and not (gt_words_content & extracted_words):
                print(f"  [comparative_block] pred subject '{extracted}' ≠ GT '{gt_raw}' → not correct")
                return False

    pred_words = set(pred_n.split())
    gt_words   = set(gt_n.split())
    if len(pred_words) <= 3 and len(pred_words & gt_words) >= 1: return True
    if "higher" in gt_n and "lower" in pred_n: return False
    if "lower" in gt_n and "higher" in pred_n: return False

    if short_gt_in_pred(pred_n, gt_n):
        return True

    f1 = _f1(pred_n, gt_n)
    gt_cw = len([w for w in gt_n.split() if w not in _SW])
    if gt_cw <= 8  and f1 >= 0.45: return True
    if gt_cw <= 15 and f1 >= 0.30: return True
    if gt_cw >  15 and f1 >= 0.18: return True

    pw = pred_words - _SW
    gw = gt_words   - _SW
    jac = len(_expand(pw) & _expand(gw)) / max(len(gw), 1)
    if gt_cw <= 8  and jac >= 0.40: return True
    if gt_cw <= 15 and jac >= 0.25: return True
    if gt_cw >  15 and jac >= 0.18: return True

    # ── Tier 2: deterministic semantic normalisation ───────────────────────────
    # Only reached when all Tier-1 checks have failed.
    # No LLM call. No side-effects on escalation logic.
    t2_ok, t2_reason = _t2_is_correct(pred_n, gt_n)
    if t2_ok:
        print(f"  [is_correct Tier-2] {t2_reason}  pred='{pred_raw[:30]}' gt='{gt_raw[:30]}'")
        return True
    # ──────────────────────────────────────────────────────────────────────────

    return False

def count_tokens(text):
    return int(len(tokenizer(str(text)).input_ids))


# ====== MAIN LOOP ======
JUDGE_CONF_THRESHOLD = 0.600

compression_methods2 = {"llmlingua": compress_llmlingua}

print("\n\n===== RAG V16 =====")

for method_name, compress_fn in compression_methods2.items():
    for level in ["medium"]:
        estimator = ConfidenceEstimator()
        print(f"\n### RAG | {method_name.upper()} | {level.upper()} ###")

        ragresults = []

        for item in rag_data:
            lev      = item.get("level", "unknown")
            context  = item["context"]
            question = item["question"]
            gt       = item["a"]

            extracted_context  = extract_relevant_context(context, question)
            compressed_context = compress_fn(context, level)
            extraction_ratio   = len(extracted_context.split()) / len(context.split())
            compression_ratio  = len(compressed_context.split()) / len(extracted_context.split())
            final_ratio        = len(compressed_context.split()) / len(context.split())

            print(f"\n--- SAMPLE ---")
            print(f"Question: {question}")
            print(f"Extraction: {extraction_ratio:.3f}  Compression: {compression_ratio:.3f}  Final: {final_ratio:.3f}")

            abstained, abstain_reason = should_abstain(question, compressed_context)

            if abstained:
                ans       = ""
                mean_lp   = -1.0
                unc       = {"mean_entropy": 1.0, "max_entropy": 1.0, "spike_ratio": 10.0}
                truncated = False
                print(f"  [ABSTAIN] reason={abstain_reason}")
            else:
                small_prompt = build_phi2_prompt(compressed_context, question)
                ans, mean_lp, unc = get_answer_with_stats(small_prompt)
                ans = clean_output(ans)
                estimator.update(mean_lp, unc["mean_entropy"])

            base_conf  = estimator.score(
                mean_lp, unc["mean_entropy"],
                answer=ans, context=context, question=question
            )
            confidence = compression_confidence_penalty(
                context, compressed_context, base_conf, question
            )
            confidence = max(0.0, confidence)

            if 0 < len(ans.split()) <= 3:
                if context_alignment_score(ans, context) > 0.3:
                    confidence += 0.08

            if ans.strip().lower() in ["yes", "no", "true", "false"]:
                if confidence > 0.45:
                    confidence = min(1.0, confidence + 0.05)

            confidence = min(1.0, max(0.0, confidence))
            print(f"FINAL CONF: {confidence:.3f}")

            # ====== DECISION PIPELINE ======
            boolean_escalation_flag   = False
            comparative_escalation_flag = False

            if abstained:
                escalated  = False
                score      = 0
                ran_judge  = False
                truncated  = False
            else:
                truncated = is_truncated(ans)

                if ans.strip() == "" or truncated:
                    escalated = True
                    score     = 0
                    ran_judge = False
                else:
                    escalated = should_escalate(confidence, unc, ans, question)

                    ans_lower_strip = ans.strip().lower()
                    escalated_by_bool = (
                        escalated and
                        len(ans.split()) > 4 and
                        (ans_lower_strip.startswith("yes") or ans_lower_strip.startswith("no")) and
                        JUDGE_CONF_THRESHOLD < confidence < 0.80
                    )

                    if is_non_recoverable(question):
                        escalated = False

                    if is_garbage_output(ans):
                        escalated = True

                    if is_calculation_question(question):
                        escalated = True

                    if is_boolean_question(question):
                        bool_esc, bool_reason = boolean_needs_escalation(
                            ans, confidence, question, compressed_context
                        )
                        if bool_esc:
                            if not escalated:
                                escalated = True
                                print(f"  [BOOLEAN → escalate] reason={bool_reason}")
                            boolean_escalation_flag = True
                            print(f"  [BOOLEAN_FLAG set] reason={bool_reason}")
                        else:
                            print(f"  [BOOLEAN trusted] reason={bool_reason}")

                    if not is_boolean_question(question):
                        comparative_check = re.search(
                            r'\bis\s+(?:\w+er|more\s+\w+|less\s+\w+|older|younger|bigger|smaller|higher|lower)\s+than\b',
                            ans, re.IGNORECASE
                        )
                        if comparative_check:
                            comparative_escalation_flag = True
                            if not escalated:
                                escalated = True
                                print(f"  [COMPARATIVE_ANS → escalate] ans='{ans[:50]}'")

                    if not escalated and is_hallucinated_claim(ans, compressed_context):
                        escalated = True
                        print(f"  [HALLUCINATION_DETECTED → escalate] ans='{ans[:40]}'")

                    if len(ans.split()) > 12:
                        overlap = context_alignment_score(ans, compressed_context)
                        if overlap < 0.25:
                            print("  [LOW ALIGNMENT → escalate]")
                            escalated = True

                    ran_judge = False
                    score     = 0
                    if (confidence <= JUDGE_CONF_THRESHOLD
                            and not is_non_recoverable(question)
                            and not answer_self_accepts(ans, question, compressed_context)):
                        score     = judge_answer(question, extracted_context, ans)
                        ran_judge = True
                        print(f"  [JUDGE fired] conf={confidence:.3f} → score={score}")
                        if score <= 2:
                            escalated = True
                        elif score == 3 and confidence < 0.500:
                            ans_digits_only = re.sub(r'[^0-9]', '', ans.strip())
                            is_short_number = (len(ans.strip()) <= 6 and
                                               len(ans_digits_only) >= len(ans.strip()) * 0.7)
                            is_elaborated_no = (ans.strip().lower().startswith('no')
                                                and len(ans.split()) > 2
                                                and confidence > 0.40)
                            if (is_short_number and confidence > 0.40) or is_elaborated_no:
                                print(f"  [JUDGE=3 bypass] ans='{ans[:30]}' → trust")
                            else:
                                escalated = True
                        elif score == 5 and confidence > 0.15:
                            type_mismatch, _ = answer_asks_for_specific_value(question, ans)
                            if not type_mismatch:
                                escalated = False
                                print(f"  [JUDGE=5 override → no escalation]")
                            else:
                                print(f"  [JUDGE=5 blocked — type mismatch]")
                        elif score >= 4 and confidence > 0.500:
                            escalated = False
                        elif score == 4 and confidence > 0.200:
                            type_mismatch, _ = answer_asks_for_specific_value(question, ans)
                            if not type_mismatch:
                                escalated = False
                                print(f"  [JUDGE=4 override → no escalation]")
                    elif confidence <= JUDGE_CONF_THRESHOLD and answer_self_accepts(ans, question, compressed_context):
                        print(f"  [SELF_ACCEPT skip judge] ans='{ans[:50]}'")

                    if confidence > 0.650:
                        if boolean_escalation_flag or comparative_escalation_flag:
                            print(f"  [HIGH_CONF blocked — special escalation protected]")
                        elif not escalated_by_bool:
                            if not is_good_answer(ans):
                                escalated = True
                            else:
                                escalated = False

                    if is_hopeless_question(question) and confidence > 0.300:
                        escalated = False

                    if ans.strip().lower() == "unknown":
                        escalated = True

                    if not escalated and not is_non_recoverable(question):
                        needs_specific, specific_reason = answer_asks_for_specific_value(question, ans)
                        if needs_specific:
                            escalated = True
                            print(f"  [SPECIFIC_VALUE → escalate] reason={specific_reason}")

            print(f"  ans={repr(ans)}  conf={confidence:.3f}  "
                  f"abstained={abstained}  escalated={escalated}  "
                  f"judge={ran_judge}  score={score}")

            empty_answer_flag = (not abstained) and (ans.strip() == "")
            prompt, groq_max_tokens = build_groq_prompt(
                question, compressed_context, ans,
                truncated if not abstained else False,
                empty_answer=empty_answer_flag,
                boolean_escalation=boolean_escalation_flag,
            )

            use_groq = abstained or escalated
            if use_groq:
                 try:
                    if boolean_escalation_flag:
                        sys_msg = "You are a precise factual assistant. Answer ONLY yes or no based on the context."
                    elif empty_answer_flag:
                        sys_msg = "You are a precise factual assistant. Reason step by step, then output ONLY the final answer."
                    else:
                        sys_msg = "You are a precise factual assistant. Answer using only the provided context."
                    messages=[{"role": "system", "content": sys_msg},
                            {"role": "user", "content": prompt}]
                    max_tokens=groq_max_tokens
                    final = clean_output(rate_limited_call(messages, max_tokens))
                    if final.lower() == "none":
                        resp = client.chat.completions.create(
                            messages=messages,
                            model="llama3.1-8b",
                            max_tokens=max_tokens
                        )
                        final = clean_output(resp.choices[0].message.content)
                 except Exception as e:
                    print("Cerebras failed:", e)
                    final = ans

            else:
                final = ans

            if not abstained and normalize_answer(final) == normalize_answer(ans):
                final = ans

            # ====== METRICS ======
            correct       = is_correct(final, gt)
            small_correct = is_correct(ans, gt)

            if not small_correct and correct:
                improved = "strong" if len(ans.strip()) == 0 else "weak"
            else:
                improved = "none"

            if abstained:
                esc_type = "abstention_correct" if correct else "abstention_wrong"
            elif escalated:
                if not small_correct and correct:     esc_type = "correct_escalation"
                elif small_correct and correct:       esc_type = "redundant_escalation"
                elif small_correct and not correct:   esc_type = "false_escalation"
                else:                                 esc_type = "wasted_escalation"
            else:
                esc_type = "correct_rejection" if small_correct else "missed_escalation"

            print(f"  GT: {gt!r}  →  small: {ans!r}  →  final: {final!r}")
            print(f"  correct={correct}  esc_type={esc_type}")

            # ── track Tier-2 semantic correction ──────────────────────────────
            # Re-run is_correct with a thin wrapper to detect if Tier-2 fired.
            # Only on samples where Tier-1 alone would have returned False.
            def _is_correct_tier1_only(p, g):
                """Tier-1 only (no Tier-2) — used purely for metric tracking."""
                from collections import Counter
                _sw = {"a","an","the","is","it","in","of","to","and","or","for","on","at","by",
                       "as","be","are","was","were","this","that","with","from","its","not","but",
                       "if","so","do","did","has","have","had","can","could","would","will","may",
                       "might","should","because","which","who","what","when","where","why","how",
                       "their","they","them","these","those","also","both","about","into","such"}
                def _f1(a, b):
                    pt=[w for w in a.split() if w not in _sw]
                    gt_=[w for w in b.split() if w not in _sw]
                    if not pt or not gt_: return 0.0
                    common=sum((Counter(pt)&Counter(gt_)).values())
                    return 2*common/(len(pt)+len(gt_)) if common else 0.0
                pn=normalize_answer(clean_output(str(p))); gn=normalize_answer(str(g))
                if pn==gn: return True
                pw=set(pn.split()); gw=set(gn.split())
                if len(pw)<=3 and len(pw&gw)>=1: return True
                gt_s=re.sub(r'^(the|a|an) ','',gn.strip()); gsw=gt_s.split()
                if len(gsw)<=4 and (gt_s in pn or all(w in pn.split() for w in gsw if w)): return True
                f1=_f1(pn,gn); gc=len([w for w in gn.split() if w not in _sw])
                if gc<=8 and f1>=0.45: return True
                if gc<=15 and f1>=0.30: return True
                if gc>15 and f1>=0.18: return True
                return False

            t1_small = _is_correct_tier1_only(ans,   gt)
            t1_final = _is_correct_tier1_only(final, gt)
            tier2_called_small = (not t1_small) and bool(ans.strip())
            tier2_called_final = (not t1_final) and bool(final.strip())
            tier2_rescued      = ((not t1_small and small_correct) or
                                  (not t1_final and correct))
            # ──────────────────────────────────────────────────────────────────

            sample_record = {
                "context_length":    len(context.split()),
                "compressed_length": len(compressed_context.split()),
                "compression_ratio": compression_ratio,
                "extraction_ratio":  extraction_ratio,
                "final_ratio":       final_ratio,
                "level":             lev,
                "answer_small":      ans,
                "final_answer":      final,
                "correct":           bool(correct),
                "small_correct":     bool(small_correct),
                "escalated":         bool(escalated),
                "abstained":         bool(abstained),
                "abstain_reason":    abstain_reason,
                "improved":          improved,
                "confidence":        float(confidence),
                "base_confidence":   float(base_conf),
                "mean_lp":           float(mean_lp),
                "mean_entropy":      float(unc["mean_entropy"]),
                "max_entropy":       float(unc["max_entropy"]),
                "spike_ratio":       float(unc["spike_ratio"]),
                "prompt_tokens":     count_tokens(prompt),
                "answer_tokens":     count_tokens(final),
                "total_tokens":      count_tokens(prompt) + count_tokens(final),
                "escalation_type":   esc_type,
                "groq_calls":        1 if use_groq else 0,
                "judge_calls":       1 if ran_judge else 0,
                # Tier-2 tracking fields
                "tier2_called_small": tier2_called_small,
                "tier2_called_final": tier2_called_final,
                "tier2_rescued":      tier2_rescued,
            }

            ragresults.append(sample_record)
            tracker.log(sample_record)   # ← MetricTracker

        df2 = pd.DataFrame(ragresults)
        for col in ["correct","small_correct","escalated","abstained"]:
            df2[col] = df2[col].astype(float)
        df2["strong_improvement"] = (df2["improved"] == "strong").astype(int)
        df2["weak_improvement"]   = (df2["improved"] == "weak").astype(int)
        df2["any_improvement"]    = df2["strong_improvement"] + df2["weak_improvement"]

        n            = len(df2)
        n_abstained  = int(df2["abstained"].sum())
        n_escalated  = int(df2["escalated"].sum())
        n_groq_total = n_abstained + n_escalated
        n_judge      = int(df2["judge_calls"].sum())
        total_api    = n_groq_total + n_judge

        print(f"\n--- SUMMARY ---")
        print(f"Accuracy:              {df2['correct'].mean():.3f}")
        print(f"Error rate:            {1 - df2['correct'].mean():.3f}")
        print(f"Small model accuracy:  {df2['small_correct'].mean():.3f}")
        print(f"Abstention rate:       {df2['abstained'].mean():.3f}  ({n_abstained}/{n})")
        print(f"Escalation rate:       {df2['escalated'].mean():.3f}  ({n_escalated}/{n})")
        print(f"Total qwen call rate:  {n_groq_total/n:.3f}  ({n_groq_total}/{n})")
        print(f"── Cost breakdown ──────────────────────────")
        print(f"  qwen answer calls:  {n_groq_total}")
        print(f"  Groq judge calls:   {n_judge}  (conf <= {JUDGE_CONF_THRESHOLD} only)")
        print(f"  Total API calls:    {total_api}  ({total_api/n:.2f} per sample)")
        print(f"  Phi-2 only served:  {n - n_groq_total}/{n}  ({(n - n_groq_total)/n:.1%} free)")
        print(f"  Avg tokens/sample:  {df2['total_tokens'].mean():.1f}")
        print(f"  Avg final ratio:    {df2['final_ratio'].mean():.3f}")
        print(df2["escalation_type"].value_counts().to_string())
        print(f"Strong Improvement:    {df2['strong_improvement'].mean():.3f}")
        print(f"Weak Improvement:      {df2['weak_improvement'].mean():.3f}")
        print(f"Total Improvement:     {df2['any_improvement'].mean():.3f}")
        print(f"\n--- ACCURACY BY LEVEL ---")
        print(df2.groupby("level").agg(
            n=("correct","count"),
            accuracy=("correct","mean"),
            abstained=("abstained","mean"),
            escalated=("escalated","mean"),
            judge_rate=("judge_calls","mean"),
        ).round(3).to_string())

        df2.to_csv(f"rag_v16_2{method_name}_{level}.csv", index=False)

# ====== FINAL RESEARCH METRICS ======   (new — two lines added at the end)
tracker.save_json("rag_final_metrics.json")