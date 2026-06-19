# Compression-Aware Cascade RAG

**A cost-efficient multi-model inference framework that combines prompt compression with confidence-based model cascading — and shows the two amplify each other instead of just stacking.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Phi--2](https://img.shields.io/badge/Small%20Model-Phi--2%20(2.7B)-orange)]()
[![Qwen3](https://img.shields.io/badge/Large%20Model-Qwen--3--235B-purple)]()
[![Dataset](https://img.shields.io/badge/Dataset-HotpotQA%20Distractor-green)]()

---

## TL;DR

Running every query through a large LLM is accurate but expensive. Running everything through a small LLM is cheap but wrong too often. The usual fixes — prompt compression and model cascading — are normally studied in isolation. This project couples them: **compressed context is fed to a small model (Phi-2), and the small model's token-level uncertainty under compression is used as the routing signal for escalating to a large model (Qwen-3-235B).**

The result, on 200 HotpotQA (distractor split) questions:

| Configuration | Accuracy | Avg. Tokens/Query | Escalation Rate |
|---|---:|---:|---:|
| Small model only (uncompressed) | 53.5% | 1446.8 | 0% |
| Small model only (compressed) | 49.0% | 681.1 | 0% |
| Large model only (uncompressed) | 81.0% | 1411.3 | 100% |
| Cascade only (no compression) | 72.5% | 1427.6 | 53% |
| **Compression + Cascade (this work)** | **80.0%** | **678.2** | **46.5%** |

**80.0% accuracy — within 1.5 points of the large-model ceiling — at 52% fewer tokens than any uncompressed baseline, and 38% as many large-model API calls as a "send everything to the big model" approach.**

The accuracy gain from combining the two methods (72.5% → 80.0%) is *larger* than what the two components would predict if their effects simply added up — evidence of a genuine compression–routing coupling, not just two independent savings stacked together. See [Analysis](#why-this-works-the-compressionrouting-coupling) below.

---

## Why this exists

Most RAG cost-reduction work picks one lane:

- **Compression researchers** (LLMLingua-2, ProCut, StyleCompress) shrink the prompt before inference, but evaluate it as a standalone optimization — they don't ask what compression does to a *downstream* model's confidence.
- **Cascade researchers** (FrugalGPT, CascadeServe) route cheap queries to cheap models, but every tier still sees the full, uncompressed prompt — so the cheap tier is still expensive to run at scale.

No prior work treats compression and routing as a **coupled system**. This project does, and finds that compression isn't just a token-count optimization — it's a *signal amplifier* for the cascade router. Stripping context makes the small model's uncertainty a much more reliable predictor of whether it actually knows the answer.

---

## How it works

```
Query → Retrieve(D) → Extract(D′) → Compress(D′) → Phi-2 → Confidence Router → Accept / Escalate → Qwen-3-235B
```

1. **Retrieve & extract** — relevant passages are pulled and the most query-relevant sentences are kept (`extract_relevant_context`).
2. **Compress** — the extracted context is compressed with **LLMLingua-2** (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`) at a target retention ratio of `r ≈ 0.50`. The question string itself is never compressed — it's force-retained.
3. **Small model inference** — **Phi-2** (2.7B params) generates an answer locally, with full token-level log-probabilities and entropy captured during generation.
4. **Confidence scoring** — a calibrated score combines:
   - mean log-probability and mean entropy (online-normalized via **Welford's algorithm**, weighted 0.3 / 0.7)
   - a **compression-quality penalty** for excessive compression, poor entity retention, or lost question words
   - question-type-specific heuristics (boolean grounding, comparative-entity validation, truncation/hallucination/garbage detection)
5. **Routing** — `Route(yₛ) = 1` if confidence `< τ = 0.58` **or** any hard heuristic fires (calculation question, ungrounded boolean claim, low context-alignment). Otherwise the small model's answer is returned directly.
6. **Abstention** — multi-hop relational queries (`"the X who Y"` patterns), calculation questions, and fictional/hypothetical prompts skip Phi-2 entirely and go straight to the large model — these are patterns Phi-2 fails at systematically, not stochastically.
7. **Escalation** — routed queries go to **Qwen-3-235B-A22B** via the Cerebras Cloud API (with Gemini 3.1 Flash Lite as fallback), receiving the *same compressed context* Phi-2 saw — preserving the token savings even on escalated queries.
8. **Judge (optional, rare)** — for borderline cases (`confidence ≈ τ`, ~12 of 200 queries), **Gemma-4-26B** scores the small model's answer 1–5 against the context as a tie-breaker.

A full architecture diagram is in [`docs/pipeline.png`](docs/pipeline.png).

---

## Results

### The headline trade-off

This system sits in the upper-left "ideal zone" — high accuracy, low token cost — that no individual baseline configuration reaches:

| Metric | Value |
|---|---:|
| Final accuracy | **80.0%** |
| Distance from large-model ceiling | **1.5 points** |
| Token reduction vs. uncompressed cascade | **52%** |
| Large-model API calls per query | **0.38** (vs. 1.0 for "always escalate") |
| Cost efficiency (Δaccuracy / API calls per sample) | **0.609** |

### Why neither method alone is enough

| Configuration | Accuracy | Tokens |
|---|---:|---:|
| No Compression, No Cascade | 53.5% | 1446.8 |
| Compression Only | 49.0% | 681.1 |
| Cascade Only | 72.5% | 1427.6 |
| **Full System** | **80.0%** | **678.2** |

Compression *alone* actually **hurts** the small model (53.5% → 49.0%) — Phi-2 leans on contextual redundancy to "complete the logic" in multi-hop reasoning, and compression removes exactly that redundancy. Cascading *alone* recovers 19 points of accuracy but doesn't save a single token, since every escalated query still carries the full 1,400-token context.

Combined, accuracy doesn't just average these effects — it **exceeds** what additive composition would predict (72.5% + cascade gain ≠ 80.0%). The mechanism: compression makes Phi-2's *failures more visible to the confidence router*, not just cheaper to attempt.

### Calibration & escalation quality

| Metric | Value |
|---|---:|
| Expected Calibration Error (ECE) | 0.281 |
| Escalation precision | 0.733 |
| Escalation recall | 0.668 |
| Escalation F1 | 0.692 |
| Abstention quality (accuracy on abstained queries) | 0.889 |
| Semantic rescue rate (Tier-2 answer matching) | 0.165 |

### Routing outcome breakdown (n=200)

| Outcome | Count | % |
|---|---:|---:|
| Correct Rejection | 76 | 38.0% |
| Correct Escalation | 49 | 24.5% |
| Missed Escalation | 31 | 15.5% |
| Redundant Escalation | 19 | 9.5% |
| Abstention (Correct) | 16 | 8.0% |
| Wasted Escalation | 6 | 3.0% |
| Abstention (Wrong) | 2 | 1.0% |
| False Escalation | 1 | 0.5% |

**70.5%** of all queries were routed correctly on the first decision (correct rejection + correct escalation + correct abstention).

### Accuracy by question type

| Type | n | Small Model Acc | Final Acc | Δ |
|---|---:|---:|---:|---:|
| Factual | 121 | 54.5% | 82.6% | +28.1 |
| Boolean | 41 | 46.3% | 75.6% | +29.3 |
| Multi-hop | 27 | 33.3% | 74.1% | +40.8 |
| Calculation | 11 | 9.1% | 72.7% | +63.6 |

Calculation questions show the largest gain — Phi-2 is unconditionally escalated on these because it gets them right only **9.1%** of the time, with no confidence signal reliable enough to selectively trust it.

---

## Why this works: the compression–routing coupling

The core empirical finding: **compression systematically increases entropy on questions the small model can't actually answer, while leaving easy-question entropy nearly untouched.**

- **Easy query** ("which country is Paris the capital of?") — heavily overdetermined by context. Even aggressive compression leaves enough anchoring tokens for Phi-2 to answer with low entropy. No escalation needed.
- **Hard multi-hop query** ("what was the profession of the director of the film discussed in the context?") — requires chaining 2–3 facts. Compression strips the bridging sentences between entities, spiking entropy. The router catches this and escalates correctly — *because* the information was removed, not despite it.

This means compression isn't purely a cost optimization — it's a **free difficulty classifier**. No separate query-complexity model was trained; the coupling falls out of how a small model's generation entropy responds to missing context.

This also turns the compression ratio into a **deployment control knob** with zero changes to the routing algorithm itself:

| Compression Level | Target r | Accuracy | Mean Tokens |
|---|---:|---:|---:|
| Light | 0.70 | 93.3%* | ~966 |
| Medium | 0.50 | 90.0%* | ~692 |
| Aggressive | 0.40 | 73.3%* | ~577 |

*\*30-sample pilot set used for threshold calibration, not the main 200-sample evaluation.*

Loosen `r` toward 0.70 to favor cost; tighten toward 0.40 to favor accuracy — without retraining or re-tuning anything else.

---

## Evaluation methodology

Accuracy isn't scored by naive string equality. The `is_correct()` function uses a two-tier procedure:

- **Tier 1** — exact match after normalization, numeric tolerance matching, comparative-answer subject extraction, token-level F1/Jaccard overlap against the ground truth.
- **Tier 2** — deterministic semantic equivalence for cases Tier 1 misses: number-word ↔ digit (`seven` = `7`), boolean synonyms (`True` = `yes`), broadcast abbreviations (`NBC` = `National Broadcasting Company`), occupation synonyms (`filmmaker` = `director`). Zero additional API cost — pure lookup tables, not an LLM call.

Tier 2 rescued **16.5%** of answers that Tier 1 would have falsely marked wrong.

Confidence calibration is tracked via **Expected Calibration Error (ECE)** across 10 confidence buckets, and escalation quality is reported as standard **precision/recall/F1** (treating "small model was wrong" as the positive class for escalation).

---

## Repository structure

```
.
├── rag_final.py              # Full pipeline: retrieval → compression → cascade → metrics
├── graphs/
│   └── generate_graphs.py    # Research-grade visualization suite (calibration curve,
│                              # escalation breakdown, cost-accuracy trade-off, etc.)
├── docs/
│   ├── pipeline.png           # Architecture diagram
│   └── paper.pdf              # Full write-up of methodology and results
├── results/
│   ├── rag_final_metrics.json # Full per-sample + aggregate metrics from a run
│   └── rag_final_*.csv        # Per-sample results table
└── README.md
```

## Quickstart

```bash
pip install transformers torch llmlingua datasets pandas google-generativeai cerebras-cloud-sdk

export CEREBRAS_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

python rag_final.py
```

This runs the full pipeline over a 200-question HotpotQA (distractor split) sample, prints a live per-sample trace, and writes:
- `rag_final_*.csv` — per-sample results
- `rag_final_metrics.json` — full metrics report (accuracy, ECE, escalation P/R/F1, cost efficiency, etc.)

Generate the research figures:

```bash
python graphs/generate_graphs.py
```

---

## Models used

| Role | Model | Access |
|---|---|---|
| Small model (Mₛ) | Phi-2 (2.7B) | Local, HuggingFace Transformers |
| Large model (Mₗ) | Qwen-3-235B-A22B | Cerebras Cloud API |
| Fallback large model | Gemini 3.1 Flash Lite | Google Generative AI API |
| Judge | Gemma-4-26B | Google Generative AI API |
| Compressor | LLMLingua-2 (multilingual BERT-base) | Local |

---

## Limitations

- **Sample size**: 200-question evaluation (±3.5 points at 95% CI). Full 7,405-question validation was cost-prohibitive given paid large-model API calls per escalation; treated as a directional study.
- **Large model also sees compressed context** on escalation — necessary for token savings, but means information lost during compression is unavailable at every tier, not just the small model.
- **Threshold tuning** (`τ = 0.58`, `r = 0.50`) was calibrated on a 30-sample pilot set and is task/domain-specific, not guaranteed to generalize.
- **Single benchmark**: HotpotQA distractor split only (multi-hop factual QA). Generalization to summarization, dialogue, or other generation tasks is untested.
- **Latency** from large-model API round-trips is not measured — token count is used as the cost proxy.

---

## Future directions

- **Adaptive compression ratio** — compress harder on long/complex queries, lighter on short ones, instead of a fixed `r = 0.50`.
- **Learned routing** — replace the hand-crafted confidence heuristic with a trained classifier over the same features plus question-type signals.
- **RL over the cascade policy** — treat `(τ, r)` as a joint action space with reward = accuracy gain − API cost.
- **Multi-tier cascades** — insert a mid-size model (e.g. 7B) between Phi-2 and Qwen-3-235B to avoid sending every escalated query straight to the largest, most expensive tier.

---

## Citation

If you use this work, please cite:

```bibtex
@article{amjad2025compression,
  title   = {Compression-Aware Cascade RAG: A Cost-Efficient Multi-Model
             Framework for Long-Context LLM Inference},
  author  = {Amjad, Ayman and G R, Asha},
  institution = {Department of Computer Science, BMS College of Engineering, Bangalore},
  year    = {2025}
}
```

---

## Acknowledgements

Built on [LLMLingua-2](https://github.com/microsoft/LLMLingua), [Phi-2](https://huggingface.co/microsoft/phi-2), and [Qwen-3](https://github.com/QwenLM/Qwen) via the [Cerebras Cloud API](https://cerebras.ai/). Evaluated on [HotpotQA](https://hotpotqa.github.io).
