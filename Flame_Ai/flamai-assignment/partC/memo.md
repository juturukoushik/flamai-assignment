# Strategic Decision Memo: Casual Indic Conversational Assistant (Part C)

**To:** Product & Executive Leadership  
**From:** Lead AI Systems Architect  
**Date:** September 7, 2026  
**Subject:** Technical Recommendation & Deployment Strategy for Conversational Indic Assistant  

---

## 1. Scenario & Core Constraints

**Objective:** Upgrade assistant output style to sound casual and natural across 6 target Indic languages (**Hindi, Kannada, Tamil, Telugu, Bengali, Marathi**) — moving away from formal textbook responses.

**Fixed Operating Constraints:**
1. Hardware: **1× NVIDIA A100 (80GB VRAM)** dedicated for 2 weeks.
2. Human Evaluation: **1 Native Speaker Reviewer** (covers Hindi + Kannada only) for **10 hours/week** (20 total evaluation hours across 2 weeks).
3. Delivery Timeline: **Launch review in 3 weeks**.
4. Financial Constraint: **$0 external API budget**.

---

## 2. Evaluation of Proposed Options

| Option | Engineering / Compute Feasibility | Serving Latency & Cost Impact | Reviewer Bottleneck Risk | Recommendation |
|---|---|---|---|---|
| **(a) SFT Pass on Synthetic Casual Pairs** | **High**: QLoRA fine-tuning of an open 8B model takes ~14h on A100-80GB. | **Zero Latency Overhead**: Single-model inference at runtime. | **Manageable**: Reviewer audits high-leverage eval sets. | **RECOMMENDED (PRIMARY)** |
| **(b) ≤1B Rewriter Model** | **Medium**: Training rewriter model feasible, but doubles inference calls. | **High Latency Penalty**: 2× forward passes; adds 150-250ms per request. | **High Risk**: Compounding errors hard to debug in 3 weeks. | **REJECTED** |
| **(c) Prompt Engineering Only** | **Low Barrier**: Fast initial setup. | **High Cost Penalty**: System prompt bloat (+500 tokens/req) increases TTFT & KV cache footprint. | **Unreliable**: Base model ignores casual style cues in Dravidian scripts. | **FALLBACK ONLY** |

---

## 3. Recommended Technical Plan: Path (a) QLoRA SFT Pipeline

### Key Technical Assumptions
- We select an open instruction-tuned multilingual backbone model (e.g. **Qwen-2.5-7B-Instruct** or **Llama-3.1-8B-Instruct**).
- We generate a synthetic dataset of 2,000 "casualized" conversational pairs per language (12,000 total pairs across 6 languages) using open local LLMs (e.g. Qwen-2.5-72B running quantized on local cluster or open models).

### Back-of-the-Envelope Arithmetic

1. **Training Compute**:
   - Fine-tuning an 8B model with QLoRA (rank $r=16$, alpha $\alpha=32$, sequence length 2048) on 12,000 pairs (average 300 tokens/pair = 3.6M tokens total).
   - Training speed on 1× A100-80GB: ~15,000 tokens/sec.
   - 3 epochs = 10.8M tokens = **~12 minutes per training run**.
   - Hyperparameter exploration (10 iterations): **< 5 hours total GPU time**, leaving 95% of A100 time free for inference batch generation.

2. **Reviewer Throughput & Statistical Power**:
   - Reviewer availability: 10 h/week × 2 weeks = **20 total hours**.
   - Verification speed: Reviewer audits paired outputs (Formal Baseline vs. Casual Candidate) at a rate of **12 pairs per hour**.
   - Total evaluation capacity: $20 \text{ hours} \times 12 \text{ pairs/hour} = \mathbf{240 \text{ golden test pairs}}$ (120 Hindi, 120 Kannada).
   - Sample size $n=120$ per language provides a 95% confidence interval ($\pm 7.5\%$ margin of error) to statistically validate preference wins.

3. **Serving Cost & Latency**:
   - Runtime latency overhead = **0 ms** (same parameter count as base model).
   - KV cache memory footprint per request remains unchanged.

---

## 4. Success Metrics & Quantitative Thresholds

- **Primary Success Metric**: Native speaker preference win rate over current baseline.
  - **Threshold**: $\ge \mathbf{70\% \text{ preference win rate}}$ on double-blind evaluation of 120 Hindi and 120 Kannada test prompts.
- **Safety / Quality Floor**:
  - Grammatical correctness degradation $\le \mathbf{3\%}$.
  - Toxicity / hallucination rate $\le \mathbf{1\%}$.

---

## 5. Kill Criterion & Contingency Trigger

> [!CAUTION]
> **KILL CRITERION**: If by **Day 5** (after initial QLoRA training run and baseline validation), the native reviewer preference win rate on Hindi sample prompts is **$< 55\%$**, OR if the rate of ungrammatical / corrupted script outputs exceeds **$5\%$**, **abandon Path (a) immediately**.

**Contingency Strategy (Day 6 onward)**:
Pivot to **Path (c) Enhanced In-Context Few-Shot Prompting**. Utilize the remaining 2 weeks to curate 5 pristine, native-verified casual exemplars per language, embedded dynamically via vector retrieval into the system prompt.

---

## 6. Day 1 Action Item: First Experiment

On **Day 1**, execute the following minimal experiment:
1. Sample **30 representative Hindi prompts** from user logs.
2. Generate outputs under 3 conditions: (i) Baseline system prompt, (ii) 3-shot Casual Prompting, (iii) Zero-shot Qwen-2.5-72B casual translation.
3. Submit the 90 anonymized outputs to the native reviewer during their first 2-hour slot to establish the **baseline formality score** and quantify the stylistic delta.
