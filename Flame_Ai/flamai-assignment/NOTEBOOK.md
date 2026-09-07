# Lab Notebook: Multilingual Tokenizer & Serving Capacity Audit (`NOTEBOOK.md`)

*This notebook is a chronological lab journal recording hypotheses, experimental setups, empirical results, revisions, and dead ends.*

---

## Day 1 — Initial Setup & System Orientation

### 09:30 — Initial Workspace Inspection & Artifact Extraction
- **Goal**: Unpack `starter_kit (1).zip` and audit existing files (`fertility.py`, `REPORT_v0.md`, `bench/model_spec.md`, `bench/bench_log.csv`).
- **Initial Observation**: `REPORT_v0.md` asserts Hindi tokenization is 5.89× more expensive than English based on `gpt2` tokenizer results on a 10-line sample.
- **Hypothesis H1**: The 6× cost multiplier claim for Hindi is an artifact of tokenizer vocabulary bias (`gpt2`) rather than an intrinsic property of the Hindi language or script.

---

### 10:15 — Audit of `fertility.py` Code Mechanics
- **Experiment E1**: Inspected `fertility.py` line-by-line to identify bugs and metric assumptions.
- **Findings**:
  1. `line.split(" ")` (line 62): Splits strictly on single ASCII space `" "`. Fails to strip punctuation attached to words, and multiple spaces produce empty strings `""`, inflating word count.
  2. Macro-Averaging (line 67): Calculates `len(tokens) / len(words)` per line and takes `sum(r)/N`. Short lines with few words carry disproportionated weight.
  3. Denominator `len(line)` (line 63): Uses Python code point length, treating combining vowel signs (matras in Devanagari) as separate characters.
  4. `unicodedata.normalize("NFC", line)` (line 49): Verified as standard composition hygiene. Tested removal on sample text; produced 0 change in tokens. **Flagged as harmless**.

---

### 11:30 — Initial Code Experiment & Macro vs. Micro Average Test
- **Experiment E2**: Ran legacy analysis script on `corpus_sample/eng_sample.txt` and `corpus_sample/hin_sample.txt` comparing macro-average vs micro-average.
- **Results**:
  - Legacy Macro-average Hindi fertility: **7.45 tok/word**.
  - Micro-average (Ratio of Totals `sum(tokens)/sum(words)`): **6.91 tok/word**.
  - **Delta**: Macro-averaging inflates Hindi fertility by **+0.54 tok/word (+7.8%)** on sample data.

---

## Day 2 — Parallel Corpus Construction & Multilingual Tokenizer Audit

### 09:00 — Building A1 Evaluation Corpus (FLORES-101)
- **Goal**: Construct a 200-sentence parallel evaluation corpus across 4 languages: English (`eng`), Hindi (`hin`), Kannada (`kan`), Telugu (`tel`).
- **Implementation**: Created `partA/build_corpus.py` to query `gsarti/flores_101` parallel dataset via HuggingFace Datasets Server.
- **Result**: Successfully downloaded and aligned 200 parallel sentences per language. Saved to `partA/corpus/`.

---

### 10:45 — Dead End: Gated Model Tokenizer Attempt
- **Hypothesis H2**: Testing `google/gemma-2-2b` tokenizer will provide a modern LLM tokenization baseline.
- **Execution**: Attempted to load `google/gemma-2-2b` via `AutoTokenizer.from_pretrained()`.
- **Result (Dead End)**: Failed with HTTP 401 Unauthorized / Gated Repo Error (requires HuggingFace token login).
- **Revision**: Switched to open public tokenizers: `xlm-roberta-base` (multilingual BPE) and `bert-base-multilingual-cased` (`mBERT`), plus `gpt2` for baseline comparison.

---

### 13:15 — Corrected Multilingual Tokenizer Benchmark (A3)
- **Experiment E3**: Ran `partA/audit_tokenizer.py` on the 200-sentence parallel corpus across all tokenizers and denominators.
- **Results Summary**:

| Tokenizer | Language | Micro tok/sentence | Micro tok/word | tok/grapheme | tok/byte |
|---|---|---|---|---|---|
| `gpt2` | ENG | 27.23 | 1.24 | 0.203 | 0.203 |
| `gpt2` | HIN | 202.32 | 7.83 | 2.199 | 0.594 |
| `gpt2` | KAN | 372.10 | 22.20 | 4.036 | 0.977 |
| `gpt2` | TEL | 352.26 | 20.60 | 4.045 | 0.987 |
| `xlm-roberta` | ENG | 31.05 | 1.42 | 0.232 | 0.232 |
| **`xlm-roberta`** | **HIN** | **38.98** | **1.51** | **0.424** | **0.114** |
| `xlm-roberta` | KAN | 42.05 | 2.51 | 0.456 | 0.110 |
| `xlm-roberta` | TEL | 40.59 | 2.37 | 0.466 | 0.114 |

- **Key Takeaway**: Under `xlm-roberta-base`, Hindi token requirement per parallel sentence is **38.98 tokens vs 31.05 tokens for English** — a ratio of only **1.26×** (NOT 6×!). Dravidian languages (`kan`, `tel`) require only **1.35×** and **1.31×** English token counts per sentence!

---

## Day 3 — Serving Capacity Reconciliation (Part B)

### 09:30 — Analytical KV Cache Derivation (B1)
- **Calculation**:
  - Model: FLM-4B-Instruct ($L=28, n_{\text{kv}}=8, d_k=128$, FP16).
  - KV bytes/token = $2 \times 28 \times 8 \times 128 \times 2 = 114,688 \text{ bytes} = \mathbf{112 \text{ KiB/token}}$.
  - VRAM Pool: $0.92 \times 24 \text{ GB} = 22.08 \text{ GiB}$. Subtracting 8.4 GB weights & 1.6 GB overhead leaves **13.708 GB** KV pool.
  - Per sequence KV (4096 tokens) = $4096 \times 114,688 = 448 \text{ MiB}$.
  - Max concurrent sequences = $\lfloor 13.708 / 0.448 \rfloor = \mathbf{29 \text{ sequences}}$.
- **Empirical Check**: In `bench_log.csv`, batch 24 has 0 preemptions (`kv_cache_util` = 0.93). Batch 32 has 7 preemptions (`kv_cache_util` = 0.97). This confirms our mathematical capacity boundary of 29 sequences.

---

### 11:00 — Long-Context Throughput Anomaly Analysis (B2 & B3)
- **Analysis of `reported_tok_s`**:
  - Uncovered why Section 2 of `REPORT_v0` falsely claimed long prompts give better throughput.
  - `reported_tok_s` counts total prefill + gen tokens. At batch 16 long prompt (3584 prompt, 512 gen), 87.5% of tokens are prompt prefill tokens processed in 1 pass.
  - Actual output generation rate (**Goodput**) at batch 24 long prompt:
    $$\text{Goodput} = \frac{24 \text{ reqs} \times 512 \text{ gen}}{61.16 \text{ s}} = \mathbf{200.92 \text{ output tok/s}}$$
  - At batch $\ge 32$, VRAM exhaustion forces 7 to 23 sequence preemptions, causing wall-clock time and latency to explode.

---

## Day 4 — Strategic Decision Memo (Part C)

### 14:00 — Strategic Options Assessment
- Evaluated 3 paths for casual Indic response generation under tight constraints (1 A100-80GB, 1 reviewer for 10h/wk, 3 weeks, $0 budget).
- Formulated Path (a) QLoRA SFT recommendation with complete arithmetic for reviewer capacity (240 golden test pairs evaluated over 20 hours) and training compute (<15 min per run).
- Established explicit Kill Criterion (abandon Path (a) if Hindi win rate < 55% by Day 5) and Day 1 experiment.

---

## Day 5 — Final Artifact Packaging & Repository Polish

### 10:00 — Visualization & Repository Verification
- Executed `partA/plot_results.py` to generate high-resolution `partA/tokenizer_audit.png`.
- Compiled `partA_memo.md`, `partB_analysis.md`, `memo.md`, `NOTEBOOK.md`, `AI_USAGE.md`, and `README.md`.
- Verified directory structure and initialized Git repository for final submission.
