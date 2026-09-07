# AI Usage Disclosure (`AI_USAGE.md`)

*Honest disclosure of where AI tools (Gemini / Claude / ChatGPT / Cursor) assisted, where they provided guidance, and where their suggestions were corrected through manual verification.*

---

## 1. Where AI Assisted & Streamlined Execution

- **Corpus Fetch Pipeline (`partA/build_corpus.py`)**: AI generated the boilerplate Python script utilizing `requests` to fetch parallel sentences from HuggingFace Datasets Server (`gsarti/flores_101`).
- **Data Visualization (`partA/plot_results.py`)**: AI generated `matplotlib` / `seaborn` plotting code to produce multi-bar comparison charts for token counts per parallel sentence and relative expansion ratios.
- **Mathematical Formula Structuring (`partB/partB_analysis.md`)**: AI formatted the LaTeX mathematical equations for KV cache memory footprint per token ($2 \times L \times n_{\text{kv\_heads}} \times d_k \times 2$) and VRAM sequence capacity limits.
- **Markdown Document Boilerplate**: AI assisted in drafting initial structural layouts for decision memos and lab notebook entries.

---

## 2. Where AI Misled or Hallucinated (and How It Was Corrected)

- **Gated Hugging Face Repository Download**: AI initially attempted to load `google/gemma-2-2b` tokenizer without accounting for HuggingFace authentication gates, resulting in HTTP 401 GatedRepoErrors. **Correction**: Manually caught the error, diagnosed HF Hub auth requirements, and updated the benchmark script to use open, ungated models (`xlm-roberta-base` and `bert-base-multilingual-cased`).
- **Plotting Script Dictionary Key Mismatch**: AI generated plotting code assuming a fixed list of tokenizer keys (`qwen2.5-7b`), which threw a `KeyError` when switching tokenizers. **Correction**: Refactored `plot_results.py` to inspect `audit_results.json` keys dynamically and assign color palettes dynamically.
- **Sanity Checking Math**: AI initially rounded L4 usable GPU memory naively ($0.92 \times 24 = 22.08 \text{ GB}$ vs $22.08 \text{ GiB}$). **Correction**: Manually verified binary ($1024^3$) vs decimal ($10^9$) conversions to ensure exact equivalence in memory pool subtractions.

---

## 3. Human Verification & Ownership Statement

All empirical claims, bug identifications, benchmark numbers, memory calculations, goodput derivations, and strategic memo recommendations in this repository were independently verified against raw dataset outputs, empirical log files (`bench_log.csv`), and model specs (`model_spec.md`).
