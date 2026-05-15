# Speculative Decoding for Turkish NLP

> A publication-ready research codebase investigating speculative decoding efficiency across Turkish and English language tasks, with morphological analysis of token acceptance rates.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Datasets](#datasets)
- [Models](#models)
- [Experiment Design](#experiment-design)
- [Statistical Analysis](#statistical-analysis)
- [Linguistic Analysis](#linguistic-analysis)
- [Figures](#figures)
- [Results Layout](#results-layout)
- [Configuration](#configuration)
- [Citation](#citation)

---

## Overview

**Speculative decoding** is a lossless inference acceleration technique in which a small, fast *draft model* proposes multiple tokens in parallel and a large *target model* verifies them in a single forward pass. Tokens that pass the accept/reject criterion are kept; rejected tokens are resampled from a corrected residual distribution, preserving the exact output distribution of the target model.

This repository presents the **first systematic empirical study of speculative decoding on Turkish** — an agglutinative language whose rich suffix chains create morphologically complex surface forms. We hypothesise that these complex forms are harder for the draft model to predict, leading to systematically lower acceptance rates compared to English, and we quantify that gap with rigorous statistics and two complementary linguistic analyses: BPE subword fragmentation and Stanza morphological feature counting.

The implementation uses a **target-side KV cache**: the target model is initialised once on the full prompt (O(L) cost) and thereafter called only on the γ draft tokens per iteration (O(γ)), eliminating the O(L) re-encoding cost of a naive implementation and making latency much less sensitive to generation length.

---

## Key Contributions

- **First cross-lingual study of speculative decoding efficiency on Turkish**, an agglutinative language, comparing directly against English as a control group.
- Full implementation of the **accept/reject speculative decoding algorithm** (Leviathan et al., 2023) with target-side KV cache and per-token logging.
- Controlled comparison across **6 model pairs**: Turkish GPT-2 small/medium, English GPT-2 small/medium, Llama-3.2 Instruct (1B→8B), and Qwen2.5 (0.5B→7B).
- **3-seed robustness runs** for TR-small, EN-small, Llama, and Qwen primary conditions to quantify variance.
- **Cross-lingual pair**: Turkish-SFT Qwen2.5-0.5B draft vs multilingual Qwen2.5-7B-Instruct target — tests whether a Turkish-adapted draft generalises to a multilingual target.
- **Position-bucket analysis**: acceptance rates across early / mid / late token positions.
- **Dual linguistic analysis pipeline**: (1) BPE subword fragmentation — Spearman correlation between fragment count and per-word acceptance rate; (2) Stanza morphological feature counting — mean active features per word as a language-agnostic morphological complexity metric, cross-compared between Turkish and English.
- **Rejected-token frequency analysis**: identifies systematic draft failure modes by surface form.
- **Ablation study** over draft steps γ ∈ {1, 3, 5, 7, 10}.
- **Output quality validation**: ROUGE-1/2/L + BLEU comparing greedy vs speculative outputs to verify losslessness in practice.
- Complete statistical battery: Wilcoxon signed-rank, Mann-Whitney U, Cohen's d, bootstrap confidence intervals.
- Eight publication-quality figures (PDF + PNG, 300 dpi, serif font).

---

## Architecture

```
ALL business logic lives in .py files inside src/.
The notebook (run_experiments.ipynb) contains ZERO logic —
it only clones the repo, installs deps, imports from src/, and calls functions.
```

This strict separation means:

- Every function is unit-testable in isolation.
- Results are 100% reproducible by re-running the notebook.
- The notebook acts as a reproducible experiment script, not a development environment.

---

## Repository Structure

```
Speculative_decoding/
│
├── src/
│   ├── __init__.py          # empty
│   ├── config.py            # all constants and hyperparameters
│   ├── utils.py             # seed_everything, save_json, check_gpu, git_push
│   ├── data.py              # dataset loaders: XQuAD-TR, TR-News, SQuAD-EN (GPT-2 + instruct formats)
│   ├── models.py            # draft (float16) and target (float16 / NF4) model loaders
│   ├── speculative.py       # accept/reject algorithm + target KV cache, greedy, beam, run_experiment
│   ├── metrics.py           # ROUGE, BLEU, bootstrap CI, Wilcoxon, Cohen d, speedup
│   ├── linguistic.py        # position/fragmentation/rejection analysis + Stanza morphology
│   └── figures.py           # 8 publication-quality figure generators
│
├── run_experiments.ipynb    # zero-logic Colab notebook (28 cells)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Option A — Google Colab (recommended)

1. Open `run_experiments.ipynb` in Google Colab with a **T4 or A100 GPU runtime**.
2. In **Cell 1**, fill in your tokens:
   ```python
   HF_TOKEN = "hf_..."   # Hugging Face access token (read scope)
   GH_TOKEN = "ghp_..."  # GitHub Personal Access Token (repo scope)
   ```
3. Run all cells top-to-bottom.

### Option B — Local

```bash
git clone https://github.com/CengizhanBayram/Speculative_decoding.git
cd Speculative_decoding
pip install -r requirements.txt
```

**Minimum hardware:**

| Experiment | VRAM needed |
|------------|-------------|
| GPT-2 scale (all four pairs) | ≥ 4 GB |
| Llama-3.2-1B → Turkish-Llama-8B (4-bit) | ≥ 6 GB |
| Qwen2.5-0.5B → Qwen2.5-7B-Instruct (4-bit) | ≥ 5 GB |

Tested on: T4 16 GB (Colab free tier), A100 40 GB (Colab Pro+).

---

## Running Experiments

The entire pipeline is orchestrated by `run_experiments.ipynb`. Cells are ordered for sequential execution; model-memory management cells (6e, 6f) free GPU memory before loading the next model family.

| Cell | Action | Output |
|------|--------|--------|
| 1 | Mount Drive, enter tokens, HF login | — |
| 2 | Clone / pull repo, `pip install` | — |
| 3 | `sys.path` + all imports from `src/` | — |
| 4 | `seed_everything(42)`, `check_gpu()` | GPU info dict |
| 5 | Load datasets: XQuAD-TR, TR-News, SQuAD-EN, instruct formats | 4 × list[dict] |
| 6 | Load TR-small draft + target (GPT-2 117M + 774M) | 2 models |
| 6b | Load TR-medium draft (GPT-2-medium 354M; reuses target) | 1 model |
| 6c | Load EN-small draft + target (gpt2 + gpt2-large) | 2 models |
| 6d | Load EN-medium draft (gpt2-medium; reuses target) | 1 model |
| 7 | Greedy baseline — Turkish | `baseline_tr_results.csv` |
| 7b | Greedy baseline — English | `baseline_en_results.csv` |
| 8 | Speculative — TR-small (3 seeds) | `speculative_tr_small_seed{s}.csv` × 3 |
| 8b | Speculative — TR-medium (γ=5) | `speculative_tr_medium_results.csv` |
| 9 | Speculative — EN-small (3 seeds) | `speculative_en_small_seed{s}.csv` × 3 |
| 9b | Speculative — EN-medium (γ=5) | `speculative_en_medium_results.csv` |
| 10 | γ ablation over {1,3,5,7,10} on 100 TR samples | `ablation_gamma.csv` |
| 6e | Free GPT-2 memory; load Llama-3.2-1B + Turkish-Llama-8B (4-bit) | 2 models |
| 7c | Greedy baseline — Turkish-Llama-8B | `baseline_llama_results.csv` |
| 9c | Speculative — Llama 1B→8B (3 seeds) | `speculative_llama_seed{s}.csv` × 3 |
| 6f | Free Llama memory; load Qwen2.5-0.5B + Qwen2.5-7B-Instruct (4-bit) | 2 models |
| 7d | Greedy baseline — Qwen2.5-7B | `baseline_qwen_results.csv` |
| 9d | Speculative — Qwen 0.5B→7B (3 seeds) | `speculative_qwen_seed{s}.csv` × 3 |
| 11 | Full statistical test battery (all families) | `statistical_tests.json` |
| 11b | Output quality: ROUGE + BLEU | `quality_metrics.csv` |
| 12 | Position / fragmentation / rejected-token analysis | 6 × `.csv` |
| 12b | Stanza morphological analysis — TR & EN | `stanza_morphology_tr.csv`, `stanza_morphology_en.csv`, `case_distribution_tr.csv` |
| 13 | Generate all 8 figures | 16 files (8 × PDF + PNG) |
| 14 | `git_push(f"results: {timestamp}")` | commit hash |

To run a subset, execute the relevant cells — all intermediate data lives in Python variables.

---

## Datasets

### XQuAD-TR — Turkish Question Answering

- **Source:** Turkish subset of XQuAD (Artetxe et al., 2020). Loaded via `google/xquad` (`xquad.tr`); falls back to `boun-tabilab/XQuAD-TR` and `gorkemgoknar/tr-nlp-qa-xquad-trquad`.
- **Split:** validation (500 samples, configurable via `NUM_SAMPLES_QA`).
- **GPT-2 prompt format:**
  ```
  Soru: {question}
  Bağlam: {context[:300]}
  Cevap:
  ```
- **Instruct prompt format** (for Llama target):
  ```
  <|system|>
  Sen bir Türkçe soru-cevap asistanısın.
  <|user|>
  Bağlam: {context[:300]}

  Soru: {question}
  <|assistant|>
  Cevap:
  ```

### TR-News — Turkish Summarisation

- **Source:** `batubayk/TR-News` on Hugging Face Hub.
- **Split:** test (500 samples, configurable via `NUM_SAMPLES_SUM`).
- **Prompt format:**
  ```
  Aşağıdaki haberi özetle:
  {article[:400]}
  Özet:
  ```

### SQuAD — English Question Answering (control group)

- **Source:** `rajpurkar/squad`. Loaded via `rajpurkar/squad` with fallback to `squad`.
- **Split:** validation (500 samples, configurable via `NUM_SAMPLES_EN`).
- **GPT-2 prompt format:**
  ```
  Question: {question}
  Context: {context[:300]}
  Answer:
  ```
- **Instruct prompt format** (for Qwen target):
  ```
  <|system|>
  You are a helpful QA assistant.
  <|user|>
  Context: {context[:300]}

  Question: {question}
  <|assistant|>
  Answer:
  ```

All datasets are shuffled with `SEED=42` before sampling.

---

## Models

All draft–target pairs within a family **share the same tokenizer and vocabulary** — a hard requirement of speculative decoding.

### GPT-2 Turkish Pairs

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft (small) | `ytu-ce-cosmos/turkish-gpt2` | ~117 M | float16 |
| Draft (medium) | `ytu-ce-cosmos/turkish-gpt2-medium` | ~354 M | float16 |
| Target | `ytu-ce-cosmos/turkish-gpt2-large` | ~774 M | float16 |

All three models share the GPT-2 BPE tokenizer (50,257 tokens).

### GPT-2 English Pairs

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft (small) | `gpt2` | ~117 M | float16 |
| Draft (medium) | `gpt2-medium` | ~354 M | float16 |
| Target | `gpt2-large` | ~774 M | float16 |

Target size (774 M) matches the Turkish target exactly, enabling a fair cross-linguistic comparison where acceptance-rate differences are attributable to language, not model capacity.

### Llama-3 Pair

| Role | Model | Params | Fine-tuning | Dtype |
|------|-------|--------|-------------|-------|
| Draft | `unsloth/Llama-3.2-1B-Instruct` (or `meta-llama/Llama-3.2-1B-Instruct`) | ~1 B | **instruct** | float16 |
| Target | `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` | ~8 B | **instruct** | 4-bit NF4 |

Both models share the Llama-3 tokenizer (128,256 tokens) — the target is fine-tuned from Llama-3.1-8B and inherits it. Both are instruction-tuned, ensuring the draft and target are aligned in fine-tuning type for a fair comparison.

> **Note on dtype:** 4-bit NF4 is a weight storage format only. BitsAndBytes dequantizes to float16 during the forward pass (`bnb_4bit_compute_dtype=torch.float16`), so both models produce float16 logits and are numerically compatible in the accept/reject step.

### Qwen2.5 Pair

| Role | Model | Params | Fine-tuning | Dtype |
|------|-------|--------|-------------|-------|
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | ~0.5 B | **instruct** | float16 |
| Target | `Qwen/Qwen2.5-7B-Instruct` | ~7 B | **instruct** | 4-bit NF4 |

Both models share the Qwen2.5 tokenizer (151,936 tokens) and are instruction-tuned, ensuring type-matched speculative decoding. This is a clean same-type pair for cross-scale comparison.

> **Note on dtype:** Same as Llama above — NF4 is storage-only; logits are float16 and compatible.

> Model names are configured in `src/config.py`. Swap them to run your own pairs — just ensure both models share the same tokenizer.

---

## Experiment Design

### Speculative Decoding Algorithm

```
One-time prompt initialisation:
  target_model(prompt) → builds target KV cache, yields logit for position 0

For each generation step:
  Draft phase (KV-cached):
    1. Draft model generates γ tokens: d₀, d₁, …, d_{γ-1}

  Target verification (O(γ), not O(L)):
    2. target_model([d₀ … d_{γ-1}], past_kv=target_cache) in ONE forward pass

  Accept / reject:
    3. For each dᵢ:
         acceptance_prob = min(1, p_target(dᵢ) / p_draft(dᵢ))
         if U ~ Uniform(0,1) ≤ acceptance_prob → accept
         else → sample corrected token from max(0, p_target − p_draft), stop

  Update:
    4. If all γ accepted → sample bonus token from target at position γ
    5. Target KV cache is updated by ≤ γ+1 tokens (never re-encodes full context)
```

This preserves the **exact output distribution** of the target model (lossless acceleration).

### Decoding Modes

| Mode | Description | Use |
|------|-------------|-----|
| `greedy` | `do_sample=False`, autoregressive via `model.generate` | Speed baseline |
| `speculative` | Accept/reject with target-side KV cache | Main experiment |
| `beam` | Beam search, `num_beams=4` | Alternative baseline |

### Ablation Study

γ ∈ {1, 3, 5, 7, 10} on 100 TR samples (seed = SEEDS[0]). Measures trade-off between:
- **Acceptance rate** (tends to decrease as γ grows — longer drafts are harder to fully accept).
- **Latency** (non-monotone — very small γ under-exploits parallelism; very large γ wastes draft compute).

---

## Statistical Analysis

All tests are run in `src/metrics.py` and saved to `results/statistical_tests.json`.

| Test | Applied to | Purpose |
|------|-----------|---------|
| Wilcoxon signed-rank | Paired latencies | Non-parametric significance of speedup |
| Mann-Whitney U | Unpaired latencies | Robustness check |
| Cohen's d | Latency distributions | Effect size |
| Bootstrap CI (n=10 000) | Acceptance rates, speedup ratios | 95 % confidence intervals |

Seed stability is quantified by reporting mean ± std of per-seed acceptance rates for all 3-seed conditions (TR-small, EN-small, Llama, Qwen).

Significance threshold: α = 0.05 (two-sided).

---

## Linguistic Analysis

Implemented in `src/linguistic.py`. Two complementary analysis pipelines quantify morphological complexity and its effect on acceptance rates.

### Position-Bucket Analysis

Each sample's generated sequence is split into **early / mid / late** thirds. Per-bucket acceptance rates reveal whether acceptance degrades as generation progresses (longer context = harder draft alignment).

### Subword Fragmentation (BPE-based)

Since draft and target share the same tokenizer, BPE fragmentation reflects the inherent morphological complexity of each language relative to its vocabulary.

**Cross-linguistic comparison** (`oov_analysis_tr.csv` vs `oov_analysis_en.csv`): mean subword fragments per word for Turkish vs English prompts, showing Turkish's higher morphological load even when a Turkish-specific BPE vocabulary is used.

**Fragmentation–acceptance correlation** (`fragmentation_acceptance.csv`): Spearman ρ between per-word fragment count and mean acceptance rate, testing whether morphologically complex words (more subwords) are systematically harder for the draft model.

### Stanza Morphological Analysis

`stanza_morphology_analysis` uses the Stanza NLP library to extract language-agnostic morphological features (Universal Dependencies tags) for each word in the prompts.

**Morphological feature count per word** (`stanza_morphology_tr.csv` / `stanza_morphology_en.csv`): For each word the pipeline records the full `feats` string (e.g., `Case=Dat|Number=Sing|Person=3|Tense=Past`) and counts active features as `n_feats`. Turkish content words typically exhibit 3–8 features; English content words 0–2. This metric is language-model-independent — it does not depend on any tokenizer or BPE vocabulary.

**Case distribution** (`case_distribution_tr.csv`): counts the grammatical case inventory (Nominative, Accusative, Dative, Genitive, Locative, Ablative, Instrumental) to illustrate the breadth of Turkish morphological marking relative to English.

Together, the BPE fragmentation and Stanza analyses triangulate morphological complexity from two independent sources, strengthening the claim that agglutination — not tokenizer design — drives lower acceptance rates.

> **Prerequisites:** `pip install stanza` and `python -c "import stanza; stanza.download('tr')"` (run once; model is cached). GPU is used automatically if available.

### Rejected-Token Frequency Analysis

`rejected_token_analysis` counts total proposals and rejections per token surface form, identifying systematic failure modes (e.g., specific suffixes, function words, or punctuation that the draft model consistently mispredicts).

---

## Figures

All figures are saved as PDF (vector, for papers) and PNG (raster, for presentations) to `figures/`. Style: serif font, 11 pt, 300 dpi — ready for ACL/EMNLP submission.

| File stem | Content |
|-----------|---------|
| `acceptance_distribution` | Overlaid density histograms of per-sample acceptance rates for all speculative conditions |
| `latency_violin` | Violin + embedded box plots of per-sample speedup ratios, one panel per condition |
| `speedup_boxplot` | Notched box plots of speedup ratios vs greedy baseline with 1× reference line |
| `ablation_gamma` | Dual-panel: acceptance rate + latency vs γ with IQR shading |
| `position_acceptance` | Bar chart of acceptance rates in early / mid / late token position buckets |
| `model_comparison` | Side-by-side grouped bars: mean acceptance rate + median speedup across all model pairs |
| `fragmentation_comparison` | Two-panel: fragment distribution histogram + per-task mean fragments for Turkish vs English |
| `quality_metrics` | Grouped bar chart: ROUGE-1/2/L (+ BLEU) for greedy vs speculative across conditions |

---

## Results Layout

After a full run, `results/` contains:

```
results/
│
├── baseline_tr_results.csv               # greedy baseline — Turkish (GPT-2)
├── baseline_en_results.csv               # greedy baseline — English (GPT-2)
├── baseline_llama_results.csv            # greedy baseline — Turkish-Llama-8B
├── baseline_qwen_results.csv             # greedy baseline — Qwen2.5-7B
│
├── speculative_tr_small_seed42.csv       # TR-small, seed 42
├── speculative_tr_small_seed123.csv      # TR-small, seed 123
├── speculative_tr_small_seed456.csv      # TR-small, seed 456
├── speculative_tr_small_results.csv      # TR-small, all seeds combined
├── speculative_tr_medium_results.csv     # TR-medium, γ=5
│
├── speculative_en_small_seed42.csv       # EN-small, seed 42
├── speculative_en_small_seed123.csv      # EN-small, seed 123
├── speculative_en_small_seed456.csv      # EN-small, seed 456
├── speculative_en_small_results.csv      # EN-small, all seeds combined
├── speculative_en_medium_results.csv     # EN-medium, γ=5
│
├── speculative_llama_seed42.csv          # Llama 1B→8B, seed 42
├── speculative_llama_seed123.csv
├── speculative_llama_seed456.csv
├── speculative_llama_results.csv         # Llama, all seeds combined
│
├── speculative_qwen_seed42.csv           # Qwen 0.5B→7B, seed 42
├── speculative_qwen_seed123.csv
├── speculative_qwen_seed456.csv
├── speculative_qwen_results.csv          # Qwen, all seeds combined
│
├── ablation_gamma.csv                    # γ ablation (TR-small, 100 samples)
├── statistical_tests.json               # all statistical test outputs
├── quality_metrics.csv                  # ROUGE + BLEU: greedy vs speculative
│
├── position_acceptance.csv              # acceptance rate per position bucket
├── oov_analysis_tr.csv                  # Turkish subword fragmentation per word
├── oov_analysis_en.csv                  # English subword fragmentation per word
├── fragmentation_acceptance.csv         # Spearman ρ: fragment count vs acceptance
├── stanza_morphology_tr.csv             # Turkish word-level morphological features (Stanza)
├── stanza_morphology_en.csv             # English word-level morphological features (Stanza)
├── case_distribution_tr.csv             # Turkish grammatical case distribution
├── rejected_tokens_tr.csv               # top-30 most-proposed tokens (Turkish)
└── rejected_tokens_en.csv               # top-30 most-proposed tokens (English)
```

CSV columns for speculative results:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Input prompt |
| `reference` | str | Ground-truth answer/summary |
| `task` | str | `qa_tr`, `summarization_tr`, `qa_en`, `qa_tr_instruct`, `qa_en_instruct` |
| `mode` | str | `speculative` / `greedy` / `beam` |
| `draft_steps` | int | γ value used |
| `generated_text` | str | Model output |
| `acceptance_rate` | float | Fraction of draft tokens accepted |
| `num_target_calls` | int | Number of target forward passes |
| `latency_ms` | float | Wall-clock generation time (ms) |
| `token_level_log` | list[dict] | Per-token: position, token_str, p_draft, p_target, acceptance_prob, accepted |
| `seed` | int | Random seed (present in multi-seed runs only) |

---

## Configuration

All hyperparameters live in `src/config.py`:

```python
SEED  = 42
SEEDS = [42, 123, 456]   # seeds for multi-seed robustness runs

# Turkish GPT-2 pairs (shared 50,257-token BPE vocabulary)
DRAFT_MODEL_TR_SMALL_NAME  = "ytu-ce-cosmos/turkish-gpt2"          # 117 M
DRAFT_MODEL_TR_MEDIUM_NAME = "ytu-ce-cosmos/turkish-gpt2-medium"   # 354 M
TARGET_MODEL_TR_NAME       = "ytu-ce-cosmos/turkish-gpt2-large"    # 774 M

# English GPT-2 pairs (same BPE vocabulary as above but English-trained)
DRAFT_MODEL_EN_SMALL_NAME  = "gpt2"           # 117 M
DRAFT_MODEL_EN_MEDIUM_NAME = "gpt2-medium"    # 354 M
TARGET_MODEL_EN_NAME       = "gpt2-large"     # 774 M

# Llama-3 pair (128,256-token tiktoken vocabulary)
DRAFT_MODEL_LLAMA_NAME  = "unsloth/Llama-3.2-1B-Instruct"
TARGET_MODEL_LLAMA_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
QUANTIZATION_BITS_LLAMA = 4    # 4-bit NF4 for 8B target

# Qwen2.5 pair (151,936-token vocabulary)
DRAFT_MODEL_QWEN_NAME  = "ytu-ce-cosmos/tr-Qwen2.5-0.5B-SFT-v1"
TARGET_MODEL_QWEN_NAME = "Qwen/Qwen2.5-7B-Instruct"
QUANTIZATION_BITS_QWEN = 4    # 4-bit NF4 for 7B target

MAX_NEW_TOKENS      = 128
DRAFT_STEPS_LIST    = [1, 3, 5, 7, 10]
DEFAULT_DRAFT_STEPS = 5

NUM_SAMPLES_QA    = 500   # XQuAD-TR
NUM_SAMPLES_SUM   = 500   # TR-News
NUM_SAMPLES_EN    = 500   # SQuAD
NUM_SAMPLES_LLAMA = 300   # reduced — 8B is slower per sample
NUM_SAMPLES_QWEN  = 300   # reduced — 7B is slower per sample

QUANTIZATION_BITS = 0    # 0 = float16 for GPT-2 scale models
```

No changes to any other file are needed to swap models or adjust sample sizes.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{bayram2025speculative,
  title   = {Speculative Decoding for Turkish NLP: Morphological Analysis of Token Acceptance Rates},
  author  = {Bayram, Cengizhan},
  year    = {2025},
  url     = {https://github.com/CengizhanBayram/Speculative_decoding}
}
```

### References

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). **Fast Inference from Transformers via Speculative Decoding.** ICML 2023.
- Chen, C., et al. (2023). **Accelerating Large Language Model Decoding with Speculative Sampling.** arXiv:2302.01318.
- Dettmers, T., et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs.** NeurIPS 2023.
- Artetxe, M., et al. (2020). **On the Cross-lingual Transferability of Monolingual Representations.** ACL 2020.
