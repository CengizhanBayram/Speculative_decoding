# Speculative Decoding for Turkish NLP

> A publication-ready research codebase investigating speculative decoding efficiency on Turkish and English language tasks, with morphological analysis of token acceptance rates.

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

This repository studies speculative decoding specifically in the context of **agglutinative Turkish morphology**, hypothesising that Turkish's complex suffix chains cause systematically lower draft-token acceptance rates compared to English — and quantifying that gap with rigorous statistics and linguistic analysis.

---

## Key Contributions

- Full implementation of the **accept/reject speculative decoding algorithm** (Leviathan et al., 2023) with per-token logging.
- Controlled comparison between **Turkish QA**, **Turkish summarisation**, and **English QA** tasks.
- **Morpheme-category rejection analysis** using `zeyrek` — the first study to break down acceptance rates by Turkish morphological category (ROOT\_ONLY, NOMINAL\_SUFFIX, VERBAL\_SUFFIX, DERIVATIONAL, COMPOUND).
- **Position-bucket analysis**: acceptance rates across early / mid / late token positions.
- **OOV fragmentation analysis**: Spearman correlation between subword fragment counts in draft vs. target vocabularies.
- **Ablation study** over draft steps γ ∈ {1, 3, 5, 7, 10}.
- Complete statistical battery: Wilcoxon signed-rank, Mann-Whitney U, Cohen's d, bootstrap confidence intervals.
- Five publication-quality figures (PDF + PNG, 300 dpi, serif font).

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
│   ├── data.py              # dataset loaders: TQuAD, TR-News, SQuAD-EN
│   ├── models.py            # draft (float16) and target (NF4) model loaders
│   ├── speculative.py       # accept/reject algorithm, greedy, beam, run_experiment
│   ├── metrics.py           # ROUGE, BLEU, bootstrap CI, Wilcoxon, Cohen d, speedup
│   ├── linguistic.py        # morpheme categorisation, rejection/position/OOV analysis
│   └── figures.py           # 5 publication-quality figure generators
│
├── run_experiments.ipynb    # zero-logic Colab notebook (17 cells)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Option A — Google Colab (recommended)

1. Open `run_experiments.ipynb` in Google Colab (T4/A100 GPU runtime).
2. In **Cell 1**, fill in your tokens directly:
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

> **Minimum hardware:** NVIDIA GPU with ≥ 8 GB VRAM (gpt2-xl ~3 GB, turkish-gpt2-large ~1.5 GB in float16).  
> Tested on: T4 16 GB (Colab free tier), A100 40 GB (Colab Pro+).

---

## Running Experiments

The entire experiment pipeline is orchestrated by `run_experiments.ipynb`:

| Cell | Action | Output |
|------|--------|--------|
| 1 | Mount Drive, enter tokens, HF login | — |
| 2 | Clone / pull repo, `pip install` | — |
| 3 | `sys.path` + imports from `src/` | — |
| 4 | `seed_everything(42)`, `check_gpu()` | GPU info dict |
| 5 | Load TQuAD + TR-News + SQuAD-EN | 3 × list[dict] |
| 6 | Load Turkish draft + target models | 2 models |
| 6b | Load English draft + target models | 2 models |
| 7 | Greedy baseline — Turkish | `baseline_tr_results.csv` |
| 7b | Greedy baseline — English | `baseline_en_results.csv` |
| 8 | Speculative decoding — Turkish | `speculative_tr_results.csv` |
| 9 | Speculative decoding — English | `speculative_en_results.csv` |
| 10 | γ ablation over {1,3,5,7,10} | `ablation_gamma.csv` |
| 11 | Full statistical test battery | `statistical_tests.json` |
| 12 | Morpheme / position / OOV analysis | 3 × `.csv` |
| 13 | Generate all figures | 10 files (5 × PDF + PNG) |
| 14 | `git_push(f"results: {timestamp}")` | commit hash |

To run a subset, simply execute the relevant cells — all intermediate data lives in Python variables.

---

## Datasets

### TQuAD — Turkish Question Answering

- **Source:** Turkish translation/adaptation of SQuAD.
- **Split used:** validation.
- **Prompt format:**
  ```
  Soru: {question}
  Bağlam: {context[:300]}
  Cevap:
  ```
- **Metric:** ROUGE-1/2/L, corpus BLEU.
- **Samples:** 500 (configurable via `NUM_SAMPLES_QA`).

### TR-News — Turkish Summarisation

- **Source:** `batubayk/TR-News` on Hugging Face Hub.
- **Split used:** test.
- **Prompt format:**
  ```
  Aşağıdaki haberi özetle:
  {article[:400]}
  Özet:
  ```
- **Metric:** ROUGE-1/2/L.
- **Samples:** 500 (configurable via `NUM_SAMPLES_SUM`).

### SQuAD — English Question Answering (control group)

- **Source:** `rajpurkar/squad` on Hugging Face Hub.
- **Split used:** validation.
- **Prompt format:**
  ```
  Question: {question}
  Context: {context[:300]}
  Answer:
  ```
- **Metric:** ROUGE-1/2/L, corpus BLEU.
- **Samples:** 500 (configurable via `NUM_SAMPLES_EN`).

All datasets are shuffled with `SEED=42` before sampling.

---

## Models

Both draft–target pairs **share the same tokenizer and vocabulary**, which is a hard requirement of the speculative decoding algorithm.

### Turkish Pair

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `ytu-ce-cosmos/turkish-gpt2` | ~117 M | float16 |
| Target | `ytu-ce-cosmos/turkish-gpt2-large` | ~774 M | float16 |

Both models use the GPT-2 BPE tokenizer (50,257 tokens), so token IDs are compatible across draft and target.

### English Pair

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `gpt2` | ~117 M | float16 |
| Target | `gpt2-xl` | ~1.5 B | float16 |

Both models use the standard GPT-2 BPE tokenizer, ensuring a shared vocabulary.

> Model names are configured in `src/config.py`. Swap them to reproduce experiments with different model pairs — just ensure both models in a pair share the same tokenizer.

---

## Experiment Design

### Speculative Decoding Algorithm

```
For each generation step:
  1. Draft model autoregressively generates γ tokens: d₁, d₂, …, dγ
  2. Target model scores all γ+1 positions in ONE forward pass
  3. For each dᵢ:
       acceptance_prob = min(1, p_target(dᵢ) / p_draft(dᵢ))
       if U ~ Uniform(0,1) ≤ acceptance_prob → accept
       else → sample from max(0, p_target − p_draft), stop
  4. If all γ accepted → sample bonus token from target at position γ+1
```

This preserves the **exact output distribution** of the target model (lossless acceleration).

### Decoding Modes

| Mode | Description | Use |
|------|-------------|-----|
| `greedy` | `do_sample=False`, autoregressive | Speed baseline |
| `speculative` | Accept/reject with draft model | Main experiment |
| `beam` | Beam search, `num_beams=4` | Quality upper bound |

### Ablation Study

γ ∈ {1, 3, 5, 7, 10} on 100 TR samples each. Measures trade-off between:
- Acceptance rate (decreases as γ grows — longer drafts are harder to fully accept).
- Latency (non-monotone — small γ wastes target capacity; large γ wastes draft capacity).

---

## Statistical Analysis

All tests are run in `src/metrics.py` and saved to `results/statistical_tests.json`.

| Test | Applied to | Purpose |
|------|-----------|---------|
| Wilcoxon signed-rank | Paired latencies | Non-parametric significance of speedup |
| Mann-Whitney U | Unpaired latencies | Robustness check |
| Cohen's d | Latency distributions | Effect size |
| Bootstrap CI (n=10 000) | Acceptance rates, speedup ratios | 95 % confidence intervals |

Significance threshold: α = 0.05 (two-sided).

---

## Linguistic Analysis

Implemented in `src/linguistic.py` using the `zeyrek` Turkish morphological analyser.

### Morpheme Categories

| Category | Description | Example |
|----------|-------------|---------|
| `ROOT_ONLY` | No suffixes attached | *ev* (house) |
| `NOMINAL_SUFFIX` | Case, number, possession | *evlerde* (in the houses) |
| `VERBAL_SUFFIX` | Tense, mood, person | *gidiyorum* (I am going) |
| `DERIVATIONAL` | Word-class changing suffix | *güzellik* (beauty ← beautiful) |
| `COMPOUND` | Multiple morpheme boundaries | *başbakan* (prime minister) |
| `UNKNOWN` | Unanalysable / OOV | proper nouns, loanwords |

### Position-Bucket Analysis

Each sample's generated sequence is split into **early / mid / late** thirds. Acceptance rates per bucket reveal whether morphological complexity accumulates as generation progresses.

### OOV Fragmentation Analysis

For every prompt word, subword fragment counts are measured in both draft and target vocabularies. **Spearman correlation** between draft and target fragmentation indicates vocabulary alignment — low correlation predicts higher rejection rates.

---

## Figures

All figures are saved as PDF (vector, for papers) and PNG (raster, for presentations) to `figures/`.

| File stem | Content |
|-----------|---------|
| `acceptance_distribution` | Overlaid density histograms of per-sample acceptance rates (TR vs. EN) |
| `speedup_boxplot` | Notched box-plots of speedup ratios vs. greedy baseline, with 1× reference line |
| `ablation_gamma` | Dual-panel: acceptance rate + latency vs. γ with IQR shading |
| `morpheme_rejection` | Horizontal bars coloured by rejection rate per morpheme category |
| `position_acceptance` | Bar chart of acceptance rates in early / mid / late token position buckets |

Style: serif font, 11 pt, 300 dpi — ready for ACL/EMNLP submission.

---

## Results Layout

After a full run, `results/` contains:

```
results/
├── baseline_tr_results.csv       # greedy baseline — Turkish (latency, generated text, task)
├── baseline_en_results.csv       # greedy baseline — English
├── speculative_tr_results.csv    # TR speculative (+ acceptance_rate, token_level_log)
├── speculative_en_results.csv    # EN speculative
├── ablation_gamma.csv            # γ ablation across all draft steps
├── statistical_tests.json        # all statistical test outputs
├── morpheme_rejection.csv        # rejection rate per morpheme category
├── position_acceptance.csv       # acceptance rate per position bucket
└── oov_analysis.csv              # per-word fragmentation + Spearman r
```

CSV columns for speculative results:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Input prompt |
| `reference` | str | Ground-truth answer/summary |
| `task` | str | `qa_tr`, `summarization_tr`, `qa_en` |
| `mode` | str | `speculative` / `greedy` / `beam` |
| `draft_steps` | int | γ value used |
| `generated_text` | str | Model output |
| `acceptance_rate` | float | Fraction of draft tokens accepted |
| `num_target_calls` | int | Number of target forward passes |
| `latency_ms` | float | Wall-clock generation time (ms) |
| `token_level_log` | list[dict] | Per-token: position, token_str, p_draft, p_target, accepted |

---

## Configuration

All hyperparameters live in `src/config.py`:

```python
SEED                 = 42
DRAFT_MODEL_NAME     = "ytu-ce-cosmos/turkish-gpt2"        # ~117 M — Turkish draft
TARGET_MODEL_NAME    = "ytu-ce-cosmos/turkish-gpt2-large"  # ~774 M — Turkish target
DRAFT_MODEL_EN_NAME  = "gpt2"                              # ~117 M — English draft
TARGET_MODEL_EN_NAME = "gpt2-xl"                           # ~1.5 B — English target
MAX_NEW_TOKENS       = 128
DRAFT_STEPS_LIST     = [1, 3, 5, 7, 10]
DEFAULT_DRAFT_STEPS  = 5
NUM_SAMPLES_QA       = 500
NUM_SAMPLES_SUM      = 500
NUM_SAMPLES_EN       = 500
QUANTIZATION_BITS    = 0   # 0 = float16 (no quantization); set to 4 for 7B+ models
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
- Şahin, G. G., & Steedman, M. (2018). **Data Augmentation via Dependency Tree Morphological Inflection.** EMNLP 2018.

---


