# Does Agglutinative Morphology Impede Speculative Decoding?

> A controlled study on Turkish and English using matched GPT-2 family model pairs, morphological fragmentation analysis, and cross-scale experimentation.

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
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

This repository presents the **first systematic empirical study of speculative decoding on Turkish** — an agglutinative language whose rich suffix chains fragment into 1.40× more BPE subword tokens per word than English (Cohen's *d* = 0.626, *p* ≈ 0). Contrary to the expectation that agglutination would hurt the draft model, Turkish acceptance rates *exceed* English (α = 0.768 vs. 0.716; Δ = 5.2 pp, *p* ≈ 0). The explanation is same-corpus co-training: the `ytu-ce-cosmos` draft and target were pre-trained on the identical 35 GB Turkish corpus, producing stronger argmax alignment than the cross-run OpenAI GPT-2 family.

At **T = 0.0** (greedy decoding), the accept/reject step is **fully deterministic**: a draft token is accepted if and only if it equals the target's argmax, so acceptance rates are perfectly reproducible (σ = 0.000 across three independent seeds). Speculative outputs are empirically lossless (ΔROUGE-1 < 0.001).

A cross-scale experiment across six draft–target pairs shows that the **draft-to-target parameter ratio** — not language morphology — is the primary efficiency lever. Draft models at ~15% of target size achieve speedup (1.250× TR, 1.159× EN); draft models at ~46% of target size flip to slowdown (0.918× TR, 0.859× EN) despite higher acceptance rates.

The implementation uses a **target-side KV cache**: the target model is initialised once on the full prompt (O(L) cost) and thereafter called only on the γ draft tokens per iteration (O(γ)), eliminating the O(L) re-encoding cost of a naive implementation.

---

## Key Findings

| Finding | Value |
|---------|-------|
| Turkish acceptance rate | α = 0.768 |
| English acceptance rate | α = 0.716 |
| TR vs EN gap | Δ = 5.2 pp (*p* ≈ 0) |
| σ across seeds (T = 0.0) | 0.000 (perfectly deterministic) |
| TR small-draft speedup | 1.250× (Wilcoxon *p* < 10⁻¹⁵¹) |
| EN small-draft speedup | 1.159× (Wilcoxon *p* < 10⁻⁴³) |
| TR medium-draft speedup | 0.918× (slowdown) |
| BPE fragmentation ratio TR/EN | 1.40× (Cohen's *d* = 0.626) |
| Llama-3 cross-lingual α | 0.448 (−32 pp vs. TR same-corpus) |
| Qwen-2.5 cross-lingual α | 0.370 (−40 pp vs. TR same-corpus) |

---

## Key Contributions

- **First cross-lingual study of speculative decoding on Turkish**, an agglutinative language, with English as a controlled baseline.
- Full implementation of the **accept/reject speculative decoding algorithm** (Leviathan et al., 2023) with target-side KV cache and per-token logging.
- Controlled comparison across **6 draft–target pairs**: Turkish GPT-2 small/medium, English GPT-2 small/medium, Llama-3.2 (1B→8B), and Qwen2.5 (0.5B→7B).
- **3-seed robustness evaluation** for Turkish small-draft, confirming σ = 0.000 acceptance rate variance at T = 0.0.
- **Ablation study** over draft steps γ ∈ {1, 3, 5, 7, 10}, identifying γ = 5 as the optimal default.
- **Position-bucket analysis**: acceptance rates by early / mid / late thirds of the generated sequence, ruling out morphological-error accumulation.
- **BPE subword fragmentation analysis**: quantifies Turkish vs. English morphological load and its (non-)effect on acceptance rates.
- **Rejected-token frequency analysis**: identifies which token surface forms the draft model systematically mispredicts.
- **Output quality validation**: ROUGE-1/2/L + BLEU confirming empirical losslessness (Δ < 0.001 for all metrics).
- Complete statistical battery: Wilcoxon signed-rank, Mann-Whitney U, Cohen's *d*, bootstrap confidence intervals.
- **Cross-lingual failure analysis**: vocabulary mismatch and domain/instruction-tuning gaps as causes of the 32–40 pp acceptance penalty in Llama and Qwen pairs.
- Five publication-quality figures (PDF + PNG, 300 dpi, serif font).
- Publicly released code, results, and paper draft for full reproducibility.

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
│   ├── data.py              # dataset loaders: XQuAD-TR, TR-News, SQuAD-EN
│   ├── models.py            # draft (float16) and target (float16 / NF4) model loaders
│   ├── speculative.py       # accept/reject algorithm + target KV cache, greedy, run_experiment
│   ├── metrics.py           # ROUGE, BLEU, bootstrap CI, Wilcoxon, Cohen d, speedup
│   ├── linguistic.py        # position/fragmentation/rejection analysis
│   └── figures.py           # 5 publication-quality figure generators
│
├── paper/
│   ├── main.tex             # paper source (ACL-style, multicol layout)
│   ├── references.bib       # 15 BibTeX entries
│   └── figures/             # PDF figure files included by main.tex
│
├── run_experiments.ipynb    # zero-logic Colab notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Option A — Google Colab (recommended)

1. Open `run_experiments.ipynb` in Google Colab with an **L4 or A100 GPU runtime**.
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

Tested on: L4 24 GB (Colab Pro), A100 40 GB (Colab Pro+).

---

## Running Experiments

The entire pipeline is orchestrated by `run_experiments.ipynb`. Cells are ordered for sequential execution; model-memory management cells (6e, 6f) free GPU memory before loading the next model family.

| Cell | Action | Output |
|------|--------|--------|
| 1 | Mount Drive, enter tokens, HF login | — |
| 2 | Clone / pull repo, `pip install` | — |
| 3 | `sys.path` + all imports from `src/` | — |
| 4 | `seed_everything(42)`, `check_gpu()` | GPU info dict |
| 5 | Load datasets: XQuAD-TR, TR-News, SQuAD-EN | 3 × list[dict] |
| 6 | Load TR-small draft + target (GPT-2 117M + 774M) | 2 models |
| 6b | Load TR-medium draft (GPT-2-medium 354M; reuses target) | 1 model |
| 6c | Load EN-small draft + target (gpt2 + gpt2-large) | 2 models |
| 6d | Load EN-medium draft (gpt2-medium; reuses target) | 1 model |
| 7 | Greedy baseline — Turkish | `baseline_tr_results.csv` |
| 7b | Greedy baseline — English | `baseline_en_results.csv` |
| 8 | Speculative — TR-small (3 seeds × 1,000 samples) | `speculative_tr_small_seed{s}.csv` × 3 |
| 8b | Speculative — TR-medium (γ=5) | `speculative_tr_medium_results.csv` |
| 9 | Speculative — EN-small (seed 42, 500 samples) | `speculative_en_small_seed42.csv` |
| 9b | Speculative — EN-medium (γ=5) | `speculative_en_medium_results.csv` |
| 10 | γ ablation over {1,3,5,7,10} on 100 TR QA samples | `ablation_gamma.csv` |
| 6e | Free GPT-2 memory; load Llama-3.2-1B + Turkish-Llama-8B (4-bit) | 2 models |
| 7c | Greedy baseline — Turkish-Llama-8B | `baseline_llama_results.csv` |
| 9c | Speculative — Llama 1B→8B | `speculative_llama_results.csv` |
| 6f | Free Llama memory; load Qwen2.5-0.5B + Qwen2.5-7B-Instruct (4-bit) | 2 models |
| 7d | Greedy baseline — Qwen2.5-7B | `baseline_qwen_results.csv` |
| 9d | Speculative — Qwen 0.5B→7B | `speculative_qwen_results.csv` |
| 11 | Full statistical test battery | `statistical_tests.json` |
| 11b | Output quality: ROUGE + BLEU | `quality_metrics.csv` |
| 12 | Position / fragmentation / rejected-token analysis | 6 × `.csv` |
| 13 | Generate all 5 figures | 10 files (5 × PDF + PNG) |
| 14 | `git_push(f"results: {timestamp}")` | commit hash |

---

## Datasets

### XQuAD-TR — Turkish Question Answering

- **Source:** Turkish subset of XQuAD (Artetxe et al., 2020). Loaded via `google/xquad` (`xquad.tr`).
- **Split:** validation (500 samples per seed).
- **Prompt format:**
  ```
  Soru: {question}
  Bağlam: {context[:300]}
  Cevap:
  ```

### TR-News — Turkish Summarisation

- **Source:** Baykara & Güngör (2022), `batubayk/TR-News` on Hugging Face Hub.
- **Split:** test (500 samples per seed).
- **Prompt format:**
  ```
  Aşağıdaki haberi özetle:
  {article[:400]}
  Özet:
  ```

### SQuAD — English Question Answering (control group)

- **Source:** `rajpurkar/squad`, validation split (500 samples, seed 42 only).
- **Prompt format:**
  ```
  Question: {question}
  Context: {context[:300]}
  Answer:
  ```

All datasets are shuffled with the configured seed before sampling.

---

## Models

All draft–target pairs within a family **share the same tokenizer and vocabulary** — a hard requirement of speculative decoding.

### GPT-2 Turkish Pairs

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft (small) | `ytu-ce-cosmos/turkish-gpt2` | ~117 M | float16 |
| Draft (medium) | `ytu-ce-cosmos/turkish-gpt2-medium` | ~354 M | float16 |
| Target | `ytu-ce-cosmos/turkish-gpt2-large` | ~774 M | float16 |

All three models share the GPT-2 BPE tokenizer (50,257 tokens) and were pre-trained on the **same 35 GB Turkish corpus** — the key enabler of high acceptance rates.

### GPT-2 English Pairs

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft (small) | `gpt2` | ~117 M | float16 |
| Draft (medium) | `gpt2-medium` | ~354 M | float16 |
| Target | `gpt2-large` | ~774 M | float16 |

Target size (774 M) matches the Turkish target exactly, enabling a fair cross-linguistic comparison.

### Llama-3 Pair (cross-lingual)

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `unsloth/Llama-3.2-1B-Instruct` | ~1 B | float16 |
| Target | `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` | ~8 B | 4-bit NF4 |

Cross-lingual pair: English-base draft paired with Turkish instruction-tuned target. Acceptance rate α = 0.448 (−32 pp vs. Turkish same-corpus pair), confirming vocabulary mismatch as the primary penalty source.

### Qwen2.5 Pair

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | ~0.5 B | float16 |
| Target | `Qwen/Qwen2.5-7B-Instruct` | ~7 B | 4-bit NF4 |

Same-family, same-vocabulary pair. Acceptance rate α = 0.370 on Turkish text (−40 pp vs. Turkish same-corpus pair), reflecting that a non-Turkish-native vocabulary cannot represent Turkish morphology compactly.

> **Note on 4-bit NF4:** BitsAndBytes dequantizes to float16 during the forward pass (`bnb_4bit_compute_dtype=torch.float16`), so both models produce float16 logits and are numerically compatible in the accept/reject step.

---

## Experiment Design

### Speculative Decoding Algorithm

```
One-time prompt initialisation:
  target_model(prompt) → builds target KV cache

For each generation step:
  Draft phase (KV-cached):
    1. Draft model generates γ tokens: d₀, d₁, …, d_{γ-1}

  Target verification (O(γ), not O(L)):
    2. target_model([d₀ … d_{γ-1}], past_kv=target_cache) in ONE forward pass

  Accept / reject (at T = 0.0, fully deterministic):
    3. For each dᵢ:
         accept iff dᵢ == argmax p_target   (no stochastic draw at T = 0.0)
         else → sample corrected token, stop

  Update:
    4. If all γ accepted → sample bonus token from target at position γ
    5. Target KV cache updated by ≤ γ+1 tokens (never re-encodes full context)
```

At **T = 0.0** the accept/reject criterion is fully deterministic: there is no uniform random draw. A draft token is accepted if and only if it equals the target model's argmax. This guarantees σ = 0.000 acceptance rate variance across seeds and produces **identical output** to autoregressive greedy decoding (verified by ΔROUGE-1 < 0.001 in all conditions).

### Decoding Modes

| Mode | Description | Use |
|------|-------------|-----|
| `greedy` | `do_sample=False`, autoregressive via `model.generate` | Speed baseline |
| `speculative` | Accept/reject with target-side KV cache, T = 0.0 | Main experiment |

### Ablation Study

γ ∈ {1, 3, 5, 7, 10} on 100 Turkish QA samples (seed 42). Key results:

| γ | α | Latency (ms) | Speedup |
|---|---|-------------|---------|
| 1 | 0.926 | 4,780 | 0.791× |
| 3 | 0.860 | 3,434 | 1.111× |
| 5 | 0.792 | 3,075 | 1.251× |
| 7 | 0.732 | 3,019 | 1.288× |
| 10 | 0.664 | 2,998 | 1.323× |

Acceptance rate decreases with γ (compounding draft error); latency decreases monotonically. **γ = 5** is recommended as the default — it sits at the knee of the speedup curve before diminishing returns.

---

## Statistical Analysis

All tests are run in `src/metrics.py` and saved to `results/statistical_tests.json`.

| Test | Applied to | Purpose |
|------|-----------|---------|
| Wilcoxon signed-rank | Paired latencies (greedy vs. speculative) | Non-parametric significance of speedup |
| Mann-Whitney U | TR vs. EN acceptance rates | Cross-linguistic significance |
| Cohen's *d* | Latency and fragmentation distributions | Effect size |
| Bootstrap CI (n=10,000) | Acceptance rates, speedup ratios | 95% confidence intervals |

**Seed stability:** At T = 0.0, acceptance is fully deterministic — σ = 0.000 across all three seeds for Turkish small-draft. Latency shows minor variation (σ ≈ 14 ms) attributable to GPU scheduling jitter, not algorithmic variance. A single-seed English run is therefore statistically equivalent to a three-seed run for the purpose of acceptance rate comparison.

Significance threshold: α = 0.05 (two-sided).

---

## Linguistic Analysis

Implemented in `src/linguistic.py`.

### Position-Bucket Analysis

Each sample's generated sequence is split into **early / mid / late** thirds. Acceptance rates increase slightly across position (87.5% → 93.3% → 93.7%), ruling out accumulation of morphological difficulty over generation.

### Subword Fragmentation (BPE-based)

Turkish prompts require 1.40× more BPE subword tokens per word than English (mean 2.156 vs. 1.545 fragments/word; Cohen's *d* = 0.626, *p* ≈ 0). A Spearman rank correlation between per-word fragment count and per-word acceptance rate yields ρ = +0.138 (*p* = 0.423) — non-significant and in the *positive* direction, meaning higher fragmentation does not predict lower acceptance. Domain-adapted Turkish models learn morpheme-boundary distributions well enough to predict complex words accurately.

**Fragmentation sources** beyond pure agglutination: foreign proper nouns with Turkish case suffixes (e.g., *Oxlade-Chamberlain'in*, 12 subwords), URLs/technical notation, and non-Latin scripts in XQuAD-TR Wikipedia contexts.

### Rejected-Token Frequency Analysis

`rejected_token_analysis` counts total proposals and rejections per token surface form. Top rejected tokens: demonstratives (*bu*: 24.1% rejection rate), hyphens (22.4%), conjunctions (*ve*: 20.4%). Punctuation has the lowest rejection rates (period: 8.5%), consistent with local predictability. Common function words carry elevated rejection rates due to variable morphosyntactic contexts the smaller draft model captures less reliably.

---

## Figures

All figures saved as PDF (vector) and PNG (raster, 300 dpi) to `figures/`. Style: serif font, 11 pt — ready for ACL/EMNLP submission.

| File stem | Content |
|-----------|---------|
| `latency_violin` | Violin + box plots of per-sample speedup ratios for all four GPT-2 pairs |
| `model_comparison` | Side-by-side grouped bars: acceptance rate + median speedup across all model pairs |
| `ablation_gamma` | Dual-panel: acceptance rate + latency vs. γ with IQR shading |
| `position_acceptance` | Bar chart of acceptance rates in early / mid / late token position buckets |
| `fragmentation_comparison` | Two-panel: fragment distribution histogram + per-task mean fragments, TR vs. EN |

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
├── speculative_tr_small_seed42.csv       # TR-small, seed 42  (1,000 samples)
├── speculative_tr_small_seed123.csv      # TR-small, seed 123 (1,000 samples)
├── speculative_tr_small_seed456.csv      # TR-small, seed 456 (1,000 samples)
├── speculative_tr_small_results.csv      # TR-small, all seeds combined (3,000 samples)
├── speculative_tr_medium_results.csv     # TR-medium, γ=5
│
├── speculative_en_small_seed42.csv       # EN-small, seed 42  (500 samples)
├── speculative_en_small_results.csv      # EN-small (same as seed42)
├── speculative_en_medium_results.csv     # EN-medium, γ=5
│
├── speculative_llama_results.csv         # Llama 1B→8B
├── speculative_qwen_results.csv          # Qwen 0.5B→7B
│
├── ablation_gamma.csv                    # γ ablation (TR-small, 100 QA samples)
├── statistical_tests.json               # all statistical test outputs
├── quality_metrics.csv                  # ROUGE + BLEU: greedy vs. speculative
│
├── position_acceptance.csv              # acceptance rate per position bucket
├── oov_analysis_tr.csv                  # Turkish subword fragmentation per word
├── oov_analysis_en.csv                  # English subword fragmentation per word
├── fragmentation_acceptance.csv         # Spearman ρ: fragment count vs. acceptance
└── rejected_tokens_tr.csv              # top-30 most-proposed tokens + rejection rates
```

CSV columns for speculative results:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Input prompt |
| `reference` | str | Ground-truth answer/summary |
| `task` | str | `qa_tr`, `summarization_tr`, `qa_en` |
| `mode` | str | `speculative` / `greedy` |
| `draft_steps` | int | γ value used |
| `generated_text` | str | Model output |
| `acceptance_rate` | float | Fraction of draft tokens accepted |
| `num_target_calls` | int | Number of target forward passes |
| `latency_ms` | float | Wall-clock generation time (ms) |
| `seed` | int | Random seed (present in multi-seed runs) |

---

## Configuration

All hyperparameters live in `src/config.py`:

```python
SEED  = 42
SEEDS = [42, 123, 456]   # seeds for Turkish 3-seed robustness runs

# Turkish GPT-2 pairs (shared 50,257-token BPE vocabulary, same training corpus)
DRAFT_MODEL_TR_SMALL_NAME  = "ytu-ce-cosmos/turkish-gpt2"          # 117 M
DRAFT_MODEL_TR_MEDIUM_NAME = "ytu-ce-cosmos/turkish-gpt2-medium"   # 354 M
TARGET_MODEL_TR_NAME       = "ytu-ce-cosmos/turkish-gpt2-large"    # 774 M

# English GPT-2 pairs (same BPE vocabulary, English-trained)
DRAFT_MODEL_EN_SMALL_NAME  = "gpt2"           # 117 M
DRAFT_MODEL_EN_MEDIUM_NAME = "gpt2-medium"    # 354 M
TARGET_MODEL_EN_NAME       = "gpt2-large"     # 774 M

# Llama-3 pair (128,256-token vocabulary)
DRAFT_MODEL_LLAMA_NAME  = "unsloth/Llama-3.2-1B-Instruct"
TARGET_MODEL_LLAMA_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
QUANTIZATION_BITS_LLAMA = 4    # 4-bit NF4 for 8B target

# Qwen2.5 pair (151,936-token vocabulary)
DRAFT_MODEL_QWEN_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_QWEN_NAME = "Qwen/Qwen2.5-7B-Instruct"
QUANTIZATION_BITS_QWEN = 4    # 4-bit NF4 for 7B target

MAX_NEW_TOKENS      = 128
DRAFT_STEPS_LIST    = [1, 3, 5, 7, 10]
DEFAULT_DRAFT_STEPS = 5
TEMPERATURE         = 0.0    # greedy decoding; at T=0.0 accept/reject is fully
#                              deterministic — no stochastic draw, σ=0.000 across seeds

NUM_SAMPLES_QA    = 500   # XQuAD-TR (per seed)
NUM_SAMPLES_SUM   = 500   # TR-News (per seed)
NUM_SAMPLES_EN    = 500   # SQuAD (single seed)
NUM_SAMPLES_LLAMA = 300   # reduced — 8B is slower per sample
NUM_SAMPLES_QWEN  = 300   # reduced — 7B is slower per sample

QUANTIZATION_BITS = 0    # 0 = float16 for GPT-2 scale models
```

---

## Citation

If you use this codebase or paper in your research, please cite:

```bibtex
@misc{bayram2025speculative,
  title   = {Does Agglutinative Morphology Impede Speculative Decoding?
             A Controlled Study on Turkish and English},
  author  = {Bayram, Cengizhan},
  year    = {2025},
  url     = {https://github.com/CengizhanBayram/Speculative_decoding}
}
```

### References

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). **Fast Inference from Transformers via Speculative Decoding.** ICML 2023.
- Chen, C., et al. (2023). **Accelerating Large Language Model Decoding with Speculative Sampling.** arXiv:2302.01318.
- Safaya, A., et al. (2022). **Mukayese: Turkish NLP Strikes Back.** ACL 2022 Findings.
- Artetxe, M., et al. (2020). **On the Cross-lingual Transferability of Monolingual Representations.** ACL 2020.
- Baykara, B., & Güngör, T. (2022). **Abstractive Text Summarization and New Large-Scale Datasets for Agglutinative Languages.** Language Resources and Evaluation.
- Kaya, Y. B., & Tantuğ, A. C. (2022). **Effect of Tokenization Granularity for Turkish Large Language Models.** arXiv:2204.11454.
