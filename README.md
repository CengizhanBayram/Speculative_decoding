# Does Agglutinative Morphology Impede Speculative Decoding?

> **Evidence That Corpus Focus, Not Morphology, Governs Speculative Decoding Acceptance**
>
> A controlled study on Turkish and English using matched GPT-2 family model pairs, a same-corpus English control (Pythia), morphological fragmentation and Stanza analyses, temperature sweep, and cross-scale experimentation.

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

This repository presents the **first systematic empirical study of speculative decoding on Turkish** — an agglutinative language whose rich suffix chains fragment into 1.40× more BPE subword tokens per word than English (Cohen's *d* = 0.626, *p* ≈ 0). Contrary to the expectation that agglutination would hurt the draft model, Turkish acceptance rates *exceed* English (α = 0.768 vs. 0.727; Δ = 4.1 pp, *p* ≈ 0).

The explanation is **corpus focus**: the `ytu-ce-cosmos` draft and target were pre-trained on the same focused 35 GB Turkish corpus, producing stronger argmax alignment than either the cross-run OpenAI GPT-2 family or a same-corpus-but-diverse English control (Pythia 160M→1B on the 300B-token Pile, α = 0.687). This rules out "same-corpus alone ⇒ better acceptance" — the homogeneity and domain focus of the training corpus is the decisive factor.

At **T = 0.0** (greedy decoding), the accept/reject step is **fully deterministic**: a draft token is accepted if and only if it equals the target's argmax, so acceptance rates are perfectly reproducible (σ ≈ 10⁻¹⁶ across three independent seeds). Speculative outputs are empirically lossless (ΔROUGE-1 < 0.001). A temperature sweep T ∈ {0.3, 0.7, 1.0} shows monotone acceptance decay for both languages; latency rises above the greedy baseline near T ≈ 0.7.

A cross-scale experiment across **eight draft–target pairs** shows that the **draft-to-target parameter ratio** and **corpus focus** — not language morphology — are the primary efficiency levers. Small-draft GPT-2 pairs (1:6.6) achieve speedup (1.230× TR, 1.163× EN); medium-draft pairs (1:2.2) and all three 7B-scale pairs we test (Llama-3 native EN, Llama-3 cross-lingual TR, Qwen-2.5) incur slowdown on consumer L4 GPUs.

The implementation uses a **target-side KV cache**: the target model is initialised once on the full prompt (O(L) cost) and thereafter called only on the γ draft tokens per iteration (O(γ)), eliminating the O(L) re-encoding cost of a naive implementation.

---

## Key Findings

| Finding | Value |
|---------|-------|
| Turkish acceptance rate (small draft, T=0) | α = 0.768 |
| English acceptance rate (GPT-2, T=0) | α = 0.727 |
| TR vs EN gap | Δ = 4.1 pp (*p* ≈ 0, Mann-Whitney) |
| Pythia EN same-corpus control (M1) | α = 0.687, 0.770× (slowdown) |
| σ across seeds (T = 0.0) | ≈ 10⁻¹⁶ (fully deterministic) |
| TR small-draft speedup | 1.230× (Wilcoxon *p* < 10⁻¹⁴⁷) |
| EN small-draft speedup | 1.163× (Wilcoxon *p* < 10⁻⁸⁵) |
| TR medium-draft speedup | 0.902× (slowdown) |
| BPE fragmentation ratio TR/EN | 1.40× (Cohen's *d* = 0.626) |
| Stanza morphological features TR/EN | 2.30× (4.21 vs. 1.83 features/word) |
| TR acceptance at T=0.3 / T=0.7 / T=1.0 | 0.744 / 0.523 / 0.415 |
| Llama-3 native EN α (1B→8B) | 0.546, 0.692× (slowdown) |
| Llama-3 cross-lingual TR α (1B→8B) | 0.448, 0.800× (slowdown) |
| Qwen-2.5 TR α (0.5B→7B) | 0.370, 0.466× (slowdown) |

---

## Key Contributions

- **First cross-lingual study of speculative decoding on Turkish** with a fully symmetric 3-seed design (3 seeds × 1,000 samples per language = 3,000 samples per language; QA + summarisation).
- Full implementation of the **accept/reject speculative decoding algorithm** (Leviathan et al., 2023) with target-side KV cache, T>0 support, and per-token logging.
- **Same-corpus English control (M1 experiment):** Pythia-160M→1B on the 300B-token Pile, providing the first same-corpus English baseline. Result (α = 0.687, 0.770× slowdown) shows same-corpus co-training is necessary but not sufficient; corpus focus is the additional decisive factor.
- **Temperature sensitivity sweep** T ∈ {0.3, 0.7, 1.0}: monotone acceptance decay for both languages; latency crosses the greedy baseline near T ≈ 0.7, establishing T = 0.0 as the only robust speedup regime.
- Controlled comparison across **eight draft–target pairs**: Turkish GPT-2 small/medium, English GPT-2 small/medium, Pythia 160M/1B, Llama-3.2 native EN (1B→8B), Llama-3 cross-lingual TR (1B→8B), and Qwen-2.5 (0.5B→7B).
- **3-seed robustness evaluation** confirming deterministic acceptance (σ ≈ 10⁻¹⁶) at T = 0.0 for both languages.
- **γ ablation study** (γ ∈ {1, 3, 5, 7, 10}) on a mixed QA + summarisation subset for both Turkish and English, showing optimal γ is language-independent.
- **Position-bucket analysis**: acceptance by early/mid/late thirds of the generated sequence — rules out morphological-error accumulation during generation.
- **Subword fragmentation analysis** (raw + cleaned re-analysis removing URL, non-Latin, and foreign-name artifacts) and **Stanza tokenizer-agnostic morphological feature comparison** (TR/EN = 2.30×).
- **Tokenizer comparison** across four model families (ytu-cosmos BPE, OpenAI GPT-2, Llama-3, Qwen-2.5) quantifying TR/EN token-count ratios.
- **Rejected-token frequency analysis** aggregated into coarse morphosyntactic categories (punctuation, function words, clitics, content fragments).
- **Hardware timing profile** (GPT-2 small and Llama EN): per-phase latency breakdown explaining the medium-draft and Pythia slowdowns.
- **Output quality validation**: ROUGE-1/2/L + BLEU confirming empirical losslessness (Δ < 0.001 for all metrics).
- Complete statistical battery: Wilcoxon signed-rank, Mann-Whitney U, Cohen's *d*, bootstrap confidence intervals.
- Five publication-quality figures (PDF + PNG, 300 dpi, serif font).
- Publicly released code, result CSVs, additional experiment results, and paper draft for full reproducibility.

---

## Architecture

```
ALL business logic lives in .py files inside src/.
The notebooks contain ZERO logic —
they only clone the repo, install deps, import from src/, and call functions.
```

This strict separation means:

- Every function is unit-testable in isolation.
- Results are 100% reproducible by re-running the notebooks.
- The notebooks act as reproducible experiment scripts, not development environments.

---

## Repository Structure

```
Speculative_decoding/
│
├── src/
│   ├── __init__.py          # empty
│   ├── config.py            # all constants and hyperparameters
│   ├── utils.py             # seed_everything, save_json, check_gpu, git_push
│   ├── data.py              # dataset loaders: XQuAD-TR, TR-News, SQuAD-EN, CNN/DailyMail
│   ├── models.py            # draft (float16) and target (float16 / NF4) model loaders
│   ├── speculative.py       # accept/reject algorithm + target KV cache, T>0 support
│   ├── metrics.py           # ROUGE, BLEU, bootstrap CI, Wilcoxon, Cohen d, speedup
│   ├── linguistic.py        # position/fragmentation/rejection/Stanza analysis
│   └── figures.py           # 5 publication-quality figure generators
│
├── paper/
│   ├── main.tex             # paper source (ACL-style, multicol layout)
│   ├── references.bib       # 21 BibTeX entries
│   └── figures/             # PDF figure files included by main.tex
│
├── results/                 # main experiment CSVs (GPT-2 TR/EN, Llama, Qwen, ablation)
│
├── additional_results/      # M1 Pythia control + gamma ablation (mixed TR/EN)
│   ├── speculative_pythia_en_results.csv
│   ├── baseline_pythia_en_results.csv
│   ├── ablation_gamma_gpt2_tr_mixed.csv
│   ├── ablation_gamma_gpt2_en.csv
│   ├── ablation_gamma_pythia_en.csv
│   ├── timing_profile.json
│   └── additional_experiments_summary.json
│
├── run_experiments.ipynb            # main experiment notebook (GPT-2, Llama, Qwen)
├── run_additional_experiments.ipynb # M1 Pythia + mixed ablation + timing profile
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
4. For additional experiments (Pythia M1, mixed γ ablation, timing profile), run `run_additional_experiments.ipynb` in a separate session.

### Option B — Local

```bash
git clone https://github.com/CengizhanBayram/Speculative_decoding.git
cd Speculative_decoding
pip install -r requirements.txt
```

**Minimum hardware:**

| Experiment | VRAM needed |
|------------|-------------|
| GPT-2 scale (TR + EN, all four pairs) | ≥ 4 GB |
| Pythia 160M → 1B (M1 experiment) | ≥ 4 GB |
| Llama-3.2-1B → Turkish-Llama-8B (4-bit) | ≥ 6 GB |
| Llama-3.2-1B → Llama-3.1-8B (native EN, 4-bit) | ≥ 6 GB |
| Qwen2.5-0.5B → Qwen2.5-7B-Instruct (4-bit) | ≥ 5 GB |

Tested on: L4 24 GB (Colab Pro). Full pipeline ≈ 12 h wall-clock on L4.

---

## Running Experiments

### Main notebook (`run_experiments.ipynb`)

Orchestrates GPT-2 TR/EN (all seeds), Llama, and Qwen experiments.

| Cell | Action | Output |
|------|--------|--------|
| 1 | Mount Drive, enter tokens, HF login | — |
| 2 | Clone / pull repo, `pip install` | — |
| 3 | `sys.path` + all imports from `src/` | — |
| 4 | `seed_everything(42)`, `check_gpu()` | GPU info dict |
| 5 | Load datasets: XQuAD-TR, TR-News, SQuAD-EN, CNN/DailyMail | 4 × list[dict] |
| 6 | Load TR-small draft + target (117M + 774M) | 2 models |
| 6b | Load TR-medium draft (354M; reuses target) | 1 model |
| 6c | Load EN-small draft + target (gpt2 + gpt2-large) | 2 models |
| 6d | Load EN-medium draft (gpt2-medium; reuses target) | 1 model |
| 7 | Greedy baseline — Turkish | `baseline_tr_results.csv` |
| 7b | Greedy baseline — English | `baseline_en_results.csv` |
| 8 | Speculative — TR-small (3 seeds × 1,000 samples) | `speculative_tr_small_seed{s}.csv` × 3 |
| 8b | Speculative — TR-medium (γ=5) | `speculative_tr_medium_results.csv` |
| 9 | Speculative — EN-small (3 seeds × 1,000 samples) | `speculative_en_small_seed{s}.csv` × 3 |
| 9b | Speculative — EN-medium (γ=5) | `speculative_en_medium_results.csv` |
| 10 | Temperature sweep T ∈ {0.3, 0.7, 1.0} (100 samples/lang) | `temperature_sweep.csv` |
| 6e | Free GPT-2 memory; load Llama-3.2-1B + Turkish-Llama-8B (4-bit) | 2 models |
| 7c | Greedy baseline — Turkish-Llama-8B | `baseline_llama_results.csv` |
| 9c | Speculative — Llama 1B→8B (TR cross-lingual) | `speculative_llama_results.csv` |
| 9c-en | Speculative — Llama 1B→8B (EN native) | `speculative_llama_en_results.csv` |
| 6f | Free Llama memory; load Qwen2.5-0.5B + Qwen2.5-7B-Instruct (4-bit) | 2 models |
| 7d | Greedy baseline — Qwen2.5-7B | `baseline_qwen_results.csv` |
| 9d | Speculative — Qwen 0.5B→7B | `speculative_qwen_results.csv` |
| 11 | Full statistical test battery | `statistical_tests.json` |
| 11b | Output quality: ROUGE + BLEU | `quality_metrics.csv` |
| 12 | Position / fragmentation / Stanza / rejected-token analysis | 8 × `.csv` |
| 13 | Generate all 5 figures | 10 files (5 × PDF + PNG) |
| 14 | `git_push(f"results: {timestamp}")` | commit hash |

### Additional experiments notebook (`run_additional_experiments.ipynb`)

Runs M1 Pythia control, mixed-task γ ablation (both languages), and timing profile.

| Cell | Action | Output |
|------|--------|--------|
| M1 | Load Pythia-160M + Pythia-1B; greedy + speculative (3 seeds) | `speculative_pythia_en_results.csv` |
| M5-TR | γ ablation — TR mixed (50 QA + 50 SUM, γ ∈ {1,3,5,7,10}) | `ablation_gamma_gpt2_tr_mixed.csv` |
| M5-EN | γ ablation — EN mixed (same setup) | `ablation_gamma_gpt2_en.csv` |
| M5-Py | γ ablation — Pythia mixed | `ablation_gamma_pythia_en.csv` |
| M2 | Per-phase timing profile (GPT-2 TR, Llama EN) | `timing_profile.json` |
| — | Aggregate summary | `additional_experiments_summary.json` |

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

### SQuAD — English Question Answering

- **Source:** `rajpurkar/squad`, validation split (500 samples per seed).
- **Prompt format:**
  ```
  Question: {question}
  Context: {context[:300]}
  Answer:
  ```

### CNN/DailyMail — English Summarisation

- **Source:** `cnn_dailymail` v3.0.0, validation split (500 samples per seed). Symmetric counterpart to TR-News.
- **Prompt format:**
  ```
  Summarise the following article:
  {article[:400]}
  Summary:
  ```

All datasets are shuffled with the configured seed before sampling.

---

## Models

All draft–target pairs within a family **share the same tokenizer and vocabulary** — a hard requirement of speculative decoding.

### GPT-2 Turkish Pairs (`ytu-ce-cosmos`)

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft (small) | `ytu-ce-cosmos/turkish-gpt2` | ~117 M | float16 |
| Draft (medium) | `ytu-ce-cosmos/turkish-gpt2-medium` | ~354 M | float16 |
| Target | `ytu-ce-cosmos/turkish-gpt2-large` | ~774 M | float16 |

All three models share a GPT-2 BPE tokenizer (50,257 tokens) and were pre-trained on the **same focused 35 GB Turkish corpus** — the primary enabler of high acceptance rates (α = 0.768).

### GPT-2 English Pairs (OpenAI)

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft (small) | `gpt2` | ~117 M | float16 |
| Draft (medium) | `gpt2-medium` | ~354 M | float16 |
| Target | `gpt2-large` | ~774 M | float16 |

Separately trained on WebText; same architecture and vocabulary size as the Turkish family but different merge table.

### Pythia English Pair (M1 same-corpus control)

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `EleutherAI/pythia-160m` | ~160 M | float16 |
| Target | `EleutherAI/pythia-1b` | ~1 B | float16 |

Both trained on the Pile (300 B tokens, 143 K gradient steps each; Biderman et al., 2023). Uses the GPT-NeoX tokenizer (50,257 tokens, different merge table from GPT-2). Acceptance α = 0.687, speedup 0.770× — showing same-corpus on a large diverse corpus is insufficient for tight draft–target agreement.

### Llama-3 Cross-Lingual Pair (Turkish)

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `unsloth/Llama-3.2-1B-Instruct` | ~1 B | float16 |
| Target | `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` | ~8 B | 4-bit NF4 |

Cross-lingual pair: English-base draft + Turkish instruction-tuned target. α = 0.448 (−32 pp vs. Turkish same-corpus pair).

### Llama-3 Native English Pair

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `unsloth/Llama-3.2-1B-Instruct` | ~1 B | float16 |
| Target | `meta-llama/Llama-3.1-8B-Instruct` | ~8 B | 4-bit NF4 |

Native English pair providing a clean cross-lingual penalty isolator. α = 0.546 (+9.8 pp over cross-lingual), speedup 0.692× — still slowdown because 8B target is compute-bound on L4.

### Qwen-2.5 Pair

| Role | Model | Params | Dtype |
|------|-------|--------|-------|
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | ~0.5 B | float16 |
| Target | `Qwen/Qwen2.5-7B-Instruct` | ~7 B | 4-bit NF4 |

Same-family pair on Turkish text. α = 0.370 (−40 pp vs. Turkish same-corpus pair), speedup 0.466×.

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

  Accept / reject:
    T = 0.0 (deterministic):  accept iff dᵢ == argmax p_target
    T > 0.0 (stochastic):     draw uᵢ ~ U(0,1); accept iff uᵢ ≤ min(1, p_t/p_d)
    On rejection at i*: sample corrected token from max(0, p_t − p_d), restart

  Update:
    4. If all γ accepted → sample bonus token from target at position γ
    5. Target KV cache updated by ≤ γ+1 tokens (never re-encodes full context)
```

At **T = 0.0** the accept/reject criterion is fully deterministic: a draft token is accepted if and only if it equals the target model's argmax. This guarantees σ ≈ 10⁻¹⁶ acceptance variance across seeds and produces **identical output** to autoregressive greedy decoding (verified by ΔROUGE-1 < 0.001).

### Ablation Study (Mixed TR + EN)

γ ∈ {1, 3, 5, 7, 10} on 50 QA + 50 summarisation samples per language (seed 42). Greedy baselines: TR 3,820 ms, EN 3,790 ms.

| γ | TR α | TR lat (ms) | TR speedup | EN α | EN speedup |
|---|------|------------|-----------|------|-----------|
| 1 | 0.926 | 4,723 | 0.809× | 0.903 | 0.799× |
| 3 | 0.860 | 3,352 | 1.140× | 0.820 | 1.091× |
| 5 | 0.792 | 3,014 | 1.268× | 0.743 | 1.235× |
| 7 | 0.732 | 2,887 | 1.323× | 0.679 | 1.263× |
| 10 | 0.664 | 2,866 | 1.333× | 0.602 | 1.274× |

Acceptance decreases with γ (compounding draft error); latency decreases monotonically. **γ = 5** is recommended as the default. The optimal γ is the same for both languages (plateau at γ = 7–10), confirming agglutinative morphology does not shift the optimal speculation width.

---

## Statistical Analysis

All tests are run in `src/metrics.py` and saved to `results/statistical_tests.json`.

| Test | Applied to | Purpose |
|------|-----------|---------|
| Wilcoxon signed-rank | Paired latencies (greedy vs. speculative) | Non-parametric significance of speedup |
| Mann-Whitney U | TR vs. EN acceptance rates | Cross-linguistic significance |
| Cohen's *d* | Latency and fragmentation distributions | Effect size |
| Bootstrap CI (n=10,000) | Acceptance rates, speedup ratios | 95% confidence intervals |
| Two-proportion *z*-test | TR vs. EN at T > 0 | Significance of cross-lingual gap at higher temperatures |

**Seed stability:** At T = 0.0, acceptance is fully deterministic — σ ≈ 10⁻¹⁶ (machine epsilon) across all three seeds for both TR and EN small-draft. Latency varies by ≤ 9 ms (≤ 0.3%) attributable entirely to GPU scheduling jitter.

All primary significance claims survive Bonferroni correction at α = 0.05 for K = 8 tests (every *p* < 10⁻⁴⁰).

---

## Linguistic Analysis

Implemented in `src/linguistic.py`.

### Position-Bucket Analysis

Each sample's generated sequence is split into **early / mid / late** thirds. Turkish acceptance rates increase from 87.5% → 93.3% → 93.7%; English shows an even stronger increase (83.4% → 90.1% → 91.2%). Both languages exhibit the same pattern, ruling out TR-specific morphological accumulation.

### Subword Fragmentation (BPE-based)

Turkish prompts require 1.40× more BPE subword tokens per word than English (mean 2.156 vs. 1.545 fragments/word; Cohen's *d* = 0.626, *p* ≈ 0). A cleaned re-analysis removing URL, non-Latin, and foreign proper-noun artifacts reduces the ratio to 1.34× (Cohen's *d* = 0.564) — both large and highly significant.

A Spearman rank correlation between per-word fragment count and per-word acceptance rate yields ρ = +0.138 (*p* = 0.423) — non-significant and in the *positive* direction, meaning higher fragmentation does not predict lower acceptance.

### Stanza Morphological Feature Analysis

Using the Stanza pipeline (Qi et al., 2020), Turkish content words carry **2.30×** more Universal Dependencies morphological feature tags than English (4.21 vs. 1.83 features/word). This tokenizer-agnostic measure confirms Turkish morphological complexity is real even after artefact cleaning — yet does not predict lower acceptance, reinforcing the corpus-focus hypothesis.

### Tokenizer Comparison

A fixed 100-word Turkish paragraph encoded under four tokenizer families:

| Tokenizer | Vocab | TR tokens | TR/EN ratio |
|-----------|-------|-----------|-------------|
| ytu-cosmos BPE | 50,257 | 213 | 1.32 |
| OpenAI GPT-2 | 50,257 | 287 | 1.78 |
| Llama-3 | 128,256 | 252 | 1.56 |
| Qwen-2.5 | 151,936 | 233 | 1.45 |

### Rejected-Token Frequency Analysis

`rejected_token_analysis` counts total proposals and rejections per token surface form. Top rejected tokens: demonstratives (*bu*: 24.1%), hyphens (22.4%), conjunctions (*ve*: 20.4%). Punctuation has the lowest rejection rates (period: 8.5%). Aggregated into four coarse categories: punctuation (12.3%), function words (21.2%), clitics (18.4%), content-word fragments (9.4%).

---

## Figures

All figures saved as PDF (vector) and PNG (raster, 300 dpi) to `paper/figures/`. Style: serif font, 11 pt — ready for ACL/EMNLP submission.

| File stem | Content |
|-----------|---------|
| `latency_violin` | Violin + box plots of per-sample speedup for all draft–target pairs |
| `model_comparison` | Side-by-side grouped bars: acceptance rate + median speedup across all pairs |
| `ablation_gamma` | Dual-panel: acceptance rate + latency vs. γ (TR and EN, with IQR shading) |
| `position_acceptance` | Bar chart of acceptance rates in early / mid / late token position buckets |
| `fragmentation_comparison` | Two-panel: fragment distribution histogram + per-task mean fragments, TR vs. EN |
| `temperature_sweep` | Dual-panel: acceptance rate + latency vs. temperature (TR and EN) |

---

## Results Layout

After a full run, `results/` and `additional_results/` contain:

```
results/
├── baseline_tr_results.csv               # greedy baseline — Turkish (GPT-2)
├── baseline_en_results.csv               # greedy baseline — English (GPT-2)
├── baseline_llama_results.csv            # greedy baseline — Turkish-Llama-8B
├── baseline_llama_en_results.csv         # greedy baseline — Llama-3.1-8B (EN native)
├── baseline_qwen_results.csv             # greedy baseline — Qwen2.5-7B
│
├── speculative_tr_small_seed42.csv       # TR-small, seed 42  (1,000 samples)
├── speculative_tr_small_seed123.csv      # TR-small, seed 123
├── speculative_tr_small_seed456.csv      # TR-small, seed 456
├── speculative_tr_small_results.csv      # TR-small, all seeds combined (3,000 samples)
├── speculative_tr_medium_results.csv     # TR-medium (354M draft), γ=5
│
├── speculative_en_small_seed42.csv       # EN-small, seed 42  (1,000 samples)
├── speculative_en_small_seed456.csv      # EN-small, seed 456
├── speculative_en_medium_results.csv     # EN-medium (354M draft), γ=5
│
├── speculative_llama_results.csv         # Llama 1B→Turkish-8B (cross-lingual)
├── speculative_llama_en_results.csv      # Llama 1B→Llama-3.1-8B (native EN)
├── speculative_qwen_results.csv          # Qwen 0.5B→7B
│
├── ablation_gamma.csv                    # γ ablation (original TR QA-only)
├── temperature_sweep.csv                 # T ∈ {0.3, 0.7, 1.0} sweep (TR + EN)
├── statistical_tests.json               # all statistical test outputs
├── quality_metrics.csv                  # ROUGE + BLEU: greedy vs. speculative
│
├── position_acceptance.csv              # acceptance rate per position bucket (TR + EN)
├── oov_analysis_tr.csv                  # Turkish subword fragmentation per word
├── oov_analysis_en.csv                  # English subword fragmentation per word
├── fragmentation_acceptance.csv         # Spearman ρ: fragment count vs. acceptance
├── rejected_tokens_tr.csv              # top-30 most-proposed tokens + rejection rates (TR)
├── rejected_tokens_en.csv              # same for English
├── stanza_morphology_tr.csv            # Stanza morphological features (Turkish)
└── stanza_morphology_en.csv            # Stanza morphological features (English)

additional_results/
├── speculative_pythia_en_results.csv    # Pythia 160M→1B speculative (M1 experiment)
├── baseline_pythia_en_results.csv       # Pythia greedy baseline
├── ablation_gamma_gpt2_tr_mixed.csv     # γ ablation TR mixed (50 QA + 50 SUM)
├── ablation_gamma_gpt2_en.csv           # γ ablation EN mixed
├── ablation_gamma_pythia_en.csv         # γ ablation Pythia
├── timing_profile.json                  # per-phase latency: GPT-2 TR, Llama EN
└── additional_experiments_summary.json  # aggregated summary of all additional runs
```

CSV columns for speculative results:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Input prompt |
| `reference` | str | Ground-truth answer/summary |
| `task` | str | `qa_tr`, `summarization_tr`, `qa_en`, `summarization_en` |
| `mode` | str | `speculative` / `greedy` |
| `draft_steps` | int | γ value used |
| `generated_text` | str | Model output |
| `acceptance_rate` | float | Fraction of draft tokens accepted |
| `num_target_calls` | int | Number of target forward passes |
| `latency_ms` | float | Wall-clock generation time (ms) |
| `gamma_setting` | int | γ (present in ablation files) |
| `seed` | int | Random seed (present in multi-seed runs) |

---

## Configuration

All hyperparameters live in `src/config.py`:

```python
SEED  = 42
SEEDS = [42, 123, 456]   # seeds for TR and EN 3-seed robustness runs

# Turkish GPT-2 pairs (same 50,257-token BPE vocabulary + training corpus)
DRAFT_MODEL_TR_SMALL_NAME  = "ytu-ce-cosmos/turkish-gpt2"          # 117 M
DRAFT_MODEL_TR_MEDIUM_NAME = "ytu-ce-cosmos/turkish-gpt2-medium"   # 354 M
TARGET_MODEL_TR_NAME       = "ytu-ce-cosmos/turkish-gpt2-large"    # 774 M

# English GPT-2 pairs (50,257-token BPE, English-trained)
DRAFT_MODEL_EN_SMALL_NAME  = "gpt2"           # 117 M
DRAFT_MODEL_EN_MEDIUM_NAME = "gpt2-medium"    # 354 M
TARGET_MODEL_EN_NAME       = "gpt2-large"     # 774 M

# Pythia same-corpus English control (GPT-NeoX tokenizer, 50,257 tokens)
DRAFT_MODEL_PYTHIA_NAME  = "EleutherAI/pythia-160m"  # 160 M
TARGET_MODEL_PYTHIA_NAME = "EleutherAI/pythia-1b"    # 1 B

# Llama-3 cross-lingual pair (128,256-token vocabulary)
DRAFT_MODEL_LLAMA_NAME    = "unsloth/Llama-3.2-1B-Instruct"
TARGET_MODEL_LLAMA_TR_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
TARGET_MODEL_LLAMA_EN_NAME = "meta-llama/Llama-3.1-8B-Instruct"
QUANTIZATION_BITS_LLAMA   = 4    # 4-bit NF4 for 8B targets

# Qwen2.5 pair (151,936-token vocabulary)
DRAFT_MODEL_QWEN_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_QWEN_NAME = "Qwen/Qwen2.5-7B-Instruct"
QUANTIZATION_BITS_QWEN = 4    # 4-bit NF4 for 7B target

MAX_NEW_TOKENS      = 128
DRAFT_STEPS_LIST    = [1, 3, 5, 7, 10]
DEFAULT_DRAFT_STEPS = 5
TEMPERATURE         = 0.0    # greedy; at T=0.0 accept/reject is fully deterministic
TEMPERATURE_SWEEP   = [0.3, 0.7, 1.0]

NUM_SAMPLES_QA    = 500   # XQuAD-TR / SQuAD-EN (per seed)
NUM_SAMPLES_SUM   = 500   # TR-News / CNN-DailyMail (per seed)
NUM_SAMPLES_LLAMA = 300   # reduced — 8B is slower per sample
NUM_SAMPLES_QWEN  = 300
NUM_SAMPLES_TEMP  = 50    # per task for temperature sweep (100 total per language)

QUANTIZATION_BITS = 0    # 0 = float16 for GPT-2 / Pythia scale models
```

---

## Citation

If you use this codebase or paper in your research, please cite:

```bibtex

```

### References

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). **Fast Inference from Transformers via Speculative Decoding.** ICML 2023.
- Chen, C., et al. (2023). **Accelerating Large Language Model Decoding with Speculative Sampling.** arXiv:2302.01318.
- Biderman, S., et al. (2023). **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.** ICML 2023.
- Safaya, A., et al. (2022). **Mukayese: Turkish NLP Strikes Back.** ACL 2022 Findings.
- Artetxe, M., et al. (2020). **On the Cross-lingual Transferability of Monolingual Representations.** ACL 2020.
- Qi, P., et al. (2020). **Stanza: A Python Natural Language Processing Toolkit for Many Human Languages.** ACL 2020.
- Baykara, B., & Güngör, T. (2022). **Abstractive Text Summarization and New Large-Scale Datasets for Agglutinative Languages.** Language Resources and Evaluation.
- Kaya, Y. B., & Tantuğ, A. C. (2022). **Effect of Tokenization Granularity for Turkish Large Language Models.** arXiv:2204.11454.
