"""
Patch run_experiments.ipynb to add Llama-3 and Qwen2.5 model family experiments.

New cells inserted (in order):
  cell-03-imports   : updated — new config constants + new data loaders
  cell-05-data      : updated — load instruct-format samples
  cell-06e-llama    : load Llama-3.2-1B draft + Turkish-Llama-8b-Instruct target (4-bit)
  cell-06f-qwen     : load Qwen2.5-0.5B draft + Qwen2.5-7B-Instruct target (4-bit)
  cell-07c-base-llama : greedy baseline with Llama-8b target
  cell-07d-base-qwen  : greedy baseline with Qwen-7B target
  cell-09c-spec-llama : speculative — Llama-1B draft → Turkish-Llama-8b (3 seeds)
  cell-09d-spec-qwen  : speculative — Qwen-0.5B draft → Qwen-7B (3 seeds)
  cell-11-stats     : updated — add Llama + Qwen stats
  cell-13-figures   : updated — pass llama/qwen results
"""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells      = nb["cells"]
cell_by_id = {c["id"]: c for c in cells}


def make_cell(cid, src):
    return {
        "id": cid, "cell_type": "code",
        "metadata": {}, "outputs": [], "execution_count": None,
        "source": [src],
    }


# ── Cell 3: add new imports ───────────────────────────────────────────────────
cell_by_id["cell-03-imports"]["source"] = ["""\
# ── Cell 3: Add repo to path and import everything from src/ ─────────────────
import sys
sys.path.insert(0, REPO_DIR)

from src.config import (
    SEED, SEEDS,
    DRAFT_MODEL_TR_SMALL_NAME, DRAFT_MODEL_TR_MEDIUM_NAME, TARGET_MODEL_TR_NAME,
    DRAFT_MODEL_EN_SMALL_NAME, DRAFT_MODEL_EN_MEDIUM_NAME, TARGET_MODEL_EN_NAME,
    DRAFT_MODEL_LLAMA_NAME, TARGET_MODEL_LLAMA_NAME, QUANTIZATION_BITS_LLAMA,
    DRAFT_MODEL_QWEN_NAME,  TARGET_MODEL_QWEN_NAME,  QUANTIZATION_BITS_QWEN,
    MAX_NEW_TOKENS, DRAFT_STEPS_LIST, DEFAULT_DRAFT_STEPS,
    NUM_SAMPLES_QA, NUM_SAMPLES_SUM, NUM_SAMPLES_EN,
    NUM_SAMPLES_LLAMA, NUM_SAMPLES_QWEN,
    QUANTIZATION_BITS, RESULTS_DIR, FIGURES_DIR,
)
from src.utils      import seed_everything, save_json, check_gpu, git_push
from src.data       import (
    load_xquad_tr, load_trnews, load_squad_en,
    load_xquad_tr_instruct, load_squad_en_instruct,
)
from src.models     import load_draft_model, load_target_model
from src.speculative import run_experiment
from src.metrics    import (
    compute_task_metrics, bootstrap_ci,
    wilcoxon_test, cohens_d, mann_whitney_test,
    compute_speedup, run_all_statistical_tests,
)
from src.linguistic import (
    position_acceptance_analysis,
    oov_analysis,
    fragmentation_acceptance_analysis,
    rejected_token_analysis,
)
from src.figures import generate_all_figures
import pandas as pd

print('All imports successful.')
print(f'Primary seed               : {SEED}')
print(f'Seeds (small-draft robust) : {SEEDS}')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results -> {RESULTS_DIR}")
print(f"Figures -> {FIGURES_DIR}")\
"""]

# ── Cell 5: add instruct-format sample loading ────────────────────────────────
cell_by_id["cell-05-data"]["source"] = ["""\
# ── Cell 5: Load all datasets ─────────────────────────────────────────────────
seed_everything(SEED)

# GPT-2 style (completion format)
tr_qa_samples  = load_xquad_tr(NUM_SAMPLES_QA,  SEED)
tr_sum_samples = load_trnews(  NUM_SAMPLES_SUM,  SEED)
tr_samples_all = tr_qa_samples + tr_sum_samples
squad_samples  = load_squad_en(NUM_SAMPLES_EN,   SEED)

# Instruct format — for Llama-8b and Qwen-7B targets
tr_instruct_samples = load_xquad_tr_instruct(NUM_SAMPLES_LLAMA, SEED)
en_instruct_samples = load_squad_en_instruct(NUM_SAMPLES_QWEN,  SEED)

print(f'TR GPT-2 samples     : {len(tr_samples_all):,}  ({NUM_SAMPLES_QA} QA + {NUM_SAMPLES_SUM} SUM)')
print(f'EN GPT-2 samples     : {len(squad_samples):,}')
print(f'TR instruct samples  : {len(tr_instruct_samples):,}  (for Llama target)')
print(f'EN instruct samples  : {len(en_instruct_samples):,}  (for Qwen target)')\
"""]

# ── New cells ─────────────────────────────────────────────────────────────────

cell_06e = make_cell("cell-06e-llama", """\
# ── Cell 6e: Load Llama-3 models ──────────────────────────────────────────────
# Draft : meta-llama/Llama-3.2-1B        (~1B, float16, ~2 GB VRAM)
# Target: Turkish-Llama-8b-Instruct-v0.1 (~8B, 4-bit NF4, ~4 GB VRAM)
# Shared tokenizer: Llama-3 (128,256 tokens)
import gc, torch

print("Loading Llama draft (1B, float16)...")
draft_model_llama, draft_tok_llama = load_draft_model(
    DRAFT_MODEL_LLAMA_NAME, device="cuda:0"
)
print(f"  Draft device: {next(draft_model_llama.parameters()).device}")

print("Loading Llama target (8B, 4-bit)...")
target_model_llama, target_tok_llama = load_target_model(
    TARGET_MODEL_LLAMA_NAME, bits=QUANTIZATION_BITS_LLAMA
)
print("Llama models ready.")
check_gpu()\
""")

cell_06f = make_cell("cell-06f-qwen", """\
# ── Cell 6f: Load Qwen2.5 models ──────────────────────────────────────────────
# Draft : tr-Qwen2.5-0.5B-SFT-v1   (~0.5B, float16, ~1 GB VRAM)
# Target: Qwen2.5-7B-Instruct       (~7B,   4-bit NF4, ~3.5 GB VRAM)
# Shared tokenizer: Qwen2.5 (151,936 tokens)
# Cross-lingual: TR-adapted draft vs multilingual target

print("Loading Qwen draft (0.5B, float16)...")
draft_model_qwen, draft_tok_qwen = load_draft_model(
    DRAFT_MODEL_QWEN_NAME, device="cuda:0"
)
print(f"  Draft device: {next(draft_model_qwen.parameters()).device}")

print("Loading Qwen target (7B, 4-bit)...")
target_model_qwen, target_tok_qwen = load_target_model(
    TARGET_MODEL_QWEN_NAME, bits=QUANTIZATION_BITS_QWEN
)
print("Qwen models ready.")
check_gpu()\
""")

cell_07c = make_cell("cell-07c-baseline-llama", """\
# ── Cell 7c: Greedy baseline — Turkish-Llama-8b-Instruct ─────────────────────
seed_everything(SEED)
baseline_llama_df = run_experiment(
    samples        = tr_instruct_samples,
    target_model   = target_model_llama,
    target_tok     = target_tok_llama,
    mode           = 'greedy',
    max_new_tokens = MAX_NEW_TOKENS,
)
baseline_llama_df.drop(columns=['token_level_log']).to_csv(
    RESULTS_DIR / 'baseline_llama_results.csv', index=False
)
print(f"Llama greedy baseline  mean latency: {baseline_llama_df.latency_ms.mean():.1f} ms")
print(f"                     median latency: {baseline_llama_df.latency_ms.median():.1f} ms")\
""")

cell_07d = make_cell("cell-07d-baseline-qwen", """\
# ── Cell 7d: Greedy baseline — Qwen2.5-7B-Instruct ───────────────────────────
seed_everything(SEED)
baseline_qwen_df = run_experiment(
    samples        = en_instruct_samples,
    target_model   = target_model_qwen,
    target_tok     = target_tok_qwen,
    mode           = 'greedy',
    max_new_tokens = MAX_NEW_TOKENS,
)
baseline_qwen_df.drop(columns=['token_level_log']).to_csv(
    RESULTS_DIR / 'baseline_qwen_results.csv', index=False
)
print(f"Qwen greedy baseline  mean latency: {baseline_qwen_df.latency_ms.mean():.1f} ms")
print(f"                    median latency: {baseline_qwen_df.latency_ms.median():.1f} ms")\
""")

cell_09c = make_cell("cell-09c-spec-llama", """\
# ── Cell 9c: Speculative — Llama-3.2-1B → Turkish-Llama-8b-Instruct (3 seeds) ─
# Primary comparison: modern instruction-tuned Turkish model.
# 1B base draft + 8B instruct target — tests the base→instruct distribution gap.
seed_frames_llama = []
for s in SEEDS:
    seed_everything(s)
    _df = run_experiment(
        samples        = tr_instruct_samples,
        draft_model    = draft_model_llama,
        draft_tok      = draft_tok_llama,
        target_model   = target_model_llama,
        target_tok     = target_tok_llama,
        mode           = 'speculative',
        draft_steps    = DEFAULT_DRAFT_STEPS,
        max_new_tokens = MAX_NEW_TOKENS,
    )
    _df['seed'] = s
    seed_frames_llama.append(_df)
    _df.drop(columns=['token_level_log']).to_csv(
        RESULTS_DIR / f'speculative_llama_seed{s}.csv', index=False
    )
    ar = _df['acceptance_rate'].mean()
    sp = baseline_llama_df['latency_ms'].mean() / _df['latency_ms'].mean()
    print(f'  seed={s}  alpha={ar:.4f}  speedup={sp:.4f}')

speculative_llama_df = pd.concat(seed_frames_llama, ignore_index=True)
speculative_llama_df.drop(columns=['token_level_log']).to_csv(
    RESULTS_DIR / 'speculative_llama_results.csv', index=False
)
print(f"Llama alpha (mean): {speculative_llama_df.acceptance_rate.mean():.4f}")
print(speculative_llama_df.groupby('task')[['latency_ms','acceptance_rate']].mean().round(4))\
""")

cell_09d = make_cell("cell-09d-spec-qwen", """\
# ── Cell 9d: Speculative — Qwen2.5-0.5B-TR → Qwen2.5-7B-Instruct (3 seeds) ──
# Cross-lingual pair: Turkish-SFT draft vs multilingual target.
# Tests whether TR-adapted draft tokens align with a multilingual target's
# distribution — a realistic production scenario (localised draft + generic target).
seed_frames_qwen = []
for s in SEEDS:
    seed_everything(s)
    _df = run_experiment(
        samples        = en_instruct_samples,
        draft_model    = draft_model_qwen,
        draft_tok      = draft_tok_qwen,
        target_model   = target_model_qwen,
        target_tok     = target_tok_qwen,
        mode           = 'speculative',
        draft_steps    = DEFAULT_DRAFT_STEPS,
        max_new_tokens = MAX_NEW_TOKENS,
    )
    _df['seed'] = s
    seed_frames_qwen.append(_df)
    _df.drop(columns=['token_level_log']).to_csv(
        RESULTS_DIR / f'speculative_qwen_seed{s}.csv', index=False
    )
    ar = _df['acceptance_rate'].mean()
    sp = baseline_qwen_df['latency_ms'].mean() / _df['latency_ms'].mean()
    print(f'  seed={s}  alpha={ar:.4f}  speedup={sp:.4f}')

speculative_qwen_df = pd.concat(seed_frames_qwen, ignore_index=True)
speculative_qwen_df.drop(columns=['token_level_log']).to_csv(
    RESULTS_DIR / 'speculative_qwen_results.csv', index=False
)
print(f"Qwen alpha (mean): {speculative_qwen_df.acceptance_rate.mean():.4f}")
print(speculative_qwen_df.groupby('task')[['latency_ms','acceptance_rate']].mean().round(4))\
""")

# ── Update cell-11: add Llama + Qwen stats ────────────────────────────────────
cell_by_id["cell-11-stats"]["source"] = ["""\
# ── Cell 11: Statistical tests — all model families ──────────────────────────
import numpy as np
from src.metrics import compute_speedup, bootstrap_ci

stat_results = run_all_statistical_tests(
    baseline_df    = baseline_tr_df,
    spec_tr_df     = speculative_tr_df,
    spec_en_df     = speculative_en_df,
    baseline_en_df = baseline_en_df,
)

# ── Seed stability ────────────────────────────────────────────────────────────
for label, df in [('tr_small', speculative_tr_df), ('en_small', speculative_en_df)]:
    per_seed = df.groupby('seed')['acceptance_rate'].mean()
    stat_results[f'{label}_seed_mean'] = float(per_seed.mean())
    stat_results[f'{label}_seed_std']  = float(per_seed.std())
    stat_results[f'{label}_seeds']     = per_seed.to_dict()

# ── Medium-draft pairs (single seed) ─────────────────────────────────────────
for label, spec_df, base_df in [
    ('tr_medium', speculative_tr_med_df, baseline_tr_df),
    ('en_medium', speculative_en_med_df, baseline_en_df),
]:
    lat_b = base_df['latency_ms'].tolist()
    lat_s = spec_df['latency_ms'].tolist()
    n     = min(len(lat_b), len(lat_s))
    stat_results[f'speedup_{label}'] = compute_speedup(lat_b[:n], lat_s[:n])
    ar = spec_df['acceptance_rate'].dropna().tolist()
    lo, hi = bootstrap_ci(ar)
    stat_results[f'acceptance_rate_ci_{label}'] = {
        'mean': float(np.mean(ar)), 'ci_lower': lo, 'ci_upper': hi,
    }

# ── Llama pair (3 seeds) ──────────────────────────────────────────────────────
per_seed_llama = speculative_llama_df.groupby('seed')['acceptance_rate'].mean()
stat_results['llama_seed_mean'] = float(per_seed_llama.mean())
stat_results['llama_seed_std']  = float(per_seed_llama.std())
lat_b = baseline_llama_df['latency_ms'].tolist()
lat_s = speculative_llama_df['latency_ms'].tolist()
n = min(len(lat_b), len(lat_s))
stat_results['speedup_llama'] = compute_speedup(lat_b[:n], lat_s[:n])
ar = speculative_llama_df['acceptance_rate'].dropna().tolist()
lo, hi = bootstrap_ci(ar)
stat_results['acceptance_rate_ci_llama'] = {
    'mean': float(np.mean(ar)), 'ci_lower': lo, 'ci_upper': hi,
}

# ── Qwen pair (3 seeds) ───────────────────────────────────────────────────────
per_seed_qwen = speculative_qwen_df.groupby('seed')['acceptance_rate'].mean()
stat_results['qwen_seed_mean'] = float(per_seed_qwen.mean())
stat_results['qwen_seed_std']  = float(per_seed_qwen.std())
lat_b = baseline_qwen_df['latency_ms'].tolist()
lat_s = speculative_qwen_df['latency_ms'].tolist()
n = min(len(lat_b), len(lat_s))
stat_results['speedup_qwen'] = compute_speedup(lat_b[:n], lat_s[:n])
ar = speculative_qwen_df['acceptance_rate'].dropna().tolist()
lo, hi = bootstrap_ci(ar)
stat_results['acceptance_rate_ci_qwen'] = {
    'mean': float(np.mean(ar)), 'ci_lower': lo, 'ci_upper': hi,
}

out_path = RESULTS_DIR / 'statistical_tests.json'
save_json(stat_results, out_path)
print(f'Saved -> {out_path}')

print(f'\\nTR-small  : alpha={stat_results["tr_small_seed_mean"]:.4f} ± {stat_results["tr_small_seed_std"]:.4f}')
print(f'EN-small  : alpha={stat_results["en_small_seed_mean"]:.4f} ± {stat_results["en_small_seed_std"]:.4f}')
print(f'Llama     : alpha={stat_results["llama_seed_mean"]:.4f} ± {stat_results["llama_seed_std"]:.4f}')
print(f'Qwen      : alpha={stat_results["qwen_seed_mean"]:.4f} ± {stat_results["qwen_seed_std"]:.4f}')
print()
print(f'Speedup TR-small : {stat_results["speedup_tr"]["median_speedup"]:.3f}x')
print(f'Speedup EN-small : {stat_results["speedup_en"]["median_speedup"]:.3f}x')
print(f'Speedup Llama    : {stat_results["speedup_llama"]["median_speedup"]:.3f}x')
print(f'Speedup Qwen     : {stat_results["speedup_qwen"]["median_speedup"]:.3f}x')\
"""]

# ── Update cell-13: add llama/qwen to figures ─────────────────────────────────
cell_by_id["cell-13-figures"]["source"] = ["""\
# ── Cell 13: Generate all publication-quality figures ────────────────────────
results_for_figs = {
    "baseline":            baseline_tr_df,
    "baseline_en":         baseline_en_df,
    "speculative_tr":      speculative_tr_df,
    "speculative_tr_med":  speculative_tr_med_df,
    "speculative_en":      speculative_en_df,
    "speculative_en_med":  speculative_en_med_df,
    "speculative_llama":   speculative_llama_df,
    "speculative_qwen":    speculative_qwen_df,
    "baseline_llama":      baseline_llama_df,
    "baseline_qwen":       baseline_qwen_df,
    "ablation":            ablation_df,
    "position_acceptance": position_df,
    "oov_tr":              oov_tr_df,
    "oov_en":              oov_en_df,
    "quality":             quality_df,
}

saved = generate_all_figures(results_for_figs, FIGURES_DIR)
print(f"Generated {len(saved)} figure files:")
for p in saved:
    print(f"  {p}")\
"""]

# ── Insert new cells in the right order ──────────────────────────────────────
# Find insertion points by id
def insert_after(cells, after_id, new_cell):
    idx = next(i for i, c in enumerate(cells) if c["id"] == after_id)
    cells.insert(idx + 1, new_cell)

insert_after(cells, "cell-06d-models-en-medium", cell_06e)
insert_after(cells, "cell-06e-llama",            cell_06f)
insert_after(cells, "cell-07b-baseline-en",      cell_07c)
insert_after(cells, "cell-07c-baseline-llama",   cell_07d)
insert_after(cells, "cell-09b-spec-en-medium",   cell_09c)
insert_after(cells, "cell-09c-spec-llama",       cell_09d)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched: cell-03, cell-05, cell-11, cell-13")
print("Inserted: cell-06e, cell-06f, cell-07c, cell-07d, cell-09c, cell-09d")
