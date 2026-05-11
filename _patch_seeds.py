"""Patch the 5 notebook cells that need multi-seed support."""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = {c["id"]: c for c in nb["cells"]}

# ── Cell 3: add SEEDS to imports ─────────────────────────────────────────────
cells["cell-03-imports"]["source"] = ["""\
# ── Cell 3: Add repo to path and import everything from src/ ─────────────────
import sys
sys.path.insert(0, REPO_DIR)

from src.config import (
    SEED, SEEDS,
    DRAFT_MODEL_TR_SMALL_NAME, DRAFT_MODEL_TR_MEDIUM_NAME, TARGET_MODEL_TR_NAME,
    DRAFT_MODEL_EN_SMALL_NAME, DRAFT_MODEL_EN_MEDIUM_NAME, TARGET_MODEL_EN_NAME,
    MAX_NEW_TOKENS, DRAFT_STEPS_LIST, DEFAULT_DRAFT_STEPS,
    NUM_SAMPLES_QA, NUM_SAMPLES_SUM, NUM_SAMPLES_EN,
    QUANTIZATION_BITS, RESULTS_DIR, FIGURES_DIR,
)
from src.utils      import seed_everything, save_json, check_gpu, git_push
from src.data       import load_xquad_tr, load_trnews, load_squad_en
from src.models     import load_draft_model, load_target_model
from src.speculative import run_experiment
from src.metrics    import (
    compute_task_metrics, bootstrap_ci,
    wilcoxon_test, cohens_d, mann_whitney_test,
    compute_speedup, run_all_statistical_tests,
)
from src.linguistic import (
    ZEYREK_AVAILABLE,
    compute_rejection_by_morpheme,
    position_acceptance_analysis,
    oov_analysis,
    fragmentation_acceptance_analysis,
)
from src.figures import generate_all_figures

print('All imports successful.')
print(f'Primary seed : {SEED}')
print(f'Seeds (TR-small robustness): {SEEDS}')
print(f'zeyrek available: {ZEYREK_AVAILABLE}')

# ── Ensure output directories exist (Drive must be mounted first) ──────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results -> {RESULTS_DIR}")
print(f"Figures -> {FIGURES_DIR}")\
"""]

# ── Cell 8: TR-small speculative — loop over SEEDS ───────────────────────────
cells["cell-08-spec-tr-small"]["source"] = ["""\
# ── Cell 8: Speculative — Turkish / small draft — 3 seeds ────────────────────
# Run with each seed in SEEDS to quantify variance. Each seed produces its own
# CSV; the combined DataFrame is used for stats and linguistic analysis.
import pandas as pd
import numpy as np

seed_frames_tr = []

for s in SEEDS:
    seed_everything(s)
    _df = run_experiment(
        samples        = tr_samples_all,
        draft_model    = draft_model_tr_small,
        draft_tok      = draft_tok_tr_small,
        target_model   = target_model_tr,
        target_tok     = target_tok_tr,
        mode           = 'speculative',
        draft_steps    = DEFAULT_DRAFT_STEPS,
        max_new_tokens = MAX_NEW_TOKENS,
    )
    _df['seed'] = s
    seed_frames_tr.append(_df)
    _df.drop(columns=['token_level_log']).to_csv(
        RESULTS_DIR / f'speculative_tr_small_seed{s}.csv', index=False)
    print(f'Seed {s}: acceptance={_df.acceptance_rate.mean():.4f}  '
          f'latency={_df.latency_ms.mean():.1f} ms')

speculative_tr_df = pd.concat(seed_frames_tr, ignore_index=True)
speculative_tr_df.drop(columns=['token_level_log']).to_csv(
    RESULTS_DIR / 'speculative_tr_small_results.csv', index=False)

per_seed_ar = speculative_tr_df.groupby('seed')['acceptance_rate'].mean()
print(f'\\nAll seeds  : acceptance = {per_seed_ar.mean():.4f} ± {per_seed_ar.std():.4f}')
print(speculative_tr_df.groupby(['seed','task'])[['latency_ms','acceptance_rate']].mean().round(4))\
"""]

# ── Cell 10: ablation — reseed with SEEDS[0] before loop ─────────────────────
cells["cell-10-ablation"]["source"] = ["""\
# ── Cell 10: Ablation over γ — Turkish small draft (seed = SEEDS[0]) ─────────
# Single seed to isolate γ effect; 100-sample subset for speed.
seed_everything(SEEDS[0])

ablation_frames = []

for gamma in DRAFT_STEPS_LIST:
    _df = run_experiment(
        samples        = tr_samples_all[:100],
        draft_model    = draft_model_tr_small,
        draft_tok      = draft_tok_tr_small,
        target_model   = target_model_tr,
        target_tok     = target_tok_tr,
        mode           = 'speculative',
        draft_steps    = gamma,
        max_new_tokens = MAX_NEW_TOKENS,
    )
    ablation_frames.append(_df)

ablation_df = pd.concat(ablation_frames, ignore_index=True)

out_path = RESULTS_DIR / 'ablation_gamma.csv'
ablation_df.drop(columns=['token_level_log']).to_csv(out_path, index=False)
print(f'Saved -> {out_path}')
print(ablation_df.groupby('draft_steps')[['latency_ms', 'acceptance_rate']].mean().round(4))\
"""]

# ── Cell 11: stats — add per-seed std for TR-small ───────────────────────────
cells["cell-11-stats"]["source"] = ["""\
# ── Cell 11: Statistical tests — all four model pairs ────────────────────────
import numpy as np
from src.metrics import compute_speedup, bootstrap_ci

# Small-draft pairs: TR speculative (3-seed combined) vs TR baseline,
#                    EN speculative (single seed)     vs EN baseline.
stat_results = run_all_statistical_tests(
    baseline_df    = baseline_tr_df,
    spec_tr_df     = speculative_tr_df,
    spec_en_df     = speculative_en_df,
    baseline_en_df = baseline_en_df,
)

# Per-seed acceptance rate variance for TR-small (robustness metric)
per_seed_ar = speculative_tr_df.groupby('seed')['acceptance_rate'].mean()
stat_results['tr_small_seed_mean'] = float(per_seed_ar.mean())
stat_results['tr_small_seed_std']  = float(per_seed_ar.std())
stat_results['tr_small_seeds']     = per_seed_ar.to_dict()

# Medium-draft pairs — single seed each
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

out_path = RESULTS_DIR / 'statistical_tests.json'
save_json(stat_results, out_path)
print(f'Saved -> {out_path}')

print(f'\\nTR-small seed stability: {per_seed_ar.mean():.4f} ± {per_seed_ar.std():.4f}')
print(per_seed_ar.round(4))
print()
for k, v in stat_results.items():
    if not k.startswith('tr_small_seed'):
        print(f'  {k}: {v}')\
"""]

# ── Cell 12: linguistic — all-seed logs for richer morpheme coverage ──────────
cells["cell-12-linguistic"]["source"] = ["""\
# ── Cell 12: Linguistic / morphological analysis ──────────────────────────────
# Use token-level logs from all 3 TR-small seeds for richer morpheme coverage.
tr_logs = speculative_tr_df["token_level_log"].tolist()

morpheme_df = compute_rejection_by_morpheme(tr_logs)
position_df = position_acceptance_analysis(tr_logs)
oov_tr_df   = oov_analysis(tr_samples_all, draft_tok_tr_small)
oov_en_df   = oov_analysis(squad_samples,  draft_tok_en_small)
frag_acc_df = fragmentation_acceptance_analysis(tr_logs)

morpheme_df.to_csv(RESULTS_DIR / "morpheme_rejection.csv",       index=False)
position_df.to_csv(RESULTS_DIR / "position_acceptance.csv",      index=False)
oov_tr_df.to_csv(  RESULTS_DIR / "oov_analysis_tr.csv",          index=False)
oov_en_df.to_csv(  RESULTS_DIR / "oov_analysis_en.csv",          index=False)
frag_acc_df.to_csv(RESULTS_DIR / "fragmentation_acceptance.csv",  index=False)

print("Morpheme rejection rates:")
print(morpheme_df.to_string(index=False))

print("\\nPosition acceptance rates:")
print(position_df.to_string(index=False))

print("\\nFragmentation stats (Turkish tokenizer):")
print(oov_tr_df.groupby("task")["fragments"].describe().round(3))

print("\\nFragmentation stats (English tokenizer):")
print(oov_en_df.groupby("task")["fragments"].describe().round(3))

if "spearman_corr" in frag_acc_df.attrs:
    r = frag_acc_df.attrs["spearman_corr"]
    p = frag_acc_df.attrs["spearman_p"]
    print(f"\\nFragmentation vs acceptance: Spearman r={r:.4f} (p={p:.4f})")\
"""]

# ── Write back ────────────────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched 5 cells: cell-03, cell-08, cell-10, cell-11, cell-12")
