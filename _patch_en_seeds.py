"""Patch cell-09 and cell-11 to run EN-small over 3 seeds (mirrors TR-small cell-08)."""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = {c["id"]: c for c in nb["cells"]}

# ── Cell 9: EN-small — loop over SEEDS ───────────────────────────────────────
cells["cell-09-spec-en-small"]["source"] = ["""\
# ── Cell 9: Speculative — English / small draft — 3 seeds ────────────────────
# Mirrors cell-08: identical seed list and per-seed CSV naming for symmetry.
seed_frames_en = []
for s in SEEDS:
    seed_everything(s)
    _df = run_experiment(
        samples        = squad_samples,
        draft_model    = draft_model_en_small,
        draft_tok      = draft_tok_en_small,
        target_model   = target_model_en,
        target_tok     = target_tok_en,
        mode           = 'speculative',
        draft_steps    = DEFAULT_DRAFT_STEPS,
        max_new_tokens = MAX_NEW_TOKENS,
    )
    _df['seed'] = s
    seed_frames_en.append(_df)
    _df.drop(columns=['token_level_log']).to_csv(
        RESULTS_DIR / f'speculative_en_small_seed{s}.csv', index=False
    )
    ar = _df['acceptance_rate'].mean()
    sp = (_df['latency_ms'].mean() / baseline_en_df['latency_ms'].mean()
          if 'latency_ms' in baseline_en_df.columns else float('nan'))
    print(f'  seed={s}  α={ar:.4f}  speedup={sp:.4f}')

speculative_en_df = pd.concat(seed_frames_en, ignore_index=True)
out_path = RESULTS_DIR / 'speculative_en_small_results.csv'
speculative_en_df.drop(columns=['token_level_log']).to_csv(out_path, index=False)
print(f'Saved combined -> {out_path}')
print(speculative_en_df.groupby('task')[['latency_ms', 'acceptance_rate']].mean().round(4))\
"""]

# ── Cell 11: add EN-small seed stability alongside TR-small ──────────────────
cells["cell-11-stats"]["source"] = ["""\
# ── Cell 11: Statistical tests — all four model pairs ────────────────────────
import numpy as np
from src.metrics import compute_speedup, bootstrap_ci

stat_results = run_all_statistical_tests(
    baseline_df    = baseline_tr_df,
    spec_tr_df     = speculative_tr_df,
    spec_en_df     = speculative_en_df,
    baseline_en_df = baseline_en_df,
)

# TR-small seed stability
per_seed_ar_tr = speculative_tr_df.groupby('seed')['acceptance_rate'].mean()
stat_results['tr_small_seed_mean'] = float(per_seed_ar_tr.mean())
stat_results['tr_small_seed_std']  = float(per_seed_ar_tr.std())
stat_results['tr_small_seeds']     = per_seed_ar_tr.to_dict()

# EN-small seed stability (now symmetric with TR-small)
per_seed_ar_en = speculative_en_df.groupby('seed')['acceptance_rate'].mean()
stat_results['en_small_seed_mean'] = float(per_seed_ar_en.mean())
stat_results['en_small_seed_std']  = float(per_seed_ar_en.std())
stat_results['en_small_seeds']     = per_seed_ar_en.to_dict()

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

print(f'\\nTR-small seed stability: {per_seed_ar_tr.mean():.4f} \\u00b1 {per_seed_ar_tr.std():.4f}')
print(per_seed_ar_tr.round(4))
print(f'\\nEN-small seed stability: {per_seed_ar_en.mean():.4f} \\u00b1 {per_seed_ar_en.std():.4f}')
print(per_seed_ar_en.round(4))
print()
for k, v in stat_results.items():
    if not k.endswith('_seeds'):
        print(f'  {k}: {v}')\
"""]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched: cell-09-spec-en-small, cell-11-stats")
