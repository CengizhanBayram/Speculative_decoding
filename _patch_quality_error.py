"""
Patch run_experiments.ipynb:
  - Update cell-03-imports: add rejected_token_analysis to linguistic imports
  - Add cell-11b after cell-11: compute BLEU/ROUGE quality metrics
  - Update cell-12: add rejected token analysis
  - Update cell-13: pass oov_tr, oov_en, quality to generate_all_figures
"""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells      = nb["cells"]
cell_by_id = {c["id"]: c for c in cells}

# ── Cell 3: add rejected_token_analysis import ────────────────────────────────
cell_by_id["cell-03-imports"]["source"] = ["""\
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
    position_acceptance_analysis,
    oov_analysis,
    fragmentation_acceptance_analysis,
    rejected_token_analysis,
)
from src.figures import generate_all_figures

print('All imports successful.')
print(f'Primary seed            : {SEED}')
print(f'Seeds (TR/EN-small robust) : {SEEDS}')

# ── Ensure output directories exist (Drive must be mounted first) ──────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results -> {RESULTS_DIR}")
print(f"Figures -> {FIGURES_DIR}")\
"""]

# ── Cell 11b: Quality metrics (BLEU / ROUGE) ──────────────────────────────────
cell_11b = {
    "id":        "cell-11b-quality",
    "cell_type": "code",
    "metadata":  {},
    "outputs":   [],
    "execution_count": None,
    "source": ["""\
# ── Cell 11b: Output quality metrics (ROUGE-1/2/L + BLEU) ────────────────────
# Speculative decoding is lossless in theory (same distribution as target greedy).
# This cell quantifies how closely speculative outputs match the reference answers
# compared to greedy baseline — both should be near-identical.
import numpy as np
from src.metrics import compute_task_metrics

def compute_quality_summary(df, label):
    rows = []
    for _, row in df.iterrows():
        if not row['reference'] or not row['generated_text']:
            continue
        m = compute_task_metrics(row['generated_text'], row['reference'], row['task'])
        for metric, value in m.items():
            rows.append({'condition': label, 'metric': metric, 'value': value})
    return rows

quality_rows = []

# Turkish: greedy baseline vs speculative (use seed-42 slice for speed)
tr_base_sample = baseline_tr_df.head(200)
tr_spec_sample = speculative_tr_df[speculative_tr_df['seed'] == 42].head(200)
quality_rows += compute_quality_summary(tr_base_sample, 'TR-Greedy')
quality_rows += compute_quality_summary(tr_spec_sample, 'TR-Spec')

# English: greedy baseline vs speculative (use seed-42 slice)
en_base_sample = baseline_en_df.head(200)
en_spec_sample = speculative_en_df[speculative_en_df['seed'] == 42].head(200)
quality_rows += compute_quality_summary(en_base_sample, 'EN-Greedy')
quality_rows += compute_quality_summary(en_spec_sample, 'EN-Spec')

import pandas as pd
quality_df = pd.DataFrame(quality_rows)
quality_df.to_csv(RESULTS_DIR / 'quality_metrics.csv', index=False)
print("Quality metrics summary (mean per condition/metric):")
print(quality_df.groupby(['condition', 'metric'])['value'].mean().round(4).unstack())
print(f"\\nSaved -> {RESULTS_DIR / 'quality_metrics.csv'}")\
"""],
}

# ── Cell 12: add rejected token analysis ─────────────────────────────────────
cell_by_id["cell-12-linguistic"]["source"] = ["""\
# ── Cell 12: Linguistic analysis — position + fragmentation + error analysis ──
# Token-level logs from all TR-small seeds and EN-small seeds.
tr_logs = speculative_tr_df["token_level_log"].tolist()
en_logs = speculative_en_df["token_level_log"].tolist()

position_df = position_acceptance_analysis(tr_logs)
oov_tr_df   = oov_analysis(tr_samples_all, draft_tok_tr_small)
oov_en_df   = oov_analysis(squad_samples,  draft_tok_en_small)
frag_acc_df = fragmentation_acceptance_analysis(tr_logs)

# Rejected token frequency analysis — top 30 by proposal count
rejected_tr_df = rejected_token_analysis(tr_logs, top_n=30)
rejected_en_df = rejected_token_analysis(en_logs, top_n=30)

position_df.to_csv(   RESULTS_DIR / "position_acceptance.csv",      index=False)
oov_tr_df.to_csv(     RESULTS_DIR / "oov_analysis_tr.csv",          index=False)
oov_en_df.to_csv(     RESULTS_DIR / "oov_analysis_en.csv",          index=False)
frag_acc_df.to_csv(   RESULTS_DIR / "fragmentation_acceptance.csv",  index=False)
rejected_tr_df.to_csv(RESULTS_DIR / "rejected_tokens_tr.csv",       index=False)
rejected_en_df.to_csv(RESULTS_DIR / "rejected_tokens_en.csv",       index=False)

print("Position acceptance rates:")
print(position_df.to_string(index=False))

print("\\nFragmentation stats (Turkish):")
print(oov_tr_df.groupby("task")["fragments"].describe().round(3))
print("\\nFragmentation stats (English):")
print(oov_en_df.groupby("task")["fragments"].describe().round(3))

tr_mean = oov_tr_df.fragments.mean()
en_mean = oov_en_df.fragments.mean()
print(f"\\nTR/EN fragmentation ratio : {tr_mean/en_mean:.3f}x")
print(f"TR complex (>1 subword)   : {(oov_tr_df.fragments>1).mean()*100:.1f}%")
print(f"EN complex (>1 subword)   : {(oov_en_df.fragments>1).mean()*100:.1f}%")

if "spearman_corr" in frag_acc_df.attrs:
    r = frag_acc_df.attrs["spearman_corr"]
    p = frag_acc_df.attrs["spearman_p"]
    print(f"\\nFragmentation vs acceptance: Spearman r={r:.4f} (p={p:.4f})")

print("\\nTop-10 most-proposed tokens (Turkish):")
print(rejected_tr_df.head(10).to_string(index=False))
print("\\nTop-10 most-rejected tokens by rejection_rate (Turkish, min 50 proposals):")
high_vol_tr = rejected_tr_df[rejected_tr_df['total'] >= 50]
print(high_vol_tr.nlargest(10, 'rejection_rate')[['token_str','total','rejected','rejection_rate']].to_string(index=False))\
"""]

# ── Cell 13: pass oov_tr, oov_en, quality to generate_all_figures ─────────────
cell_by_id["cell-13-figures"]["source"] = ["""\
# ── Cell 13: Generate all publication-quality figures ────────────────────────
results_for_figs = {
    "baseline":           baseline_tr_df,
    "baseline_en":        baseline_en_df,
    "speculative_tr":     speculative_tr_df,
    "speculative_tr_med": speculative_tr_med_df,
    "speculative_en":     speculative_en_df,
    "speculative_en_med": speculative_en_med_df,
    "ablation":           ablation_df,
    "position_acceptance": position_df,
    "oov_tr":             oov_tr_df,
    "oov_en":             oov_en_df,
    "quality":            quality_df,
}

saved = generate_all_figures(results_for_figs, FIGURES_DIR)
print(f"Generated {len(saved)} figure files:")
for p in saved:
    print(f"  {p}")\
"""]

# ── Insert cell-11b after cell-11-stats ───────────────────────────────────────
idx_11 = next(i for i, c in enumerate(cells) if c["id"] == "cell-11-stats")
cells.insert(idx_11 + 1, cell_11b)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched cells: cell-03-imports, cell-12-linguistic, cell-13-figures")
print("Inserted:      cell-11b-quality")
