"""Remove zeyrek references from notebook cells 3 and 12."""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = {c["id"]: c for c in nb["cells"]}

# ── Cell 3: remove ZEYREK_AVAILABLE import ───────────────────────────────────
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
    position_acceptance_analysis,
    oov_analysis,
    fragmentation_acceptance_analysis,
)
from src.figures import generate_all_figures

print('All imports successful.')
print(f'Primary seed            : {SEED}')
print(f'Seeds (TR-small robust) : {SEEDS}')

# ── Ensure output directories exist (Drive must be mounted first) ──────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results -> {RESULTS_DIR}")
print(f"Figures -> {FIGURES_DIR}")\
"""]

# ── Cell 12: remove morpheme analysis, keep position + fragmentation ─────────
cells["cell-12-linguistic"]["source"] = ["""\
# ── Cell 12: Linguistic analysis — position + fragmentation ──────────────────
# Token-level logs from all 3 TR-small seeds for maximum statistical power.
tr_logs = speculative_tr_df["token_level_log"].tolist()

position_df = position_acceptance_analysis(tr_logs)
oov_tr_df   = oov_analysis(tr_samples_all, draft_tok_tr_small)
oov_en_df   = oov_analysis(squad_samples,  draft_tok_en_small)
frag_acc_df = fragmentation_acceptance_analysis(tr_logs)

position_df.to_csv(RESULTS_DIR / "position_acceptance.csv",      index=False)
oov_tr_df.to_csv(  RESULTS_DIR / "oov_analysis_tr.csv",          index=False)
oov_en_df.to_csv(  RESULTS_DIR / "oov_analysis_en.csv",          index=False)
frag_acc_df.to_csv(RESULTS_DIR / "fragmentation_acceptance.csv",  index=False)

print("Position acceptance rates:")
print(position_df.to_string(index=False))

print("\\nFragmentation stats (Turkish tokenizer):")
print(oov_tr_df.groupby("task")["fragments"].describe().round(3))

print("\\nFragmentation stats (English tokenizer):")
print(oov_en_df.groupby("task")["fragments"].describe().round(3))

tr_mean = oov_tr_df.fragments.mean()
en_mean = oov_en_df.fragments.mean()
print(f"\\nTR/EN fragmentation ratio : {tr_mean/en_mean:.3f}x")
print(f"TR complex (>1 subword)   : {(oov_tr_df.fragments>1).mean()*100:.1f}%")
print(f"EN complex (>1 subword)   : {(oov_en_df.fragments>1).mean()*100:.1f}%")

if "spearman_corr" in frag_acc_df.attrs:
    r = frag_acc_df.attrs["spearman_corr"]
    p = frag_acc_df.attrs["spearman_p"]
    print(f"\\nFragmentation vs acceptance: Spearman r={r:.4f} (p={p:.4f})")\
"""]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched cells: cell-03-imports, cell-12-linguistic")
