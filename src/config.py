from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
# All random operations (dataset shuffle, numpy, torch, Python random) are seeded
# with this value via utils.seed_everything(SEED) at the start of each run.
SEED = 42

# ── Turkish model pairs ───────────────────────────────────────────────────────
# All ytu-ce-cosmos models are trained on Turkish and share the same GPT-2 BPE
# tokenizer (50,257 tokens). Shared vocabulary is required for speculative decoding.
DRAFT_MODEL_TR_SMALL_NAME  = "ytu-ce-cosmos/turkish-gpt2"         # ~117 M (draft — small)
DRAFT_MODEL_TR_MEDIUM_NAME = "ytu-ce-cosmos/turkish-gpt2-medium"  # ~354 M (draft — medium)
TARGET_MODEL_TR_NAME       = "ytu-ce-cosmos/turkish-gpt2-large"   # ~774 M (target — shared)

# ── English model pairs ───────────────────────────────────────────────────────
# Standard GPT-2 family — all sizes share the same BPE tokenizer.
# Target size (774 M) matches Turkish target exactly for a fair cross-lingual
# comparison where any acceptance-rate difference is due to language, not capacity.
DRAFT_MODEL_EN_SMALL_NAME  = "gpt2"         # ~117 M (draft — small)
DRAFT_MODEL_EN_MEDIUM_NAME = "gpt2-medium"  # ~354 M (draft — medium)
TARGET_MODEL_EN_NAME       = "gpt2-large"   # ~774 M (target — shared)

# ── Backwards-compatible aliases ──────────────────────────────────────────────
DRAFT_MODEL_NAME    = DRAFT_MODEL_TR_SMALL_NAME
TARGET_MODEL_NAME   = TARGET_MODEL_TR_NAME
DRAFT_MODEL_EN_NAME = DRAFT_MODEL_EN_SMALL_NAME

# ── Generation hyperparameters ────────────────────────────────────────────────
MAX_NEW_TOKENS      = 128
DRAFT_STEPS_LIST    = [1, 3, 5, 7, 10]   # γ values for ablation study
DEFAULT_DRAFT_STEPS = 5                   # γ used in main experiments

# ── Dataset sizes ─────────────────────────────────────────────────────────────
NUM_SAMPLES_QA  = 500   # XQuAD-TR (Turkish QA)
NUM_SAMPLES_SUM = 500   # TR-News  (Turkish summarisation)
NUM_SAMPLES_EN  = 500   # SQuAD    (English QA)

# 0 = float16, no quantization (suitable for GPT-2-scale models ≤ 1 B params)
# Set to 4 or 8 for larger models to enable BitsAndBytes quantization.
QUANTIZATION_BITS = 0

# ── Output directories (Google Drive) ────────────────────────────────────────
DRIVE_BASE  = Path("/content/drive/MyDrive/speculative decoding sonuçları")
RESULTS_DIR = DRIVE_BASE / "results"
FIGURES_DIR = DRIVE_BASE / "figures"

try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Drive not mounted (local run); directories are created explicitly in Cell 3
