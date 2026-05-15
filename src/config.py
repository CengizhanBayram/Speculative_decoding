from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
# SEED  : primary seed used for dataset shuffling, greedy baselines, and all
#         experiments except the multi-seed TR-small robustness runs.
# SEEDS : three seeds used to assess variance in the primary TR-small experiment.
#         Other model pairs (TR-medium, EN-small, EN-medium) run with SEEDS[0].
SEED  = 42
SEEDS = [42, 123, 456]

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

# ── Llama-3 family ────────────────────────────────────────────────────────────
# All Llama-3 models share the same tokenizer (128,256 tokens, tiktoken-based).
# Turkish-Llama-8b is fine-tuned from Llama-3.1-8B → tokenizer is identical.
# Draft runs in float16 (1B ≈ 2 GB); target needs 4-bit NF4 (8B ≈ 4 GB on T4).
DRAFT_MODEL_LLAMA_NAME  = "unsloth/Llama-3.2-1B-Instruct"  # gatesiz mirror; instruct — matches target fine-tuning type
TARGET_MODEL_LLAMA_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
QUANTIZATION_BITS_LLAMA = 4   # 8B target: 4-bit required on 16 GB T4

# ── Qwen2.5 family ────────────────────────────────────────────────────────────
# Qwen2.5 (0.5B → 72B) shares a single tokenizer (151,936 tokens).
# Draft: Turkish-SFT Qwen2.5-0.5B; Target: Qwen2.5-7B-Instruct (multilingual).
# Cross-lingual pair: measures whether TR-adapted draft generalises to a
# multilingual target that was never specifically Turkish-trained.
DRAFT_MODEL_QWEN_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"      # instruct — matches target type; same 151936-token vocab
TARGET_MODEL_QWEN_NAME  = "Qwen/Qwen2.5-7B-Instruct"
QUANTIZATION_BITS_QWEN  = 4   # 7B target: 4-bit fits on T4 (~3.5 GB)

# ── Backwards-compatible aliases ──────────────────────────────────────────────
DRAFT_MODEL_NAME    = DRAFT_MODEL_TR_SMALL_NAME
TARGET_MODEL_NAME   = TARGET_MODEL_TR_NAME
DRAFT_MODEL_EN_NAME = DRAFT_MODEL_EN_SMALL_NAME

# ── Generation hyperparameters ────────────────────────────────────────────────
MAX_NEW_TOKENS      = 128
DRAFT_STEPS_LIST    = [1, 3, 5, 7, 10]   # γ values for ablation study
DEFAULT_DRAFT_STEPS = 5                   # γ used in main experiments

# ── Dataset sizes ─────────────────────────────────────────────────────────────
NUM_SAMPLES_QA    = 500   # XQuAD-TR (Turkish QA)
NUM_SAMPLES_SUM   = 500   # TR-News  (Turkish summarisation)
NUM_SAMPLES_EN    = 500   # SQuAD    (English QA)
NUM_SAMPLES_LLAMA = 300   # fewer samples — 8B model is slower per sample
NUM_SAMPLES_QWEN  = 300   # fewer samples — 7B model is slower per sample

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
