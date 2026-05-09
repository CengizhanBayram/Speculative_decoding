from pathlib import Path

SEED = 42

# Turkish pair — both use the same GPT-2 BPE tokenizer (50,257 tokens).
# Speculative decoding requires a shared vocabulary; these two models satisfy that.
DRAFT_MODEL_NAME  = "ytu-ce-cosmos/turkish-gpt2"        # ~117 M params (draft)
TARGET_MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2-large"  # ~774 M params (target)

# English pair — standard GPT-2 family, same BPE tokenizer.
DRAFT_MODEL_EN_NAME  = "gpt2"     # ~117 M params (draft)
TARGET_MODEL_EN_NAME = "gpt2-xl"  # ~1.5 B params (target)

MAX_NEW_TOKENS      = 128
DRAFT_STEPS_LIST    = [1, 3, 5, 7, 10]
DEFAULT_DRAFT_STEPS = 5

NUM_SAMPLES_QA  = 500
NUM_SAMPLES_SUM = 500
NUM_SAMPLES_EN  = 500

# 0 = float16, no quantization (suitable for GPT-2 scale models on Colab)
QUANTIZATION_BITS = 0

# Google Drive output directories
DRIVE_BASE  = Path("/content/drive/MyDrive/speculative decoding sonuçları")
RESULTS_DIR = DRIVE_BASE / "results"
FIGURES_DIR = DRIVE_BASE / "figures"

try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Drive not mounted (local run); caller is responsible for mkdir
