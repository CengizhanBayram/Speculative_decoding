from pathlib import Path

SEED = 42

# Turkish draft (small, fast) + target (large, quantized)
DRAFT_MODEL_NAME  = "ytu-ce-cosmos/turkish-gpt2-large"
TARGET_MODEL_NAME = "ytu-ce-cosmos/turkish-llama-2-7b-chat"

# English draft + target (same target model, different draft)
DRAFT_MODEL_EN_NAME  = "gpt2"
TARGET_MODEL_EN_NAME = "meta-llama/Llama-2-7b-chat-hf"

MAX_NEW_TOKENS    = 128
DRAFT_STEPS_LIST  = [1, 3, 5, 7, 10]
DEFAULT_DRAFT_STEPS = 5

NUM_SAMPLES_QA  = 500
NUM_SAMPLES_SUM = 500
NUM_SAMPLES_EN  = 500

QUANTIZATION_BITS = 4

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
