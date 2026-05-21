from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
# SEED     : primary seed for greedy baselines and single-seed experiments.
# SEEDS    : three seeds for Turkish small-draft robustness runs.
# SEEDS_EN : three seeds for the symmetric English experiment (QA + summarisation,
#            3 × 1,000 samples) — matches the Turkish experimental design.
SEED     = 42
SEEDS    = [42, 123, 456]
SEEDS_EN = [42, 123, 456]

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

# ── Llama-3 Turkish pair (cross-lingual — supplementary) ─────────────────────
# Draft: English-base Llama-3.2-1B-Instruct (128,256-token tiktoken vocab).
# Target: Turkish-Llama-8B fine-tuned from Llama-3.1-8B; inherits same tokenizer.
# Cross-lingual: English-base draft on Turkish text — quantifies the vocabulary
# mismatch penalty relative to the same-corpus ytu-ce-cosmos pair.
DRAFT_MODEL_LLAMA_TR_NAME  = "unsloth/Llama-3.2-1B-Instruct"
TARGET_MODEL_LLAMA_TR_NAME = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
QUANTIZATION_BITS_LLAMA    = 4   # 8B target: 4-bit required on 16 GB T4

# Backwards-compatible alias used by existing notebook cells
DRAFT_MODEL_LLAMA_NAME  = DRAFT_MODEL_LLAMA_TR_NAME
TARGET_MODEL_LLAMA_NAME = TARGET_MODEL_LLAMA_TR_NAME

# ── Llama-3 English pair (native — new symmetric comparison) ──────────────────
# Both models are English-native; same tokenizer; both instruction-tuned.
# Provides a modern 7B-scale comparison point at the same size ratio (1:8) as
# the Turkish Llama pair, isolating language effects from architecture effects.
DRAFT_MODEL_LLAMA_EN_NAME  = "unsloth/Llama-3.2-1B-Instruct"
TARGET_MODEL_LLAMA_EN_NAME = "meta-llama/Llama-3.1-8B-Instruct"
QUANTIZATION_BITS_LLAMA_EN = 4   # 8B target: 4-bit on T4

# ── Qwen2.5 family ────────────────────────────────────────────────────────────
# Qwen2.5 (0.5B → 72B) shares a single tokenizer (151,936 tokens).
# Both models are multilingual instruct — clean same-type pair for scale comparison.
DRAFT_MODEL_QWEN_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_QWEN_NAME  = "Qwen/Qwen2.5-7B-Instruct"
QUANTIZATION_BITS_QWEN  = 4   # 7B target: 4-bit fits on T4 (~3.5 GB)

# ── Pythia (EleutherAI) — same-corpus English pair ───────────────────────────
# All Pythia checkpoints are co-trained from scratch on the Pile dataset
# (300B tokens, identical training data and order across all sizes).
# Same NeoX BPE tokenizer (50,304 tokens).
# This is the direct English analog of the ytu-cosmos same-corpus pair:
#   ytu-cosmos: TR same-corpus, 117M → 774M, ratio 1:6.6
#   pythia:     EN same-corpus, 160M → 1B,   ratio 1:6.25
# Used to break the TR/EN same-corpus / cross-run confound called out
# by reviewers.
DRAFT_MODEL_PYTHIA_NAME   = "EleutherAI/pythia-160m"
TARGET_MODEL_PYTHIA_NAME  = "EleutherAI/pythia-1b"
QUANTIZATION_BITS_PYTHIA  = 0   # 1B in float16 fits comfortably on L4 (~2GB)

# ── Backwards-compatible aliases ──────────────────────────────────────────────
DRAFT_MODEL_NAME    = DRAFT_MODEL_TR_SMALL_NAME
TARGET_MODEL_NAME   = TARGET_MODEL_TR_NAME
DRAFT_MODEL_EN_NAME = DRAFT_MODEL_EN_SMALL_NAME

# ── Generation hyperparameters ────────────────────────────────────────────────
MAX_NEW_TOKENS      = 128
DRAFT_STEPS_LIST    = [1, 3, 5, 7, 10]      # γ values for ablation study
DEFAULT_DRAFT_STEPS = 5                      # γ used in main experiments

# Temperature settings.
# TEMPERATURE       : default for all main experiments (greedy/deterministic).
#                     At T=0.0 accept/reject is fully deterministic — no stochastic draw.
# TEMPERATURES      : sweep for the T>0 sensitivity study.
#                     At T>0 the stochastic accept/reject criterion applies;
#                     results will vary across seeds.
TEMPERATURE   = 0.0
TEMPERATURES  = [0.0, 0.3, 0.7, 1.0]

# ── Dataset sizes ─────────────────────────────────────────────────────────────
# Turkish: 500 QA + 500 SUM per seed × 3 seeds = 3,000 samples total
# English (symmetric): 500 QA + 500 SUM per seed × 3 seeds = 3,000 samples total
NUM_SAMPLES_QA     = 500   # XQuAD-TR (Turkish QA, per seed)
NUM_SAMPLES_SUM    = 500   # TR-News  (Turkish summarisation, per seed)
NUM_SAMPLES_EN_QA  = 500   # SQuAD    (English QA, per seed)
NUM_SAMPLES_EN_SUM = 500   # CNN/DailyMail (English summarisation, per seed)
NUM_SAMPLES_LLAMA  = 300   # fewer samples — 8B model is slower per sample
NUM_SAMPLES_QWEN   = 300   # fewer samples — 7B model is slower per sample

# Backwards-compatible alias (existing cells use NUM_SAMPLES_EN)
NUM_SAMPLES_EN = NUM_SAMPLES_EN_QA

# 0 = float16, no quantization (suitable for GPT-2-scale models ≤ 1 B params)
# Set to 4 or 8 for larger models to enable BitsAndBytes quantization.
QUANTIZATION_BITS = 0   # 0 = float16, no quantization (GPT-2-scale models ≤ 1B)

# ── Output directories (Google Drive) ────────────────────────────────────────
DRIVE_BASE  = Path("/content/drive/MyDrive/speculative decoding sonuçları")
RESULTS_DIR = DRIVE_BASE / "results"
FIGURES_DIR = DRIVE_BASE / "figures"

try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Drive not mounted (local run); directories are created explicitly in Cell 3
