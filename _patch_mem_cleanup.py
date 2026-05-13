"""
Patch cell-06e and cell-06f to free GPU memory before loading large models.
GPT-2 models must be deleted and cache cleared before Llama/Qwen load.
"""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = {c["id"]: c for c in nb["cells"]}

# ── Cell 6e: free all GPT-2 models before loading Llama ──────────────────────
cells["cell-06e-llama"]["source"] = ["""\
# ── Cell 6e: Load Llama-3 models (free GPT-2 memory first) ───────────────────
import gc, torch

# Delete all GPT-2 scale models from GPU before loading 8B target
_to_delete = [
    'draft_model_tr_small', 'draft_tok_tr_small',
    'draft_model_tr_medium', 'draft_tok_tr_medium',
    'target_model_tr', 'target_tok_tr',
    'draft_model_en_small', 'draft_tok_en_small',
    'draft_model_en_medium', 'draft_tok_en_medium',
    'target_model_en', 'target_tok_en',
]
for _var in _to_delete:
    if _var in dir():
        del globals()[_var]
gc.collect()
torch.cuda.empty_cache()
print(f"GPU free after cleanup: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# Draft: Llama-3.2-1B — try unsloth mirror first (no gate), then original
LLAMA_DRAFT_CANDIDATES = [
    "unsloth/Llama-3.2-1B",    # gatesiz mirror, her zaman çalışır
    "meta-llama/Llama-3.2-1B", # orijinal — lisans onayı sonrası
]
draft_model_llama, draft_tok_llama = None, None
for candidate in LLAMA_DRAFT_CANDIDATES:
    try:
        print(f"Trying draft: {candidate} ...")
        draft_model_llama, draft_tok_llama = load_draft_model(candidate, device="cuda:0")
        print(f"  Loaded: {candidate}  |  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
        break
    except Exception as e:
        print(f"  Failed ({type(e).__name__}): {str(e)[:100]}")

if draft_model_llama is None:
    raise RuntimeError(
        "Llama draft yüklenemedi. "
        "https://huggingface.co/meta-llama/Llama-3.2-1B adresinde lisansı kabul et."
    )

# Target: Turkish-Llama-8b-Instruct — 4-bit NF4 (~4-5 GB)
print("Loading Llama target (8B, 4-bit NF4) ...")
target_model_llama, target_tok_llama = load_target_model(
    TARGET_MODEL_LLAMA_NAME, bits=QUANTIZATION_BITS_LLAMA
)
print(f"Llama models ready.  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
check_gpu()\
"""]

# ── Cell 6f: free Llama models before loading Qwen ───────────────────────────
cells["cell-06f-qwen"]["source"] = ["""\
# ── Cell 6f: Load Qwen2.5 models (free Llama memory first) ───────────────────
import gc, torch

# Delete Llama models before loading Qwen (avoid double-loading 8B+ models)
_to_delete = ['draft_model_llama', 'draft_tok_llama',
              'target_model_llama', 'target_tok_llama']
for _var in _to_delete:
    if _var in globals():
        del globals()[_var]
gc.collect()
torch.cuda.empty_cache()
print(f"GPU free after Llama cleanup: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# Draft: tr-Qwen2.5-0.5B-SFT-v1 (~0.5B, float16, ~1 GB)
print("Loading Qwen draft (0.5B, float16) ...")
draft_model_qwen, draft_tok_qwen = load_draft_model(DRAFT_MODEL_QWEN_NAME, device="cuda:0")
print(f"  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# Target: Qwen2.5-7B-Instruct — 4-bit NF4 (~3.5 GB)
print("Loading Qwen target (7B, 4-bit NF4) ...")
target_model_qwen, target_tok_qwen = load_target_model(
    TARGET_MODEL_QWEN_NAME, bits=QUANTIZATION_BITS_QWEN
)
print(f"Qwen models ready.  GPU free: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
check_gpu()\
"""]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched: cell-06e-llama, cell-06f-qwen (memory cleanup added)")
