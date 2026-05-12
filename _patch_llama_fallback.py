"""Patch cell-06e to use fallback list for Llama draft (gated model workaround)."""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = {c["id"]: c for c in nb["cells"]}

cells["cell-06e-llama"]["source"] = ["""\
# ── Cell 6e: Load Llama-3 models ──────────────────────────────────────────────
# Draft : Llama-3.2-1B  (~1B, float16, ~2 GB VRAM)
# Target: Turkish-Llama-8b-Instruct  (~8B, 4-bit NF4, ~4 GB VRAM)
# Shared tokenizer: Llama-3 (128,256 tokens)
#
# NOT: meta-llama/Llama-3.2-1B lisans onayı gerektiriyor.
# Onay için: https://huggingface.co/meta-llama/Llama-3.2-1B → "Agree and access"
# Onay verilene kadar unsloth mirror kullanılır (aynı ağırlıklar, gatesiz).

LLAMA_DRAFT_CANDIDATES = [
    "unsloth/Llama-3.2-1B",          # gatesiz mirror, her zaman çalışır
    "meta-llama/Llama-3.2-1B",       # orijinal — onay verildikten sonra
]

draft_model_llama, draft_tok_llama = None, None
for candidate in LLAMA_DRAFT_CANDIDATES:
    try:
        print(f"Trying Llama draft: {candidate} ...")
        draft_model_llama, draft_tok_llama = load_draft_model(candidate, device="cuda:0")
        print(f"  Loaded from: {candidate}")
        DRAFT_MODEL_LLAMA_NAME = candidate   # override for logging
        break
    except Exception as e:
        print(f"  Failed ({type(e).__name__}): {str(e)[:80]}")

if draft_model_llama is None:
    raise RuntimeError(
        "Llama draft model yüklenemedi.\\n"
        "Çözüm: https://huggingface.co/meta-llama/Llama-3.2-1B adresinde lisansı kabul et."
    )

print("\\nLoading Llama target (8B, 4-bit)...")
target_model_llama, target_tok_llama = load_target_model(
    TARGET_MODEL_LLAMA_NAME, bits=QUANTIZATION_BITS_LLAMA
)
print("Llama models ready.")
check_gpu()\
"""]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patched: cell-06e-llama (fallback list)")
