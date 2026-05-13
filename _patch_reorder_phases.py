"""
Reorder notebook phases so model loading/cleanup happens at the right time:

Phase 1 — GPT-2:
  06, 06b, 06c, 06d  (load)
  07, 07b            (baseline)
  08, 08b, 09, 09b   (speculative)

Phase 2 — Llama  (cells 06e, 07c, 09c):
  06e after 09b  (cleanup GPT-2 + load Llama)
  07c after 06e  (Llama baseline)
  09c after 07c  (Llama speculative)

Phase 3 — Qwen  (cells 06f, 07d, 09d):
  06f after 09c  (cleanup Llama + load Qwen)
  07d after 06f  (Qwen baseline)
  09d after 07d  (Qwen speculative)

Phase 4 — Analysis:
  10, 11, 11b, 12, 13, 14
"""
import json, pathlib

NB = pathlib.Path(r"c:\Users\cengh\Desktop\Speculative_decoding\run_experiments.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Desired order of cell IDs
DESIRED_ORDER = [
    "cell-md-title",
    "cell-01-auth",
    "cell-02-clone",
    "cell-03-imports",
    "cell-04-seed-gpu",
    "cell-05-data",
    # ── Phase 1: GPT-2 ──────────────────────────────────────────────────
    "cell-06-models-tr-small",
    "cell-06b-models-tr-medium",
    "cell-06c-models-en-small",
    "cell-06d-models-en-medium",
    "cell-07-baseline-tr",
    "cell-07b-baseline-en",
    "cell-08-spec-tr-small",
    "cell-08b-spec-tr-medium",
    "cell-09-spec-en-small",
    "cell-09b-spec-en-medium",
    # ── Phase 2: Llama ──────────────────────────────────────────────────
    "cell-06e-llama",       # cleanup GPT-2 + load Llama
    "cell-07c-baseline-llama",
    "cell-09c-spec-llama",
    # ── Phase 3: Qwen ───────────────────────────────────────────────────
    "cell-06f-qwen",        # cleanup Llama + load Qwen
    "cell-07d-baseline-qwen",
    "cell-09d-spec-qwen",
    # ── Phase 4: Analysis ───────────────────────────────────────────────
    "cell-10-ablation",
    "cell-11-stats",
    "cell-11b-quality",
    "cell-12-linguistic",
    "cell-13-figures",
    "cell-14-push",
]

cell_map = {c["id"]: c for c in cells}

# Verify all expected IDs exist
missing = [cid for cid in DESIRED_ORDER if cid not in cell_map]
if missing:
    print("WARNING — missing cell IDs:", missing)

nb["cells"] = [cell_map[cid] for cid in DESIRED_ORDER if cid in cell_map]

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook reordered into 4 phases:")
for c in nb["cells"]:
    src = c["source"]
    first = (src[0] if src else "(empty)")[:60].replace("\n", "")
    print(f"  {c['id']:35s}  {first}")
