"""Rebuild run_experiments.ipynb with all bug fixes applied."""
import json

with open("run_experiments.ipynb") as f:
    nb = json.load(f)

def code(cell_id, lines):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }

md_title = nb["cells"][0]  # keep markdown title as-is

cell_01 = code("cell-01-auth", [
    "# ── Cell 1: Mount Google Drive, read tokens, configure git, login to HF ──────\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive', force_remount=True)\n",
    "\n",
    "import os, subprocess\n",
    "\n",
    "SECRETS_DIR   = '/content/drive/MyDrive/secrets'\n",
    "HF_TOKEN_PATH = os.path.join(SECRETS_DIR, 'hf_token.txt')\n",
    "GH_TOKEN_PATH = os.path.join(SECRETS_DIR, 'gh_token.txt')\n",
    "\n",
    "with open(HF_TOKEN_PATH) as _f:\n",
    "    HF_TOKEN = _f.read().strip()\n",
    "\n",
    "with open(GH_TOKEN_PATH) as _f:\n",
    "    GH_TOKEN = _f.read().strip()\n",
    "\n",
    "# Configure git identity\n",
    "subprocess.run(['git', 'config', '--global', 'user.email', 'cengizhanbayramtr@gmail.com'], check=True)\n",
    "subprocess.run(['git', 'config', '--global', 'user.name',  'CengizhanBayram'],              check=True)\n",
    "subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'],                  check=True)\n",
    "with open(os.path.expanduser('~/.git-credentials'), 'w') as _f:\n",
    "    _f.write(f'https://oauth2:{GH_TOKEN}@github.com\\n')\n",
    "\n",
    "from huggingface_hub import login as hf_login\n",
    "hf_login(token=HF_TOKEN)\n",
    "print('Authentication complete.')\n",
])

cell_02 = code("cell-02-clone", [
    "# ── Cell 2: Clone repo (or pull), install dependencies ───────────────────────\n",
    "import os, sys, subprocess\n",
    "\n",
    "GITHUB_USER = 'CengizhanBayram'\n",
    "REPO_NAME   = 'Speculative_decoding'\n",
    "REPO_URL    = f'https://github.com/{GITHUB_USER}/{REPO_NAME}.git'\n",
    "REPO_DIR    = f'/content/{REPO_NAME}'\n",
    "\n",
    "if not os.path.exists(REPO_DIR):\n",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)\n",
    "    print(f'Cloned into {REPO_DIR}')\n",
    "else:\n",
    "    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)\n",
    "    print(f'Pulled latest into {REPO_DIR}')\n",
    "\n",
    "subprocess.run(\n",
    "    [sys.executable, '-m', 'pip', 'install', '-r',\n",
    "     os.path.join(REPO_DIR, 'requirements.txt'), '-q'],\n",
    "    check=True,\n",
    ")\n",
    "print('Dependencies installed.')\n",
])

cell_03 = code("cell-03-imports", [
    "# ── Cell 3: Add repo to path and import everything from src/ ─────────────────\n",
    "import sys\n",
    "sys.path.insert(0, REPO_DIR)\n",
    "\n",
    "from src.config import (\n",
    "    SEED,\n",
    "    DRAFT_MODEL_NAME, TARGET_MODEL_NAME,\n",
    "    DRAFT_MODEL_EN_NAME, TARGET_MODEL_EN_NAME,\n",
    "    MAX_NEW_TOKENS, DRAFT_STEPS_LIST, DEFAULT_DRAFT_STEPS,\n",
    "    NUM_SAMPLES_QA, NUM_SAMPLES_SUM, NUM_SAMPLES_EN,\n",
    "    QUANTIZATION_BITS, RESULTS_DIR, FIGURES_DIR,\n",
    ")\n",
    "from src.utils      import seed_everything, save_json, check_gpu, git_push\n",
    "from src.data       import load_tquad, load_trnews, load_squad_en\n",
    "from src.models     import load_draft_model, load_target_model\n",
    "from src.speculative import run_experiment\n",
    "from src.metrics    import (\n",
    "    compute_task_metrics, bootstrap_ci,\n",
    "    wilcoxon_test, cohens_d, mann_whitney_test,\n",
    "    compute_speedup, run_all_statistical_tests,\n",
    ")\n",
    "from src.linguistic import (\n",
    "    analyze_morphology,\n",
    "    compute_rejection_by_morpheme,\n",
    "    position_acceptance_analysis,\n",
    "    oov_analysis,\n",
    ")\n",
    "from src.figures import generate_all_figures\n",
    "\n",
    "print('All imports successful.')\n",
])

cell_04 = code("cell-04-seed-gpu", [
    "# ── Cell 4: Seed RNG, verify GPU ─────────────────────────────────────────────\n",
    "seed_everything(SEED)\n",
    "gpu_info = check_gpu()\n",
    "print(gpu_info)\n",
])

cell_05 = code("cell-05-data", [
    "# ── Cell 5: Load datasets ─────────────────────────────────────────────────────\n",
    "tquad_samples  = load_tquad(n=NUM_SAMPLES_QA,  seed=SEED)\n",
    "trnews_samples = load_trnews(n=NUM_SAMPLES_SUM, seed=SEED)\n",
    "squad_samples  = load_squad_en(n=NUM_SAMPLES_EN, seed=SEED)\n",
    "\n",
    "print(f'TQuAD     : {len(tquad_samples):>4d} samples')\n",
    "print(f'TR-News   : {len(trnews_samples):>4d} samples')\n",
    "print(f'SQuAD EN  : {len(squad_samples):>4d} samples')\n",
    "print('\\nSample prompt (TQuAD):', tquad_samples[0]['prompt'][:120])\n",
])

cell_06 = code("cell-06-models-tr", [
    "# ── Cell 6: Load Turkish draft + target models ───────────────────────────────\n",
    "# Both ytu-ce-cosmos models share the same GPT-2 BPE tokenizer (50,257 tokens).\n",
    "# Shared vocabulary is a hard requirement for speculative decoding.\n",
    "import torch\n",
    "\n",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "\n",
    "draft_model_tr,  draft_tok_tr  = load_draft_model(DRAFT_MODEL_NAME,   device=DEVICE)\n",
    "target_model_tr, target_tok_tr = load_target_model(TARGET_MODEL_NAME,  bits=QUANTIZATION_BITS)\n",
    "\n",
    "print('Draft TR  :', DRAFT_MODEL_NAME)\n",
    "print('Target TR :', TARGET_MODEL_NAME)\n",
    "print('Turkish models loaded.')\n",
])

cell_06b = code("cell-06b-models-en", [
    "# ── Cell 6b: Load English draft + target models ──────────────────────────────\n",
    "# gpt2 and gpt2-xl share the same GPT-2 BPE tokenizer.\n",
    "draft_model_en,  draft_tok_en  = load_draft_model(DRAFT_MODEL_EN_NAME,   device=DEVICE)\n",
    "target_model_en, target_tok_en = load_target_model(TARGET_MODEL_EN_NAME,  bits=QUANTIZATION_BITS)\n",
    "\n",
    "print('Draft EN  :', DRAFT_MODEL_EN_NAME)\n",
    "print('Target EN :', TARGET_MODEL_EN_NAME)\n",
    "print('English models loaded.')\n",
])

cell_07 = code("cell-07-baseline-tr", [
    "# ── Cell 7: Greedy baseline — Turkish ───────────────────────────────────────\n",
    "import pandas as pd\n",
    "\n",
    "tr_samples_all = tquad_samples + trnews_samples\n",
    "\n",
    "baseline_tr_df = run_experiment(\n",
    "    samples        = tr_samples_all,\n",
    "    target_model   = target_model_tr,\n",
    "    target_tok     = target_tok_tr,\n",
    "    mode           = 'greedy',\n",
    "    max_new_tokens = MAX_NEW_TOKENS,\n",
    ")\n",
    "\n",
    "out_path = RESULTS_DIR / 'baseline_tr_results.csv'\n",
    "baseline_tr_df.drop(columns=['token_level_log']).to_csv(out_path, index=False)\n",
    "print(f'Saved -> {out_path}')\n",
    "print(baseline_tr_df[['task', 'latency_ms']].groupby('task').describe())\n",
])

cell_07b = code("cell-07b-baseline-en", [
    "# ── Cell 7b: Greedy baseline — English ──────────────────────────────────────\n",
    "baseline_en_df = run_experiment(\n",
    "    samples        = squad_samples,\n",
    "    target_model   = target_model_en,\n",
    "    target_tok     = target_tok_en,\n",
    "    mode           = 'greedy',\n",
    "    max_new_tokens = MAX_NEW_TOKENS,\n",
    ")\n",
    "\n",
    "out_path = RESULTS_DIR / 'baseline_en_results.csv'\n",
    "baseline_en_df.drop(columns=['token_level_log']).to_csv(out_path, index=False)\n",
    "print(f'Saved -> {out_path}')\n",
    "print(baseline_en_df[['task', 'latency_ms']].groupby('task').describe())\n",
])

cell_08 = code("cell-08-spec-tr", [
    "# ── Cell 8: Speculative decoding — Turkish ────────────────────────────────────\n",
    "speculative_tr_df = run_experiment(\n",
    "    samples        = tr_samples_all,\n",
    "    draft_model    = draft_model_tr,\n",
    "    draft_tok      = draft_tok_tr,\n",
    "    target_model   = target_model_tr,\n",
    "    target_tok     = target_tok_tr,\n",
    "    mode           = 'speculative',\n",
    "    draft_steps    = DEFAULT_DRAFT_STEPS,\n",
    "    max_new_tokens = MAX_NEW_TOKENS,\n",
    ")\n",
    "\n",
    "out_path = RESULTS_DIR / 'speculative_tr_results.csv'\n",
    "speculative_tr_df.drop(columns=['token_level_log']).to_csv(out_path, index=False)\n",
    "print(f'Saved -> {out_path}')\n",
    "print(speculative_tr_df[['task', 'latency_ms', 'acceptance_rate']].groupby('task').mean())\n",
])

cell_09 = code("cell-09-spec-en", [
    "# ── Cell 9: Speculative decoding — English ────────────────────────────────────\n",
    "# Uses gpt2 (draft) + gpt2-xl (target): shared GPT-2 BPE vocabulary.\n",
    "speculative_en_df = run_experiment(\n",
    "    samples        = squad_samples,\n",
    "    draft_model    = draft_model_en,\n",
    "    draft_tok      = draft_tok_en,\n",
    "    target_model   = target_model_en,\n",
    "    target_tok     = target_tok_en,\n",
    "    mode           = 'speculative',\n",
    "    draft_steps    = DEFAULT_DRAFT_STEPS,\n",
    "    max_new_tokens = MAX_NEW_TOKENS,\n",
    ")\n",
    "\n",
    "out_path = RESULTS_DIR / 'speculative_en_results.csv'\n",
    "speculative_en_df.drop(columns=['token_level_log']).to_csv(out_path, index=False)\n",
    "print(f'Saved -> {out_path}')\n",
    "print(speculative_en_df[['task', 'latency_ms', 'acceptance_rate']].groupby('task').mean())\n",
])

cell_10 = code("cell-10-ablation", [
    "# ── Cell 10: Ablation over gamma (draft steps) — Turkish ────────────────────\n",
    "ablation_frames = []\n",
    "\n",
    "for gamma in DRAFT_STEPS_LIST:\n",
    "    _df = run_experiment(\n",
    "        samples        = tr_samples_all[:100],\n",
    "        draft_model    = draft_model_tr,\n",
    "        draft_tok      = draft_tok_tr,\n",
    "        target_model   = target_model_tr,\n",
    "        target_tok     = target_tok_tr,\n",
    "        mode           = 'speculative',\n",
    "        draft_steps    = gamma,\n",
    "        max_new_tokens = MAX_NEW_TOKENS,\n",
    "    )\n",
    "    ablation_frames.append(_df)\n",
    "\n",
    "ablation_df = pd.concat(ablation_frames, ignore_index=True)\n",
    "\n",
    "out_path = RESULTS_DIR / 'ablation_gamma.csv'\n",
    "ablation_df.drop(columns=['token_level_log']).to_csv(out_path, index=False)\n",
    "print(f'Saved -> {out_path}')\n",
    "print(ablation_df.groupby('draft_steps')[['latency_ms', 'acceptance_rate']].mean())\n",
])

cell_11 = code("cell-11-stats", [
    "# ── Cell 11: Statistical tests ────────────────────────────────────────────────\n",
    "stat_results = run_all_statistical_tests(\n",
    "    baseline_df    = baseline_tr_df,\n",
    "    spec_tr_df     = speculative_tr_df,\n",
    "    spec_en_df     = speculative_en_df,\n",
    "    baseline_en_df = baseline_en_df,   # English compared against English baseline\n",
    ")\n",
    "\n",
    "out_path = RESULTS_DIR / 'statistical_tests.json'\n",
    "save_json(stat_results, out_path)\n",
    "print(f'Saved -> {out_path}')\n",
    "\n",
    "for k, v in stat_results.items():\n",
    "    print(f'  {k}: {v}')\n",
])

cell_12 = code("cell-12-linguistic", [
    "# ── Cell 12: Linguistic / morphological analysis ──────────────────────────────\n",
    "tr_logs = speculative_tr_df['token_level_log'].tolist()\n",
    "\n",
    "morpheme_df = compute_rejection_by_morpheme(tr_logs)\n",
    "position_df = position_acceptance_analysis(tr_logs)\n",
    "oov_df      = oov_analysis(tr_samples_all, draft_tok_tr, target_tok_tr)\n",
    "\n",
    "morpheme_df.to_csv(RESULTS_DIR / 'morpheme_rejection.csv',  index=False)\n",
    "position_df.to_csv(RESULTS_DIR / 'position_acceptance.csv', index=False)\n",
    "oov_df.to_csv(     RESULTS_DIR / 'oov_analysis.csv',        index=False)\n",
    "\n",
    "print('Morpheme rejection rates:')\n",
    "print(morpheme_df.to_string(index=False))\n",
    "\n",
    "print('\\nPosition acceptance rates:')\n",
    "print(position_df.to_string(index=False))\n",
    "\n",
    "if 'spearman_corr' in oov_df.attrs:\n",
    "    print(f\"OOV Spearman r = {oov_df.attrs['spearman_corr']:.4f} \"\n",
    "          f\"(p = {oov_df.attrs['spearman_p']:.4f})\")\n",
])

cell_13 = code("cell-13-figures", [
    "# ── Cell 13: Generate all publication-quality figures ────────────────────────\n",
    "results_for_figures = {\n",
    "    'baseline':            baseline_tr_df,\n",
    "    'speculative_tr':      speculative_tr_df,\n",
    "    'speculative_en':      speculative_en_df,\n",
    "    'ablation':            ablation_df,\n",
    "    'morpheme_rejection':  morpheme_df,\n",
    "    'position_acceptance': position_df,\n",
    "}\n",
    "\n",
    "saved_paths = generate_all_figures(results_for_figures, save_dir=FIGURES_DIR)\n",
    "\n",
    "print(f'Generated {len(saved_paths)} figure files:')\n",
    "for p in saved_paths:\n",
    "    print(f'  {p}')\n",
])

cell_14 = code("cell-14-push", [
    "# ── Cell 14: Commit and push results to GitHub ────────────────────────────────\n",
    "from datetime import datetime\n",
    "\n",
    "commit_msg  = f\"results: {datetime.now().isoformat()[:19]}\"\n",
    "commit_hash = git_push(message=commit_msg, repo_dir=REPO_DIR)\n",
    "\n",
    "print(f'Pushed  : {commit_hash}')\n",
    "print(f'Message : {commit_msg}')\n",
])

nb["cells"] = [
    md_title,
    cell_01, cell_02, cell_03, cell_04, cell_05,
    cell_06, cell_06b,
    cell_07, cell_07b,
    cell_08, cell_09, cell_10, cell_11, cell_12, cell_13, cell_14,
]

with open("run_experiments.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. {len(nb['cells'])} cells written.")
