import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_STYLE = {
    "font.family":      "serif",
    "font.size":        11,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}
plt.rcParams.update(PLOT_STYLE)

_PALETTE = {
    "blue":   "#2196F3",
    "orange": "#FF9800",
    "red":    "#F44336",
    "green":  "#4CAF50",
    "purple": "#9C27B0",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save(fig, save_dir: Path, stem: str) -> list:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = save_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


# ── Individual figure generators ──────────────────────────────────────────────

def fig_acceptance_distribution(results_dict: dict, save_dir: Path) -> list:
    """
    Overlaid histogram of per-sample acceptance rates for each experimental
    condition supplied in results_dict (label → DataFrame).
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, df in results_dict.items():
        if "acceptance_rate" in df.columns and df["acceptance_rate"].notna().any():
            ar = df["acceptance_rate"].dropna().values
            ax.hist(ar, bins=25, alpha=0.55, density=True, label=label)

    ax.set_xlabel("Acceptance Rate")
    ax.set_ylabel("Density")
    ax.set_title("Draft Token Acceptance Rate Distribution")
    ax.legend(frameon=False)
    fig.tight_layout()

    return _save(fig, save_dir, "acceptance_distribution")


def fig_speedup_boxplot(results_dict: dict, baseline_dict: dict, save_dir: Path) -> list:
    """
    Box-plot of per-sample speedup ratios (baseline latency / speculative latency)
    for each condition in results_dict.  baseline_dict maps the same keys to their
    respective greedy baseline DataFrames.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    data, labels = [], []
    for label, df in results_dict.items():
        base_df = baseline_dict.get(label)
        if base_df is None:
            continue
        bl      = base_df["latency_ms"].values
        sp      = df["latency_ms"].values
        n       = min(len(bl), len(sp))
        speedup = bl[:n] / np.maximum(sp[:n], 1e-6)
        data.append(speedup)
        labels.append(label)

    if not data:
        return []

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)
    colors = [_PALETTE["blue"], _PALETTE["orange"]]
    for patch, color in zip(bp["boxes"], colors * len(data)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="No speedup (1×)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Speculative Decoding Speedup vs Greedy Baseline")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    return _save(fig, save_dir, "speedup_boxplot")


def fig_ablation_gamma(ablation_df: pd.DataFrame, save_dir: Path) -> list:
    """
    Two-panel plot: mean acceptance rate and mean latency as a function of
    the number of draft steps γ.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    grouped = ablation_df.groupby("draft_steps")
    steps   = sorted(ablation_df["draft_steps"].unique())
    mean_ar  = [grouped.get_group(s)["acceptance_rate"].mean() for s in steps]
    mean_lat = [grouped.get_group(s)["latency_ms"].mean()      for s in steps]

    axes[0].plot(steps, mean_ar,  marker="o", color=_PALETTE["blue"],   linewidth=2)
    axes[0].fill_between(
        steps,
        [grouped.get_group(s)["acceptance_rate"].quantile(0.25) for s in steps],
        [grouped.get_group(s)["acceptance_rate"].quantile(0.75) for s in steps],
        alpha=0.15, color=_PALETTE["blue"],
    )
    axes[0].set_xlabel("Draft Steps (γ)")
    axes[0].set_ylabel("Mean Acceptance Rate")
    axes[0].set_title("Acceptance Rate vs γ")
    axes[0].set_xticks(steps)

    axes[1].plot(steps, mean_lat, marker="s", color=_PALETTE["orange"], linewidth=2)
    axes[1].fill_between(
        steps,
        [grouped.get_group(s)["latency_ms"].quantile(0.25) for s in steps],
        [grouped.get_group(s)["latency_ms"].quantile(0.75) for s in steps],
        alpha=0.15, color=_PALETTE["orange"],
    )
    axes[1].set_xlabel("Draft Steps (γ)")
    axes[1].set_ylabel("Mean Latency (ms)")
    axes[1].set_title("Latency vs γ")
    axes[1].set_xticks(steps)

    fig.suptitle("Ablation Study: Effect of Draft Steps (γ)", fontsize=13)
    fig.tight_layout()

    return _save(fig, save_dir, "ablation_gamma")


def fig_latency_violin(results_dict: dict, baseline_dict: dict, save_dir: Path) -> list:
    """
    Violin + embedded box plot of per-sample speedup ratios for each model pair.
    Shows full distribution including the outlier structure that median alone hides.
    """
    labels, data = [], []
    for label, spec_df in results_dict.items():
        base_df = baseline_dict.get(label)
        if base_df is None:
            continue
        bl = base_df["latency_ms"].values
        sp = spec_df["latency_ms"].values
        k  = min(len(bl), len(sp))
        labels.append(label)
        data.append((bl[:k] / np.maximum(sp[:k], 1e-6)).tolist())

    if not data:
        return []

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 2), 5))

    colors = [_PALETTE["blue"], _PALETTE["purple"],
              _PALETTE["orange"], _PALETTE["red"]]

    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)

    # Embedded box-and-whisker
    bp = ax.boxplot(data, positions=range(len(data)),
                    widths=0.12, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2,
               label="No speedup (1×)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Latency Speedup Distribution (Violin + Box)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _save(fig, save_dir, "latency_violin")


def fig_fragmentation_comparison(oov_tr_df: pd.DataFrame,
                                  oov_en_df: pd.DataFrame,
                                  save_dir: Path) -> list:
    """
    Two-panel figure comparing subword fragmentation between Turkish and English.

    Left panel  : overlaid density histograms of fragment counts (1–5+).
    Right panel : mean fragments per task, error bars = 1 SD, for both languages.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Panel 1: histogram overlay ────────────────────────────────────────────
    for df, label, color in [
        (oov_tr_df, "Turkish", _PALETTE["blue"]),
        (oov_en_df, "English", _PALETTE["orange"]),
    ]:
        counts = df["fragments"].clip(upper=5)
        bins   = np.arange(0.5, 6.5, 1)
        axes[0].hist(counts, bins=bins, density=True, alpha=0.55,
                     label=label, color=color, edgecolor="white")
    axes[0].set_xlabel("Subword Fragments per Word")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Subword Fragment Distribution")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels(["1", "2", "3", "4", "5+"])
    axes[0].legend(frameon=False)

    # ── Panel 2: per-task mean ± SD ───────────────────────────────────────────
    tr_stats = oov_tr_df.groupby("task")["fragments"].agg(["mean", "std"]).reset_index()
    en_stats = oov_en_df.groupby("task")["fragments"].agg(["mean", "std"]).reset_index()

    all_tasks = sorted(set(tr_stats["task"]) | set(en_stats["task"]))
    x         = np.arange(len(all_tasks))
    w         = 0.35

    def _task_vals(stats_df):
        m = {r.task: r.mean for r in stats_df.itertuples()}
        s = {r.task: r.std  for r in stats_df.itertuples()}
        return [m.get(t, 0) for t in all_tasks], [s.get(t, 0) for t in all_tasks]

    tr_m, tr_s = _task_vals(tr_stats)
    en_m, en_s = _task_vals(en_stats)

    axes[1].bar(x - w/2, tr_m, w, yerr=tr_s, label="Turkish",
                color=_PALETTE["blue"],   alpha=0.8, capsize=4)
    axes[1].bar(x + w/2, en_m, w, yerr=en_s, label="English",
                color=_PALETTE["orange"], alpha=0.8, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_tasks, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean Fragments per Word")
    axes[1].set_title("Fragmentation by Task")
    axes[1].legend(frameon=False)

    fig.suptitle("Subword Fragmentation: Turkish vs English", fontsize=13)
    fig.tight_layout()
    return _save(fig, save_dir, "fragmentation_comparison")


def fig_quality_metrics(quality_df: pd.DataFrame, save_dir: Path) -> list:
    """
    Grouped bar chart of ROUGE-1/2/L (and BLEU where available) comparing
    greedy baseline vs speculative decoding for each language/task.

    quality_df must have columns: condition, metric, value.
    """
    if quality_df.empty:
        return []

    metrics  = quality_df["metric"].unique().tolist()
    conds    = quality_df["condition"].unique().tolist()
    n_m      = len(metrics)
    x        = np.arange(n_m)
    w        = 0.8 / max(len(conds), 1)
    colors   = [_PALETTE["blue"], _PALETTE["orange"],
                _PALETTE["green"], _PALETTE["purple"]] * 4

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, cond in enumerate(conds):
        sub   = quality_df[quality_df["condition"] == cond]
        yvals = [sub[sub["metric"] == m]["value"].mean() if not sub[sub["metric"] == m].empty
                 else 0.0 for m in metrics]
        ax.bar(x + i * w - (len(conds) - 1) * w / 2, yvals,
               w * 0.9, label=cond, color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Output Quality: Greedy vs Speculative")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    return _save(fig, save_dir, "quality_metrics")


def fig_position_acceptance(position_df: pd.DataFrame, save_dir: Path) -> list:
    """
    Bar chart of acceptance rates across early / mid / late token position buckets.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    order  = ["early", "mid", "late"]
    df     = position_df.set_index("position_bucket").reindex(order).reset_index()
    colors = [_PALETTE["blue"], _PALETTE["orange"], _PALETTE["red"]]

    bars = ax.bar(df["position_bucket"], df["acceptance_rate"],
                  color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Token Position Bucket")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Acceptance Rate by Token Position")
    ax.set_ylim(0, 1.1)

    for bar, row in zip(bars, df.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row.acceptance_rate + 0.03,
            f"{row.acceptance_rate:.2f}",
            ha="center", fontsize=10,
        )

    fig.tight_layout()
    return _save(fig, save_dir, "position_acceptance")


def fig_model_comparison(model_results: dict, baseline_dict: dict, save_dir: Path) -> list:
    """
    Side-by-side grouped bar chart comparing acceptance rate and median speedup
    across all model pairs (e.g., TR-small, TR-medium, EN-small, EN-medium).

    Parameters
    ----------
    model_results : label → speculative DataFrame (acceptance_rate, latency_ms)
    baseline_dict : label → greedy baseline DataFrame (latency_ms)
                    Keys should match those in model_results.
    """
    labels = list(model_results.keys())
    n      = len(labels)

    acceptance_means = []
    acceptance_cis   = []
    median_speedups  = []

    for label in labels:
        spec_df = model_results[label]
        ar = spec_df["acceptance_rate"].dropna().values
        acceptance_means.append(float(ar.mean()))
        sem = float(ar.std() / max(len(ar) ** 0.5, 1))
        acceptance_cis.append(sem * 1.96)   # ~95 % CI via normal approx

        base_df = baseline_dict.get(label)
        if base_df is not None:
            bl = base_df["latency_ms"].values
            sp = spec_df["latency_ms"].values
            k  = min(len(bl), len(sp))
            ratios = bl[:k] / np.maximum(sp[:k], 1e-6)
            median_speedups.append(float(np.median(ratios)))
        else:
            median_speedups.append(float("nan"))

    x      = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [_PALETTE["blue"], _PALETTE["purple"],
              _PALETTE["orange"], _PALETTE["red"]] * 4

    # ── Acceptance rate panel ─────────────────────────────────────────────────
    bars = axes[0].bar(x, acceptance_means, yerr=acceptance_cis,
                       color=colors[:n], edgecolor="white",
                       capsize=4, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean Acceptance Rate")
    axes[0].set_title("Acceptance Rate by Model Pair")
    axes[0].set_ylim(0, 1.05)
    for bar, val in zip(bars, acceptance_means):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     val + 0.02, f"{val:.3f}", ha="center", fontsize=8)

    # ── Median speedup panel ──────────────────────────────────────────────────
    valid = [(l, s) for l, s in zip(labels, median_speedups) if not np.isnan(s)]
    if valid:
        vlabels, vspeeds = zip(*valid)
        vx = np.arange(len(vlabels))
        bars2 = axes[1].bar(vx, vspeeds, color=colors[:len(vlabels)],
                            edgecolor="white", alpha=0.85)
        axes[1].axhline(1.0, color="crimson", linestyle="--",
                        linewidth=1.2, label="No speedup (1×)")
        axes[1].set_xticks(vx)
        axes[1].set_xticklabels(vlabels, rotation=15, ha="right", fontsize=9)
        axes[1].set_ylabel("Median Speedup (×)")
        axes[1].set_title("Median Speedup vs Greedy Baseline")
        axes[1].legend(frameon=False, fontsize=9)
        for bar, val in zip(bars2, vspeeds):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle("Cross-Model Comparison: Acceptance Rate & Speedup", fontsize=13)
    fig.tight_layout()
    return _save(fig, save_dir, "model_comparison")


# ── Master entry-point ────────────────────────────────────────────────────────

def generate_all_figures(results_dict: dict, save_dir) -> list:
    """
    Generate all publication-quality figures from the experiment results.

    Expected keys in results_dict
    ------------------------------
    baseline            : pd.DataFrame   (Turkish greedy baseline)
    baseline_en         : pd.DataFrame   (English greedy baseline)
    speculative_tr      : pd.DataFrame   (Turkish small-draft speculative)
    speculative_tr_med  : pd.DataFrame   (Turkish medium-draft speculative) [optional]
    speculative_en      : pd.DataFrame   (English small-draft speculative)
    speculative_en_med  : pd.DataFrame   (English medium-draft speculative) [optional]
    ablation            : pd.DataFrame   (γ ablation runs)
    position_acceptance : pd.DataFrame   (output of position_acceptance_analysis)
    oov_tr              : pd.DataFrame   (output of oov_analysis for Turkish)
    oov_en              : pd.DataFrame   (output of oov_analysis for English)
    quality             : pd.DataFrame   (columns: condition, metric, value) [optional]

    Returns
    -------
    List of saved file paths (PDF + PNG for every figure).
    """
    save_dir = Path(save_dir)
    all_paths: list = []

    baseline_df      = results_dict.get("baseline")
    baseline_en_df   = results_dict.get("baseline_en")
    baseline_llama   = results_dict.get("baseline_llama")
    baseline_qwen    = results_dict.get("baseline_qwen")
    spec_tr_df       = results_dict.get("speculative_tr")
    spec_tr_med_df   = results_dict.get("speculative_tr_med")
    spec_en_df       = results_dict.get("speculative_en")
    spec_en_med_df   = results_dict.get("speculative_en_med")
    spec_llama_df    = results_dict.get("speculative_llama")
    spec_qwen_df     = results_dict.get("speculative_qwen")
    ablation_df      = results_dict.get("ablation")
    position_df      = results_dict.get("position_acceptance")
    oov_tr_df        = results_dict.get("oov_tr")
    oov_en_df        = results_dict.get("oov_en")
    quality_df       = results_dict.get("quality")

    # All speculative conditions for distribution / violin figures
    spec_cond: dict = {}
    if spec_tr_df     is not None: spec_cond["TR-Small"]  = spec_tr_df
    if spec_tr_med_df is not None: spec_cond["TR-Medium"] = spec_tr_med_df
    if spec_en_df     is not None: spec_cond["EN-Small"]  = spec_en_df
    if spec_en_med_df is not None: spec_cond["EN-Medium"] = spec_en_med_df
    if spec_llama_df  is not None: spec_cond["Llama-1B→8B"] = spec_llama_df
    if spec_qwen_df   is not None: spec_cond["Qwen-0.5B→7B"] = spec_qwen_df

    # Baselines per condition (for speedup computation)
    baseline_per_cond: dict = {}
    if baseline_df is not None:
        if "TR-Small"  in spec_cond: baseline_per_cond["TR-Small"]  = baseline_df
        if "TR-Medium" in spec_cond: baseline_per_cond["TR-Medium"] = baseline_df
    if baseline_en_df is not None:
        if "EN-Small"  in spec_cond: baseline_per_cond["EN-Small"]  = baseline_en_df
        if "EN-Medium" in spec_cond: baseline_per_cond["EN-Medium"] = baseline_en_df
    if baseline_llama is not None:
        if "Llama-1B→8B"  in spec_cond: baseline_per_cond["Llama-1B→8B"]  = baseline_llama
    if baseline_qwen is not None:
        if "Qwen-0.5B→7B" in spec_cond: baseline_per_cond["Qwen-0.5B→7B"] = baseline_qwen

    _attempts = [
        (
            "fig_acceptance_distribution",
            lambda: fig_acceptance_distribution(spec_cond, save_dir),
            len(spec_cond) > 0,
        ),
        (
            "fig_latency_violin",
            lambda: fig_latency_violin(spec_cond, baseline_per_cond, save_dir),
            len(spec_cond) > 0 and len(baseline_per_cond) > 0,
        ),
        (
            "fig_speedup_boxplot",
            lambda: fig_speedup_boxplot(spec_cond, baseline_per_cond, save_dir),
            len(spec_cond) > 0 and len(baseline_per_cond) > 0,
        ),
        (
            "fig_ablation_gamma",
            lambda: fig_ablation_gamma(ablation_df, save_dir),
            ablation_df is not None,
        ),
        (
            "fig_position_acceptance",
            lambda: fig_position_acceptance(position_df, save_dir),
            position_df is not None and not position_df.empty,
        ),
        (
            "fig_model_comparison",
            lambda: fig_model_comparison(spec_cond, baseline_per_cond, save_dir),
            len(spec_cond) > 1,
        ),
        (
            "fig_fragmentation_comparison",
            lambda: fig_fragmentation_comparison(oov_tr_df, oov_en_df, save_dir),
            oov_tr_df is not None and oov_en_df is not None
            and not oov_tr_df.empty and not oov_en_df.empty,
        ),
        (
            "fig_quality_metrics",
            lambda: fig_quality_metrics(quality_df, save_dir),
            quality_df is not None and not quality_df.empty,
        ),
    ]

    for name, fn, condition in _attempts:
        if not condition:
            warnings.warn(f"Skipping {name}: required data not available.")
            continue
        try:
            all_paths += fn()
        except Exception as exc:
            warnings.warn(f"{name} failed: {exc}")

    return all_paths
