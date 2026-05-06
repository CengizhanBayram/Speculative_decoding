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


def fig_speedup_boxplot(results_dict: dict, baseline_df: pd.DataFrame, save_dir: Path) -> list:
    """
    Box-plot of per-sample speedup ratios (baseline latency / speculative latency)
    for each condition in results_dict.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    data, labels = [], []
    bl = baseline_df["latency_ms"].values

    for label, df in results_dict.items():
        sp      = df["latency_ms"].values
        n       = min(len(bl), len(sp))
        speedup = bl[:n] / np.maximum(sp[:n], 1e-6)
        data.append(speedup)
        labels.append(label)

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


def fig_morpheme_rejection(morpheme_df: pd.DataFrame, save_dir: Path) -> list:
    """
    Horizontal bar chart of rejection rates by Turkish morpheme category,
    coloured on a red-yellow-green diverging scale.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    df = morpheme_df.sort_values("rejection_rate", ascending=True).copy()
    n  = len(df)
    palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n))

    bars = ax.barh(df["morpheme_category"], df["rejection_rate"], color=palette)
    ax.set_xlabel("Rejection Rate")
    ax.set_title("Draft Token Rejection Rate by Turkish Morpheme Category")
    ax.set_xlim(0, 1)

    for bar, val in zip(bars, df["rejection_rate"]):
        ax.text(
            min(val + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9,
        )

    fig.tight_layout()
    return _save(fig, save_dir, "morpheme_rejection")


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


# ── Master entry-point ────────────────────────────────────────────────────────

def generate_all_figures(results_dict: dict, save_dir) -> list:
    """
    Generate all publication-quality figures from the experiment results.

    Expected keys in results_dict
    ------------------------------
    baseline            : pd.DataFrame   (greedy baseline runs)
    speculative_tr      : pd.DataFrame   (Turkish speculative runs)
    speculative_en      : pd.DataFrame   (English speculative runs)
    ablation            : pd.DataFrame   (γ ablation runs)
    morpheme_rejection  : pd.DataFrame   (output of compute_rejection_by_morpheme)
    position_acceptance : pd.DataFrame   (output of position_acceptance_analysis)

    Returns
    -------
    List of saved file paths (PDF + PNG for every figure).
    """
    save_dir = Path(save_dir)
    all_paths: list = []

    baseline_df  = results_dict.get("baseline")
    spec_tr_df   = results_dict.get("speculative_tr")
    spec_en_df   = results_dict.get("speculative_en")
    ablation_df  = results_dict.get("ablation")
    morpheme_df  = results_dict.get("morpheme_rejection")
    position_df  = results_dict.get("position_acceptance")

    spec_cond: dict = {}
    if spec_tr_df is not None:
        spec_cond["Turkish (Speculative)"] = spec_tr_df
    if spec_en_df is not None:
        spec_cond["English (Speculative)"] = spec_en_df

    _attempts = [
        (
            "fig_acceptance_distribution",
            lambda: fig_acceptance_distribution(spec_cond, save_dir),
            len(spec_cond) > 0,
        ),
        (
            "fig_speedup_boxplot",
            lambda: fig_speedup_boxplot(spec_cond, baseline_df, save_dir),
            len(spec_cond) > 0 and baseline_df is not None,
        ),
        (
            "fig_ablation_gamma",
            lambda: fig_ablation_gamma(ablation_df, save_dir),
            ablation_df is not None,
        ),
        (
            "fig_morpheme_rejection",
            lambda: fig_morpheme_rejection(morpheme_df, save_dir),
            morpheme_df is not None and not morpheme_df.empty,
        ),
        (
            "fig_position_acceptance",
            lambda: fig_position_acceptance(position_df, save_dir),
            position_df is not None and not position_df.empty,
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
