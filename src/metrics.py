import warnings
from typing import Optional

import numpy as np
import pandas as pd
import sacrebleu
from evaluate import load as eval_load
from scipy import stats


# ── Lazy-loaded ROUGE evaluator ───────────────────────────────────────────────
_rouge = None


def _get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = eval_load("rouge")
    return _rouge


# ── Per-sample quality metrics ────────────────────────────────────────────────

def compute_task_metrics(pred: str, ref: str, task: str) -> dict:
    """Compute ROUGE-1/2/L and (for QA tasks) corpus BLEU."""
    rouge        = _get_rouge()
    rouge_result = rouge.compute(predictions=[pred], references=[ref])

    metrics = {
        "rouge1":        rouge_result["rouge1"],
        "rouge2":        rouge_result["rouge2"],
        "rougeL":        rouge_result["rougeL"],
        "length_ratio":  len(pred.split()) / max(len(ref.split()), 1),
    }

    if "qa" in task:
        try:
            bleu = sacrebleu.corpus_bleu([pred], [[ref]])
            metrics["bleu"] = bleu.score / 100.0  # normalise to 0-1 (same scale as ROUGE)
        except Exception as exc:
            warnings.warn(f"BLEU computation failed: {exc}")
            metrics["bleu"] = 0.0

    return metrics


# ── Statistical utilities ─────────────────────────────────────────────────────

def bootstrap_ci(data: list, n: int = 10_000, ci: float = 0.95,
                 rng: Optional[np.random.Generator] = None) -> tuple:
    """
    Return (lower, upper) bootstrap confidence interval of the mean.

    Parameters
    ----------
    rng : optional numpy Generator for reproducibility. If None, uses the
          global numpy random state (seeded by seed_everything at startup).
    """
    arr = np.array(data, dtype=float)
    if rng is not None:
        boot_means = np.array([
            np.mean(rng.choice(arr, size=len(arr), replace=True))
            for _ in range(n)
        ])
    else:
        boot_means = np.array([
            np.mean(np.random.choice(arr, size=len(arr), replace=True))
            for _ in range(n)
        ])
    alpha = (1.0 - ci) / 2
    return (float(np.percentile(boot_means, alpha * 100)),
            float(np.percentile(boot_means, (1 - alpha) * 100)))


def wilcoxon_test(a: list, b: list) -> dict:
    """Wilcoxon signed-rank test on paired samples."""
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    stat, p = stats.wilcoxon(diff)
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def cohens_d(a: list, b: list) -> float:
    """
    Cohen's d effect size between two groups.

    Uses the pooled standard deviation with proper degrees of freedom
    (Hedges' formula), valid for unequal sample sizes.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a + n_b <= 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def mann_whitney_test(a: list, b: list) -> dict:
    """Two-sided Mann-Whitney U test for unpaired samples."""
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def compute_speedup(baseline_lats: list, spec_lats: list) -> dict:
    """Per-sample speedup ratio with bootstrapped 95 % CI."""
    bl     = np.array(baseline_lats, dtype=float)
    sp     = np.array(spec_lats,     dtype=float)
    ratios = bl / np.maximum(sp, 1e-6)
    lower, upper = bootstrap_ci(ratios.tolist())
    return {
        "mean_speedup":   float(ratios.mean()),
        "median_speedup": float(np.median(ratios)),
        "std_speedup":    float(ratios.std()),
        "ci_lower":       lower,
        "ci_upper":       upper,
    }


# ── Aggregated test battery ───────────────────────────────────────────────────

def run_all_statistical_tests(
    baseline_df:    pd.DataFrame,
    spec_tr_df:     pd.DataFrame,
    spec_en_df:     pd.DataFrame,
    baseline_en_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run speedup statistics and non-parametric tests for Turkish and English
    speculative decoding results vs their respective greedy baselines.

    Parameters
    ----------
    baseline_df    : greedy baseline latencies for Turkish samples.
    spec_tr_df     : Turkish speculative decoding results.
    spec_en_df     : English speculative decoding results.
    baseline_en_df : greedy baseline latencies for English samples (optional).
                     When provided, English speedup is computed against this;
                     otherwise the Turkish baseline is reused (less accurate).

    Returns a flat dict suitable for JSON serialisation.
    """
    results: dict = {}

    conditions = [
        ("tr", spec_tr_df, baseline_df),
        ("en", spec_en_df, baseline_en_df if baseline_en_df is not None else baseline_df),
    ]

    for label, spec_df, base_df in conditions:
        lat_base = base_df["latency_ms"].tolist()
        lat_spec = spec_df["latency_ms"].tolist()
        min_len  = min(len(lat_base), len(lat_spec))
        lat_base = lat_base[:min_len]
        lat_spec = lat_spec[:min_len]

        results[f"speedup_{label}"]              = compute_speedup(lat_base, lat_spec)
        results[f"wilcoxon_latency_{label}"]     = wilcoxon_test(lat_base, lat_spec)
        results[f"mann_whitney_latency_{label}"] = mann_whitney_test(lat_base, lat_spec)
        results[f"cohens_d_latency_{label}"]     = cohens_d(lat_base, lat_spec)

        if "acceptance_rate" in spec_df.columns:
            ar = spec_df["acceptance_rate"].dropna().tolist()
            if ar:
                lower, upper = bootstrap_ci(ar)
                results[f"acceptance_rate_ci_{label}"] = {
                    "mean":     float(np.mean(ar)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }

    return results
