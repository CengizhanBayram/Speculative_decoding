import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


# ── Zeyrek morphological analyser (optional dependency) ───────────────────────
try:
    import zeyrek
    _analyzer        = zeyrek.MorphAnalyzer()
    ZEYREK_AVAILABLE = True
except ImportError:
    _analyzer        = None
    ZEYREK_AVAILABLE = False
    warnings.warn("zeyrek not installed. Morphological analysis will return UNKNOWN for all tokens.")


# Morpheme category labels
MORPHEME_CATEGORIES = [
    "ROOT_ONLY",
    "NOMINAL_SUFFIX",
    "VERBAL_SUFFIX",
    "DERIVATIONAL",
    "COMPOUND",
    "UNKNOWN",
]

_NOMINAL_FEATURES     = {"Gen", "Dat", "Acc", "Loc", "Abl", "Ins", "Nom",
                          "Pnon", "A3sg", "A1sg", "A2sg", "A3pl", "A1pl", "A2pl"}
_VERBAL_FEATURES      = {"Prog1", "Prog2", "Aor", "Past", "Narr",
                          "Cond", "Imp", "Opt", "Neces", "Fut"}
_DERIVATIONAL_FEATURES = {"Inf1", "Inf2", "Inf3", "Agt", "With", "Without",
                           "Noun", "Adj", "Adv"}


def _categorize_token(word: str) -> str:
    """Map a single Turkish surface form to one of the five morpheme categories."""
    if not ZEYREK_AVAILABLE or _analyzer is None:
        return "UNKNOWN"
    try:
        results = _analyzer.lemmatize(word)
        if not results:
            return "UNKNOWN"
        # results[0] = (lemma, List[MorphemeState])
        _lemma, analyses = results[0]
        if not analyses:
            return "UNKNOWN"

        analysis_str = str(analyses[0])
        parts        = [p.strip() for p in analysis_str.split("+")]
        suffixes     = set(parts[1:])

        if not suffixes:
            return "ROOT_ONLY"
        if suffixes & _VERBAL_FEATURES:
            return "VERBAL_SUFFIX"
        if suffixes & _DERIVATIONAL_FEATURES:
            return "DERIVATIONAL"
        # Compound heuristic: more than two morpheme boundaries
        if len(parts) > 3:
            return "COMPOUND"
        if suffixes & _NOMINAL_FEATURES:
            return "NOMINAL_SUFFIX"
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def analyze_morphology(tokens: list) -> list:
    """
    Return a list of dicts with keys {token, category} for each surface form.
    Categories: ROOT_ONLY | NOMINAL_SUFFIX | VERBAL_SUFFIX | DERIVATIONAL | COMPOUND | UNKNOWN
    """
    return [{"token": t, "category": _categorize_token(t)} for t in tokens]


def compute_rejection_by_morpheme(token_level_logs: list) -> pd.DataFrame:
    """
    Aggregate token-level accept/reject logs by morpheme category.

    Parameters
    ----------
    token_level_logs : list of per-sample logs (each is a list of dicts with
                       keys token_str and accepted).

    Returns
    -------
    DataFrame with columns: morpheme_category, accepted, rejected, total, rejection_rate.
    Sorted descending by rejection_rate.
    """
    cat_counts: dict = defaultdict(lambda: {"accepted": 0, "rejected": 0})

    for log in token_level_logs:
        for entry in log:
            token_str = entry.get("token_str", "")
            accepted  = entry.get("accepted",  False)
            category  = _categorize_token(token_str)
            key       = "accepted" if accepted else "rejected"
            cat_counts[category][key] += 1

    rows = []
    for cat, counts in cat_counts.items():
        total         = counts["accepted"] + counts["rejected"]
        rejection_rate = counts["rejected"] / max(total, 1)
        rows.append({
            "morpheme_category": cat,
            "accepted":          counts["accepted"],
            "rejected":          counts["rejected"],
            "total":             total,
            "rejection_rate":    round(rejection_rate, 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("rejection_rate", ascending=False)
        .reset_index(drop=True)
    )


def position_acceptance_analysis(logs: list) -> pd.DataFrame:
    """
    Divide each sample's token sequence into early / mid / late thirds and
    compute per-bucket acceptance rates across all samples.

    Returns
    -------
    DataFrame with columns: position_bucket, accepted, total, acceptance_rate.
    """
    bucket_counts: dict = defaultdict(lambda: {"accepted": 0, "total": 0})

    for log in logs:
        n = len(log)
        if n == 0:
            continue
        third = max(n // 3, 1)
        for i, entry in enumerate(log):
            bucket = "early" if i < third else ("mid" if i < 2 * third else "late")
            bucket_counts[bucket]["total"] += 1
            if entry.get("accepted", False):
                bucket_counts[bucket]["accepted"] += 1

    rows = []
    for bucket in ["early", "mid", "late"]:
        bc  = bucket_counts[bucket]
        acc = bc["accepted"]
        tot = bc["total"]
        rows.append({
            "position_bucket": bucket,
            "accepted":        acc,
            "total":           tot,
            "acceptance_rate": round(acc / max(tot, 1), 4),
        })

    return pd.DataFrame(rows)


def oov_analysis(samples: list, draft_tok, target_tok) -> pd.DataFrame:
    """
    For each word in every prompt measure tokenisation fragmentation in the
    draft and target vocabularies.  Higher fragment counts signal out-of-vocabulary
    (OOV) subword splits that typically increase rejection rates.

    Returns
    -------
    DataFrame with columns: word, draft_fragments, target_fragments,
    is_oov_draft, is_oov_target, task.
    DataFrame.attrs carries spearman_corr and spearman_p.
    """
    rows = []
    for sample in samples:
        task = sample.get("task", "unknown")
        for word in sample["prompt"].split():
            d_frags = len(draft_tok.encode(word,  add_special_tokens=False))
            t_frags = len(target_tok.encode(word, add_special_tokens=False))
            rows.append({
                "word":             word,
                "draft_fragments":  d_frags,
                "target_fragments": t_frags,
                "is_oov_draft":     d_frags > 1,
                "is_oov_target":    t_frags > 1,
                "task":             task,
            })

    df = pd.DataFrame(rows)
    if not df.empty and len(df) > 2:
        corr, p = stats.spearmanr(df["draft_fragments"], df["target_fragments"])
        df.attrs["spearman_corr"] = float(corr)
        df.attrs["spearman_p"]    = float(p)

    return df
