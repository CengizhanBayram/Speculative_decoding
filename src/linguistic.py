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
    warnings.warn(
        "zeyrek not installed. Morphological analysis will return UNKNOWN for all tokens."
    )


# Morpheme category labels
MORPHEME_CATEGORIES = [
    "ROOT_ONLY",
    "NOMINAL_SUFFIX",
    "VERBAL_SUFFIX",
    "DERIVATIONAL",
    "COMPOUND",
    "UNKNOWN",
]

_NOMINAL_FEATURES      = {"Gen", "Dat", "Acc", "Loc", "Abl", "Ins", "Nom",
                           "Pnon", "A3sg", "A1sg", "A2sg", "A3pl", "A1pl", "A2pl"}
_VERBAL_FEATURES       = {"Prog1", "Prog2", "Aor", "Past", "Narr",
                           "Cond", "Imp", "Opt", "Neces", "Fut"}
_DERIVATIONAL_FEATURES = {"Inf1", "Inf2", "Inf3", "Agt", "With", "Without",
                           "Noun", "Adj", "Adv"}


# ── Word-level morphological categorisation ──────────────────────────────────

def _categorize_word(word: str) -> str:
    """
    Map a Turkish surface-form WORD (not a subword fragment) to a morpheme category.

    Expects a complete, space-stripped word such as 'evlerde' or 'gidiyorum'.
    Subword fragments like 'lerde' or '##de' return 'UNKNOWN'.
    """
    word = word.strip()
    if not word or not ZEYREK_AVAILABLE or _analyzer is None:
        return "UNKNOWN"
    try:
        results = _analyzer.lemmatize(word)
        if not results:
            return "UNKNOWN"
        _, analyses = results[0]
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
        if len(parts) > 3:
            return "COMPOUND"
        if suffixes & _NOMINAL_FEATURES:
            return "NOMINAL_SUFFIX"
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ── GPT-2 subword → word grouping ─────────────────────────────────────────────

def _group_tokens_into_words(log: list) -> list:
    """
    Group GPT-2 BPE subword token entries into word-level groups.

    GPT-2 tokenizer prefixes word-initial tokens with a space character.
    A new word group begins whenever token_str starts with ' ' (or for the
    very first token in a sequence).

    Returns
    -------
    list of word-groups; each group is a list of token-entry dicts.
    """
    groups: list = []
    current: list = []
    for entry in log:
        tok = entry.get("token_str", "")
        if tok.startswith(" ") or not current:
            if current:
                groups.append(current)
            current = [entry]
        else:
            current.append(entry)
    if current:
        groups.append(current)
    return groups


def _reconstruct_word(group: list) -> str:
    """Join subword tokens in a group into a single surface-form string."""
    return "".join(e.get("token_str", "") for e in group).strip()


# ── Public analysis functions ─────────────────────────────────────────────────

def analyze_morphology(tokens: list) -> list:
    """
    Return a list of dicts {token, category} for each token string.
    Input tokens should be complete words, not GPT-2 subword fragments.
    """
    return [{"token": t, "category": _categorize_word(t)} for t in tokens]


def compute_rejection_by_morpheme(token_level_logs: list) -> pd.DataFrame:
    """
    Aggregate token-level accept/reject logs by Turkish morpheme category.

    GPT-2 BPE subword tokens are first aggregated into complete words using
    space-prefix heuristics; zeyrek then assigns each word a morpheme category.
    All subword tokens of the same word share that category.

    Parameters
    ----------
    token_level_logs : list of per-sample logs (each is a list of dicts with
                       keys token_str and accepted).

    Returns
    -------
    DataFrame with columns: morpheme_category, accepted, rejected, total,
    rejection_rate. Sorted descending by rejection_rate.
    """
    cat_counts: dict = defaultdict(lambda: {"accepted": 0, "rejected": 0})

    for log in token_level_logs:
        for group in _group_tokens_into_words(log):
            word     = _reconstruct_word(group)
            category = _categorize_word(word)
            for entry in group:
                key = "accepted" if entry.get("accepted", False) else "rejected"
                cat_counts[category][key] += 1

    rows = []
    for cat, counts in cat_counts.items():
        total          = counts["accepted"] + counts["rejected"]
        rejection_rate = counts["rejected"] / max(total, 1)
        rows.append({
            "morpheme_category": cat,
            "accepted":          counts["accepted"],
            "rejected":          counts["rejected"],
            "total":             total,
            "rejection_rate":    round(rejection_rate, 4),
        })

    if not rows:
        return pd.DataFrame(
            columns=["morpheme_category", "accepted", "rejected", "total", "rejection_rate"]
        )

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


def oov_analysis(samples: list, tokenizer) -> pd.DataFrame:
    """
    Analyse subword fragmentation per word using a single tokenizer.

    Since draft and target share the same tokenizer, fragmentation reflects
    the inherent morphological complexity of each language relative to its
    own vocabulary — not a cross-model alignment issue.

    Call this function separately for Turkish and English samples with their
    respective tokenizers to obtain a cross-linguistic comparison.

    Parameters
    ----------
    samples   : list of {prompt, task, ...} dicts.
    tokenizer : the tokenizer for the language of these samples.

    Returns
    -------
    DataFrame with columns: word, fragments, task, is_complex.
    DataFrame.attrs carries mean_fragments and fraction_complex per task.
    """
    rows = []
    for sample in samples:
        task = sample.get("task", "unknown")
        for word in sample["prompt"].split():
            frags = len(tokenizer.encode(word, add_special_tokens=False))
            rows.append({
                "word":       word,
                "fragments":  frags,
                "task":       task,
                "is_complex": frags > 1,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Per-task summary statistics stored in attrs
    task_stats = (
        df.groupby("task")["fragments"]
        .agg(mean="mean", median="median", std="std", fraction_complex=lambda x: (x > 1).mean())
        .round(4)
    )
    df.attrs["task_fragmentation"] = task_stats.to_dict()

    return df


def fragmentation_acceptance_analysis(token_level_logs: list) -> pd.DataFrame:
    """
    Correlate subword fragmentation count with per-word acceptance rate.

    Words requiring more subword tokens (higher fragmentation, i.e. morphologically
    complex or rare words) may exhibit lower acceptance rates in speculative
    decoding because the draft model's distribution diverges more from the target
    at morpheme boundaries.

    Parameters
    ----------
    token_level_logs : list of per-sample speculative decoding logs.

    Returns
    -------
    DataFrame with columns: word, fragments, mean_acceptance, n_tokens.
    DataFrame.attrs carries spearman_corr and spearman_p (Spearman ρ between
    fragment count and mean acceptance rate, aggregated at the word level).
    """
    rows = []
    for log in token_level_logs:
        for group in _group_tokens_into_words(log):
            word     = _reconstruct_word(group)
            frags    = len(group)
            mean_acc = sum(e.get("accepted", False) for e in group) / max(len(group), 1)
            rows.append({
                "word":            word,
                "fragments":       frags,
                "mean_acceptance": round(mean_acc, 4),
                "n_tokens":        len(group),
            })

    df = pd.DataFrame(rows)
    if not df.empty and len(df) > 2:
        # Aggregate to unique words to avoid frequency bias
        word_agg = df.groupby("fragments")["mean_acceptance"].mean().reset_index()
        if len(word_agg) > 2:
            corr, p = stats.spearmanr(word_agg["fragments"], word_agg["mean_acceptance"])
            df.attrs["spearman_corr"] = float(corr)
            df.attrs["spearman_p"]    = float(p)

    return df
