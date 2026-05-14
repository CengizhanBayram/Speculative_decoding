from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


# ── Stanza morphological analysis ─────────────────────────────────────────────

def stanza_morphology_analysis(samples: list, lang: str = "tr") -> pd.DataFrame:
    """
    Characterise morphological complexity of a sample set using Stanza.

    For each word in the prompts, extracts the Universal POS tag, the full
    morphological feature string (Case=Acc|Number=Sing|Tense=Past|…), and
    counts the number of active features (n_feats) as a single scalar measure
    of surface complexity.

    Turkish typically yields 3–8 features per content word (case, number,
    person, tense, aspect, polarity, …); English typically 0–2 (number,
    tense).  This gap provides the linguistic grounding for why Turkish
    agglutination creates harder prediction targets for the draft model.

    Call separately for Turkish and English corpora, then compare mean_n_feats
    to demonstrate cross-linguistic complexity difference.

    Prerequisites (run once before calling):
        pip install stanza
        python -c "import stanza; stanza.download('<lang>')"

    Parameters
    ----------
    samples : list of dicts with keys "prompt" and "task"
    lang    : Stanza language code ("tr" for Turkish, "en" for English)

    Returns
    -------
    DataFrame with columns: word, lemma, upos, feats, n_feats, task.
    DataFrame.attrs carries mean_n_feats and std_n_feats summary statistics.
    """
    try:
        import stanza
    except ImportError:
        raise ImportError(
            "stanza is not installed. Run: pip install stanza && "
            f"python -c \"import stanza; stanza.download('{lang}')\""
        )

    import logging
    logging.getLogger("stanza").setLevel(logging.ERROR)

    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    nlp = stanza.Pipeline(
        lang,
        processors="tokenize,pos,lemma",
        use_gpu=use_gpu,
        verbose=False,
    )

    rows = []
    for sample in samples:
        task = sample.get("task", "unknown")
        doc  = nlp(sample["prompt"])
        for sent in doc.sentences:
            for word in sent.words:
                feats_str = word.feats or ""
                n_feats   = (
                    len(feats_str.split("|"))
                    if feats_str and feats_str != "_"
                    else 0
                )
                rows.append({
                    "word":    word.text,
                    "lemma":   word.lemma,
                    "upos":    word.upos,
                    "feats":   feats_str,
                    "n_feats": n_feats,
                    "task":    task,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["mean_n_feats"] = float(round(df["n_feats"].mean(), 4))
        df.attrs["std_n_feats"]  = float(round(df["n_feats"].std(),  4))

    return df


def case_distribution_analysis(stanza_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract grammatical case distribution from Stanza morphological features.

    Parses the 'feats' column for Case=X entries.  Useful for showing that
    Turkish exhibits a richer overt case inventory (Nominative, Accusative,
    Dative, Genitive, Locative, Ablative, Instrumental) compared to English,
    where case is only marked on pronouns.

    Parameters
    ----------
    stanza_df : DataFrame returned by stanza_morphology_analysis

    Returns
    -------
    DataFrame with columns: case, count, proportion.  Sorted by count
    descending.
    """
    case_counts: dict = {}
    for feats_str in stanza_df.get("feats", pd.Series(dtype=str)):
        if not feats_str or feats_str == "_":
            continue
        for feat in feats_str.split("|"):
            if feat.startswith("Case="):
                case = feat.split("=", 1)[1]
                case_counts[case] = case_counts.get(case, 0) + 1

    total = sum(case_counts.values())
    rows  = [
        {
            "case":       case,
            "count":      count,
            "proportion": round(count / max(total, 1), 4),
        }
        for case, count in sorted(case_counts.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


# ── GPT-2 subword → word grouping ─────────────────────────────────────────────

def _group_tokens_into_words(log: list) -> list:
    """
    Group GPT-2 BPE subword token entries into word-level groups.

    GPT-2 tokenizer prefixes word-initial tokens with a space character.
    A new word group begins whenever token_str starts with ' ' (or for the
    very first token in a sequence).
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
    own vocabulary.

    Call separately for Turkish and English samples with their respective
    tokenizers to obtain a cross-linguistic comparison.

    Returns
    -------
    DataFrame with columns: word, fragments, task, is_complex.
    DataFrame.attrs carries task_fragmentation summary statistics.
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

    task_stats = (
        df.groupby("task")["fragments"]
        .agg(mean="mean", median="median", std="std",
             fraction_complex=lambda x: (x > 1).mean())
        .round(4)
    )
    df.attrs["task_fragmentation"] = task_stats.to_dict()

    return df


def fragmentation_acceptance_analysis(token_level_logs: list) -> pd.DataFrame:
    """
    Correlate subword fragment count with per-word acceptance rate.

    Words requiring more subword tokens may exhibit lower acceptance rates
    if the draft model diverges from the target at morpheme boundaries.

    Returns
    -------
    DataFrame with columns: word, fragments, mean_acceptance, n_tokens.
    DataFrame.attrs carries spearman_corr and spearman_p.
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
        word_agg = df.groupby("fragments")["mean_acceptance"].mean().reset_index()
        if len(word_agg) > 2:
            corr, p = stats.spearmanr(word_agg["fragments"], word_agg["mean_acceptance"])
            df.attrs["spearman_corr"] = float(corr)
            df.attrs["spearman_p"]    = float(p)

    return df


def rejected_token_analysis(token_level_logs: list, top_n: int = 30) -> pd.DataFrame:
    """
    Frequency analysis of draft tokens that were rejected by the target model.

    For each unique token string, counts total draft proposals and rejections
    and computes a per-token rejection rate. Useful for identifying systematic
    failure modes (e.g., certain suffixes, function words, punctuation).

    Returns
    -------
    DataFrame with columns: token_str, total, rejected, rejection_rate.
    Sorted by total proposals descending (most common first).
    Top `top_n` rows returned.
    """
    from collections import Counter

    total_counter:    Counter = Counter()
    rejected_counter: Counter = Counter()

    for log in token_level_logs:
        for entry in log:
            tok = entry.get("token_str", "")
            total_counter[tok] += 1
            if not entry.get("accepted", True):
                rejected_counter[tok] += 1

    rows = []
    for tok, total in total_counter.most_common(top_n):
        rejected = rejected_counter.get(tok, 0)
        rows.append({
            "token_str":      tok,
            "total":          total,
            "rejected":       rejected,
            "rejection_rate": round(rejected / max(total, 1), 4),
        })

    return pd.DataFrame(rows)
