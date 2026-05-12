from datasets import load_dataset


def load_xquad_tr(n: int, seed: int) -> list:
    """
    Load a Turkish QA dataset (XQuAD-TR subset).

    XQuAD (Artetxe et al., 2020) is a multilingual reading-comprehension
    benchmark with SQuAD-format annotations. The Turkish subset ('xquad.tr')
    is used here as the primary Turkish QA source.

    Falls back to alternative hub paths if the primary one fails.
    Each item: {prompt, reference, task}.
    """
    candidates = [
        # (hub_name, config_name, split)
        ("google/xquad",                          "xquad.tr", "validation"),
        ("boun-tabilab/XQuAD-TR",                 None,       "validation"),
        ("gorkemgoknar/tr-nlp-qa-xquad-trquad",  None,       "train"),
    ]

    ds = None
    for name, config, split in candidates:
        try:
            if config:
                ds = load_dataset(name, config, split=split)
            else:
                ds = load_dataset(name, split=split)
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError(
            "Could not load a Turkish QA dataset. "
            "Check hub names / internet connectivity in src/data.py."
        )

    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    samples = []
    for item in ds:
        question = item.get("question", "")
        context  = item.get("context",  "")
        answers  = item.get("answers",  {})
        ref_list = answers.get("text", []) if isinstance(answers, dict) else []
        reference = ref_list[0] if ref_list else ""

        prompt = f"Soru: {question}\nBağlam: {context[:300]}\nCevap:"
        samples.append({"prompt": prompt, "reference": reference, "task": "qa_tr"})

    return samples


def load_trnews(n: int, seed: int) -> list:
    """
    Load TR-News Turkish summarisation dataset.
    Each item: {prompt, reference, task}.
    Falls back to alternative splits if the primary one is unavailable.
    """
    candidates = [
        ("batubayk/TR-News", "test"),
        ("batubayk/TR-News", "train"),
        ("batubayk/TR-News", "validation"),
    ]
    ds = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError(
            "Could not load TR-News dataset. Check hub names in src/data.py."
        )

    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    samples = []
    for item in ds:
        article   = item.get("content", item.get("text",    ""))
        reference = item.get("title",   item.get("summary", ""))

        prompt = f"Aşağıdaki haberi özetle:\n{article[:400]}\nÖzet:"
        samples.append({"prompt": prompt, "reference": reference, "task": "summarization_tr"})

    return samples


def load_xquad_tr_instruct(n: int, seed: int) -> list:
    """
    Turkish QA samples formatted for instruction-tuned models (Llama / Qwen chat style).

    Uses a system prompt + user turn structure that instruction-tuned models
    respond to reliably, rather than the raw completion format used for GPT-2.
    """
    candidates = [
        ("google/xquad",                          "xquad.tr", "validation"),
        ("boun-tabilab/XQuAD-TR",                 None,       "validation"),
        ("gorkemgoknar/tr-nlp-qa-xquad-trquad",  None,       "train"),
    ]
    ds = None
    for name, config, split in candidates:
        try:
            ds = load_dataset(name, config, split=split) if config \
                 else load_dataset(name, split=split)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load XQuAD-TR for instruction format.")

    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    samples = []
    for item in ds:
        question  = item.get("question", "")
        context   = item.get("context",  "")
        answers   = item.get("answers",  {})
        ref_list  = answers.get("text", []) if isinstance(answers, dict) else []
        reference = ref_list[0] if ref_list else ""
        prompt = (
            "<|system|>\nSen bir Türkçe soru-cevap asistanısın.\n<|user|>\n"
            f"Bağlam: {context[:300]}\n\nSoru: {question}\n<|assistant|>\nCevap:"
        )
        samples.append({"prompt": prompt, "reference": reference, "task": "qa_tr_instruct"})
    return samples


def load_squad_en_instruct(n: int, seed: int) -> list:
    """
    English SQuAD samples formatted for instruction-tuned models (Qwen / Llama chat style).
    """
    ds = None
    for name in ("rajpurkar/squad", "squad"):
        try:
            ds = load_dataset(name, split="validation")
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load SQuAD for instruction format.")

    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    samples = []
    for item in ds:
        question  = item.get("question", "")
        context   = item.get("context",  "")
        answers   = item.get("answers",  {})
        ref_list  = answers.get("text", []) if isinstance(answers, dict) else []
        reference = ref_list[0] if ref_list else ""
        prompt = (
            "<|system|>\nYou are a helpful QA assistant.\n<|user|>\n"
            f"Context: {context[:300]}\n\nQuestion: {question}\n<|assistant|>\nAnswer:"
        )
        samples.append({"prompt": prompt, "reference": reference, "task": "qa_en_instruct"})
    return samples


def load_squad_en(n: int, seed: int) -> list:
    """
    Load SQuAD English QA dataset.
    Each item: {prompt, reference, task}.
    """
    ds = None
    for name in ("rajpurkar/squad", "squad"):
        try:
            ds = load_dataset(name, split="validation")
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError("Could not load SQuAD dataset.")

    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    samples = []
    for item in ds:
        question = item.get("question", "")
        context  = item.get("context",  "")
        answers  = item.get("answers",  {})
        ref_list = answers.get("text", []) if isinstance(answers, dict) else []
        reference = ref_list[0] if ref_list else ""

        prompt = f"Question: {question}\nContext: {context[:300]}\nAnswer:"
        samples.append({"prompt": prompt, "reference": reference, "task": "qa_en"})

    return samples
