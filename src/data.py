from datasets import load_dataset


def load_tquad(n: int, seed: int) -> list:
    """
    Load TQuAD Turkish QA dataset.
    Each item: {prompt, reference, task}.
    Falls back to alternative hub paths if the primary name fails.
    """
    candidates = [
        ("erdemkan/tquad", "validation"),
        ("husseinalyasah/tquad", "validation"),
        ("maydogan/TQuAD", "validation"),
    ]
    ds = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(
            "Could not load a TQuAD dataset. Check hub names in src/data.py."
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
    """
    ds = load_dataset("batubayk/TR-News", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    samples = []
    for item in ds:
        article   = item.get("content", item.get("text", ""))
        reference = item.get("title",   item.get("summary", ""))

        prompt = f"Aşağıdaki haberi özetle:\n{article[:400]}\nÖzet:"
        samples.append({"prompt": prompt, "reference": reference, "task": "summarization_tr"})

    return samples


def load_squad_en(n: int, seed: int) -> list:
    """
    Load SQuAD English QA dataset.
    Each item: {prompt, reference, task}.
    """
    try:
        ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=True)
    except Exception:
        ds = load_dataset("squad", split="validation", trust_remote_code=True)

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
