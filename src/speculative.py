import time
import warnings

import torch
import pandas as pd
from tqdm import tqdm


# ── KV-cache utilities ────────────────────────────────────────────────────────

def _get_dynamic_cache_class():
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache
    return DynamicCache


def _to_dynamic_cache(past_kv):
    """
    Ensure past_kv is a cache object that models can consume.

    Detection uses get_seq_length() — the method transformers >= 4.43 calls
    unconditionally — rather than internal attributes (key_cache / value_cache)
    which changed across versions.  Conversion uses from_legacy_cache(), the
    stable public class-method API available since transformers 4.38.
    """
    if past_kv is None:
        return None
    if hasattr(past_kv, "get_seq_length"):
        return past_kv                      # already a valid Cache object
    DynamicCache = _get_dynamic_cache_class()
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past_kv)
    return past_kv                          # very old transformers: pass through


def _slice_past_kv(past_kv, length: int):
    """
    Trim a KV-cache to the first `length` token positions.

    Primary path: DynamicCache.crop() (available since transformers 4.43,
    same release that mandates get_seq_length — so both always co-exist).
    crop() is in-place but safe here because the caller passes a freshly
    created model output that no other code holds a reference to.

    Fallback path: to_legacy_cache / from_legacy_cache serialization API
    (stable since transformers 4.38) for older installations.

    Returns None if past_kv is None (safe no-op for callers).
    """
    if past_kv is None:
        return None

    # ── Preferred: crop() — available in every version that has get_seq_length
    if hasattr(past_kv, "crop"):
        past_kv.crop(length)
        return past_kv

    DynamicCache = _get_dynamic_cache_class()

    # ── Fallback: extract tensors via serialization API ───────────────────────
    legacy = None
    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()   # → tuple of (key, value) per layer

    if legacy is None:
        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            legacy = tuple(zip(past_kv.key_cache, past_kv.value_cache))
        else:
            legacy = past_kv                 # assume already tuple-of-tuples

    if not legacy:                           # empty cache — nothing to slice
        return past_kv

    sliced = tuple(
        (layer[0][:, :, :length, :], layer[1][:, :, :length, :])
        for layer in legacy
    )

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(sliced)
    return sliced                            # very old transformers fallback


def _sample_token(logit: torch.Tensor, temperature: float):
    """
    Sample one token from logit; return (token_tensor, probability_vector).

    temperature == 0.0 → deterministic argmax (one-hot prob).
    temperature  > 0.0 → softmax sampling.
    """
    if temperature == 0.0:
        token = logit.argmax(dim=-1, keepdim=True)
        prob  = torch.zeros_like(logit)
        prob[0, token.item()] = 1.0
    else:
        prob  = torch.softmax(logit / temperature, dim=-1)
        token = torch.multinomial(prob, num_samples=1)
    return token, prob


# ── Core decoding functions ───────────────────────────────────────────────────

def speculative_decode(
    input_ids: torch.Tensor,
    draft_model,
    draft_tok,
    target_model,
    target_tok,
    draft_steps: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> dict:
    """
    Speculative decoding with KV-cache maintained for BOTH draft and target.

    Both models are initialised once on the full prompt (O(L²) cost, paid once).
    Each outer iteration thereafter:
      - Draft:  gamma-1 single-token forwards  (step-0 reuses last_draft_logit)
      - Target: one γ-token forward
      - Post-accept/reject: 1-2 single-token draft forwards to extend the cache

    This gives O(max_new_tokens · L) total draft work instead of the
    O(max_new_tokens · L² / γ) incurred by re-encoding the full sequence on
    every outer iteration.

    Invariant maintained after each outer iteration:
      draft_past_kv   covers every token in `generated`
      last_draft_logit == distribution for generated.shape[1] (next to draft)

    Parameters
    ----------
    temperature : 0.0 → greedy (default, deterministic, matches greedy baseline);
                  > 0  → softmax sampling from the target distribution.
    """
    start_time = time.perf_counter()

    draft_device  = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    generated = input_ids.to(draft_device).clone()
    n_input   = input_ids.shape[1]

    total_accepted     = 0
    total_draft_tokens = 0
    num_target_calls   = 0
    token_level_log    = []

    eos_ids: set = set()
    for tok in (draft_tok, target_tok):
        eid = tok.eos_token_id
        if eid is None:
            continue
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        else:
            eos_ids.add(int(eid))

    # ── One-time prompt initialisation for BOTH draft and target ─────────────
    with torch.no_grad():
        prompt_tgt = target_model(generated.to(target_device), use_cache=True)
        prompt_dft = draft_model(generated.to(draft_device),   use_cache=True)

    if prompt_tgt.past_key_values is None:
        raise RuntimeError(
            "target_model returned past_key_values=None with use_cache=True. "
            "Ensure the model supports KV caching."
        )

    target_past_kv    = _to_dynamic_cache(prompt_tgt.past_key_values)
    last_target_logit = prompt_tgt.logits[:, -1, :].to(draft_device)
    draft_past_kv     = _to_dynamic_cache(prompt_dft.past_key_values)
    last_draft_logit  = prompt_dft.logits[:, -1, :].to(draft_device)

    stop = False
    while (generated.shape[1] - n_input) < max_new_tokens and not stop:
        remaining = max_new_tokens - (generated.shape[1] - n_input)
        gamma     = min(draft_steps, remaining)

        # ── Draft phase ───────────────────────────────────────────────────────
        # Step 0 uses last_draft_logit (no model call).
        # Steps 1..gamma-1: single-token forwards extending draft_kv_running.
        # After the loop draft_kv_running covers prompt + first (gamma-1) draft
        # tokens; draft_tokens[gamma-1] is NOT yet in the cache.
        draft_tokens     = []
        draft_probs_list = []
        draft_kv_running = draft_past_kv   # local reference; extended in-loop

        with torch.no_grad():
            for step in range(gamma):
                if step == 0:
                    logit = last_draft_logit   # reuse — no model call needed
                else:
                    out = draft_model(
                        draft_tokens[-1].to(draft_device),
                        past_key_values=draft_kv_running,
                        use_cache=True,
                    )
                    logit = out.logits[:, -1, :]
                    draft_kv_running = _to_dynamic_cache(out.past_key_values)

                token, prob = _sample_token(logit, temperature)
                draft_tokens.append(token)
                draft_probs_list.append(prob)

        # ── Target verification — only γ tokens, O(γ) attention ──────────────
        L         = generated.shape[1]
        draft_cat = torch.cat(draft_tokens, dim=1).to(target_device)
        with torch.no_grad():
            target_out = target_model(
                draft_cat,
                past_key_values=target_past_kv,
                use_cache=True,
            )
        num_target_calls += 1

        # ── Accept / reject ───────────────────────────────────────────────────
        n_accepted    = 0
        rejection_idx = None
        corrected_tok = None
        eos_found     = False

        for i in range(gamma):
            token_id = draft_tokens[i].item()
            total_draft_tokens += 1

            t_logit_raw = (
                last_target_logit if i == 0
                else target_out.logits[:, i - 1, :]
            )
            t_logit = t_logit_raw.to(draft_device)
            d_prob  = draft_probs_list[i]

            if temperature == 0.0:
                t_prob = torch.zeros_like(t_logit)
                t_prob[0, t_logit.argmax().item()] = 1.0
            else:
                t_prob = torch.softmax(t_logit / temperature, dim=-1)

            p_target        = t_prob[0, token_id].item()
            p_draft         = d_prob[0, token_id].item()
            acceptance_prob = min(1.0, p_target / max(p_draft, 1e-10))
            accepted        = torch.rand(1, device=draft_device).item() <= acceptance_prob

            try:
                token_str = draft_tok.decode([token_id])
            except Exception:
                token_str = f"<id={token_id}>"

            token_level_log.append({
                "position":        generated.shape[1] - n_input + i,
                "token_id":        token_id,
                "token_str":       token_str,
                "p_draft":         round(p_draft,         6),
                "p_target":        round(p_target,        6),
                "acceptance_prob": round(acceptance_prob, 6),
                "accepted":        accepted,
            })

            if accepted:
                n_accepted    += 1
                total_accepted += 1
                if token_id in eos_ids:
                    eos_found = True
                    break
            else:
                rejection_idx = i
                adjusted = torch.clamp(t_prob - d_prob, min=0.0)
                norm     = adjusted.sum()
                corrected_tok = (
                    torch.multinomial(adjusted / norm, num_samples=1).to(draft_device)
                    if norm > 1e-10
                    else t_logit.argmax(dim=-1, keepdim=True).to(draft_device)
                )
                break

        # ── Append tokens + update BOTH KV caches ────────────────────────────
        if rejection_idx is not None:
            for j in range(n_accepted):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            generated = torch.cat([generated, corrected_tok], dim=1)

            # Target: crop to L+n_accepted, then extend by corrected_tok
            kv_tgt_sliced = _slice_past_kv(target_out.past_key_values, L + n_accepted)
            with torch.no_grad():
                corr_tgt = target_model(
                    corrected_tok.to(target_device),
                    past_key_values=kv_tgt_sliced,
                    use_cache=True,
                )
            target_past_kv    = _to_dynamic_cache(corr_tgt.past_key_values)
            last_target_logit = corr_tgt.logits[:, 0, :].to(draft_device)

            # Draft: crop draft_kv_running to L+n_accepted, extend by corrected_tok
            kv_dft_sliced = _slice_past_kv(draft_kv_running, L + n_accepted)
            with torch.no_grad():
                corr_dft = draft_model(
                    corrected_tok.to(draft_device),
                    past_key_values=kv_dft_sliced,
                    use_cache=True,
                )
            draft_past_kv    = _to_dynamic_cache(corr_dft.past_key_values)
            last_draft_logit = corr_dft.logits[:, 0, :].to(draft_device)

            if corrected_tok.item() in eos_ids:
                stop = True

        elif eos_found:
            for j in range(n_accepted):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            target_past_kv = _slice_past_kv(target_out.past_key_values, L + n_accepted)
            stop = True
            # draft cache not updated — stop=True, loop will not continue

        else:
            # All γ accepted — sample bonus token from target's last logit
            for j in range(gamma):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)

            bonus_logit         = target_out.logits[:, gamma - 1, :].to(draft_device)
            bonus_token, _      = _sample_token(bonus_logit, temperature)
            generated = torch.cat([generated, bonus_token], dim=1)

            # Target: extend cache with bonus_token
            with torch.no_grad():
                bonus_tgt = target_model(
                    bonus_token.to(target_device),
                    past_key_values=_to_dynamic_cache(target_out.past_key_values),
                    use_cache=True,
                )
            target_past_kv    = _to_dynamic_cache(bonus_tgt.past_key_values)
            last_target_logit = bonus_tgt.logits[:, 0, :].to(draft_device)

            # Draft: extend draft_kv_running with draft_tokens[gamma-1], then bonus_token.
            # draft_kv_running covers prompt + draft_tokens[0..gamma-2]; adding
            # draft_tokens[gamma-1] brings it up to prompt + all gamma draft tokens.
            with torch.no_grad():
                ext_dft = draft_model(
                    draft_tokens[gamma - 1].to(draft_device),
                    past_key_values=draft_kv_running,
                    use_cache=True,
                )
            draft_kv_extended = _to_dynamic_cache(ext_dft.past_key_values)
            with torch.no_grad():
                bonus_dft = draft_model(
                    bonus_token.to(draft_device),
                    past_key_values=draft_kv_extended,
                    use_cache=True,
                )
            draft_past_kv    = _to_dynamic_cache(bonus_dft.past_key_values)
            last_draft_logit = bonus_dft.logits[:, 0, :].to(draft_device)

            if bonus_token.item() in eos_ids:
                stop = True

    latency_ms      = (time.perf_counter() - start_time) * 1000
    acceptance_rate = total_accepted / max(total_draft_tokens, 1)
    try:
        generated_text = target_tok.decode(
            generated[0, n_input:].tolist(), skip_special_tokens=True
        )
    except Exception:
        generated_text = ""

    return {
        "generated_text":   generated_text,
        "acceptance_rate":  acceptance_rate,
        "num_target_calls": num_target_calls,
        "latency_ms":       latency_ms,
        "token_level_log":  token_level_log,
    }


def greedy_decode(
    input_ids: torch.Tensor,
    target_model,
    target_tok,
    max_new_tokens: int = 128,
) -> dict:
    """Autoregressive greedy decoding — used as the speed baseline."""
    start_time = time.perf_counter()

    device    = next(target_model.parameters()).device
    input_ids = input_ids.to(device)

    pad_token_id = target_tok.pad_token_id or target_tok.eos_token_id

    with torch.no_grad():
        output = target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=target_tok.eos_token_id,
        )

    latency_ms     = (time.perf_counter() - start_time) * 1000
    new_tokens     = output[0, input_ids.shape[1]:]
    generated_text = target_tok.decode(new_tokens, skip_special_tokens=True)

    return {
        "generated_text":   generated_text,
        "acceptance_rate":  1.0,
        "num_target_calls": len(new_tokens),
        "latency_ms":       latency_ms,
        "token_level_log":  [],
    }


def beam_decode(
    input_ids: torch.Tensor,
    target_model,
    target_tok,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> dict:
    """Beam-search decoding."""
    start_time = time.perf_counter()

    device    = next(target_model.parameters()).device
    input_ids = input_ids.to(device)

    pad_token_id = target_tok.pad_token_id or target_tok.eos_token_id

    with torch.no_grad():
        output = target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=pad_token_id,
            eos_token_id=target_tok.eos_token_id,
        )

    latency_ms     = (time.perf_counter() - start_time) * 1000
    new_tokens     = output[0, input_ids.shape[1]:]
    generated_text = target_tok.decode(new_tokens, skip_special_tokens=True)

    return {
        "generated_text":   generated_text,
        "acceptance_rate":  1.0,
        "num_target_calls": len(new_tokens),
        "latency_ms":       latency_ms,
        "token_level_log":  [],
    }


def run_experiment(
    samples: list,
    draft_model=None,
    draft_tok=None,
    target_model=None,
    target_tok=None,
    mode: str = "speculative",
    draft_steps: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """
    Run decoding over `samples` and return results as a DataFrame.
    mode : "speculative" | "greedy" | "beam"

    temperature : passed to speculative_decode.
                  0.0 (default) → greedy speculative, matches greedy baseline exactly.
                  > 0           → sampling from target distribution.

    Failed samples are skipped with a warning rather than crashing the
    entire run — important for long (500–1000 sample) experiments.
    """
    records  = []
    n_failed = 0

    for sample in tqdm(samples, desc=f"[{mode}] γ={draft_steps}"):
        prompt    = sample["prompt"]
        reference = sample["reference"]
        task      = sample["task"]

        try:
            enc       = target_tok(prompt, return_tensors="pt")
            input_ids = enc["input_ids"]

            if mode == "speculative":
                result = speculative_decode(
                    input_ids=input_ids,
                    draft_model=draft_model,
                    draft_tok=draft_tok,
                    target_model=target_model,
                    target_tok=target_tok,
                    draft_steps=draft_steps,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            elif mode == "greedy":
                result = greedy_decode(
                    input_ids=input_ids,
                    target_model=target_model,
                    target_tok=target_tok,
                    max_new_tokens=max_new_tokens,
                )
            elif mode == "beam":
                result = beam_decode(
                    input_ids=input_ids,
                    target_model=target_model,
                    target_tok=target_tok,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                )
            else:
                raise ValueError(f"Unknown mode: {mode!r}")

        except Exception as exc:
            n_failed += 1
            warnings.warn(f"[{mode}] sample skipped ({type(exc).__name__}: {exc})")
            continue

        records.append({
            "prompt":           prompt,
            "reference":        reference,
            "task":             task,
            "mode":             mode,
            "draft_steps":      draft_steps,
            "generated_text":   result["generated_text"],
            "acceptance_rate":  result["acceptance_rate"],
            "num_target_calls": result["num_target_calls"],
            "latency_ms":       result["latency_ms"],
            "token_level_log":  result["token_level_log"],
        })

    if n_failed:
        warnings.warn(f"[{mode}] {n_failed}/{len(samples)} samples failed and were skipped.")

    return pd.DataFrame(records)
