import os
import time
import warnings

import torch
import pandas as pd
from tqdm import tqdm

_RESULT_COLS = [
    "prompt", "reference", "task", "mode", "draft_steps",
    "generated_text", "acceptance_rate", "num_target_calls",
    "latency_ms", "token_level_log",
]


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

    if hasattr(past_kv, "crop"):
        past_kv.crop(length)
        return past_kv

    DynamicCache = _get_dynamic_cache_class()

    legacy = None
    if hasattr(past_kv, "to_legacy_cache"):
        legacy = past_kv.to_legacy_cache()

    if legacy is None:
        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            legacy = tuple(zip(past_kv.key_cache, past_kv.value_cache))
        else:
            legacy = past_kv

    if not legacy:
        return past_kv

    sliced = tuple(
        (layer[0][:, :, :length, :], layer[1][:, :, :length, :])
        for layer in legacy
    )

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(sliced)
    return sliced


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

    Performance optimisations over a naive reference:

    1. Vectorised accept/reject — all γ acceptance probabilities are computed
       in a single GPU batch.  Random draws for the stochastic step are also
       batched (one torch.rand(γ) call instead of γ torch.rand(1) calls),
       eliminating γ sequential GPU→CPU synchronisations per outer iteration.

    2. Merged draft-cache extension (all-accepted branch) — the last draft
       token and the bonus token are fed to the draft model in a single
       two-token forward pass instead of two sequential one-token passes,
       saving one draft model call per successful iteration.

    3. Pre-allocated output buffer — tokens are written into a fixed-size
       tensor instead of growing the sequence with torch.cat on every step,
       removing O(T) tensor allocations over the full generation loop.

    Both models are initialised once on the full prompt (O(L) cost, paid once).
    The target model is then called only with the γ draft tokens per iteration,
    giving O(γ) attention cost instead of O(L) for a naive re-encoding approach.

    Invariants maintained after each outer iteration:
      draft_past_kv    covers every token in the output buffer up to n_gen
      last_draft_logit == distribution for position n_gen (next token to draft)
      target_past_kv   covers every token in the output buffer up to n_gen
      last_target_logit == distribution for position n_gen

    Parameters
    ----------
    temperature : 0.0 → greedy draft proposals; accept/reject is fully
                  deterministic (accept iff draft argmax == target argmax),
                  so acceptance rate σ = 0.000 across seeds.
                  > 0 → softmax sampling from both draft and target; stochastic
                  accept/reject with U ~ Uniform(0, 1) per token.
    """
    start_time = time.perf_counter()

    draft_device  = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    n_input = input_ids.shape[1]

    # Pre-allocated output buffer — avoids O(T) torch.cat allocations
    buf = torch.zeros(
        1, n_input + max_new_tokens + 1,
        dtype=input_ids.dtype, device=draft_device,
    )
    buf[0, :n_input] = input_ids[0]
    n_gen = n_input   # current write position (exclusive end)

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

    # ── One-time prompt initialisation for BOTH models ────────────────────────
    with torch.no_grad():
        prompt_tgt = target_model(input_ids.to(target_device), use_cache=True)
        prompt_dft = draft_model(input_ids.to(draft_device),   use_cache=True)

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
    while (n_gen - n_input) < max_new_tokens and not stop:
        remaining = max_new_tokens - (n_gen - n_input)
        gamma     = min(draft_steps, remaining)

        # ── Draft phase ───────────────────────────────────────────────────────
        # Step 0: reuse last_draft_logit — no model call.
        # Steps 1..γ-1: single-token KV-cached forwards.
        # After the loop, draft_kv_running covers n_gen + (γ-1) positions;
        # draft_tokens[γ-1] has NOT been fed through the draft model yet.
        draft_tokens     = []
        draft_probs_list = []
        draft_kv_running = draft_past_kv

        with torch.no_grad():
            for step in range(gamma):
                logit = last_draft_logit if step == 0 else out.logits[:, -1, :]
                token, prob = _sample_token(logit, temperature)
                draft_tokens.append(token)
                draft_probs_list.append(prob)
                if step < gamma - 1:   # extend KV cache for steps 0..γ-2
                    out = draft_model(
                        token.to(draft_device),
                        past_key_values=draft_kv_running,
                        use_cache=True,
                    )
                    draft_kv_running = _to_dynamic_cache(out.past_key_values)

        # ── Target verification — one call, γ tokens in parallel ──────────────
        draft_cat = torch.cat(draft_tokens, dim=1).to(target_device)
        with torch.no_grad():
            target_out = target_model(
                draft_cat,
                past_key_values=target_past_kv,
                use_cache=True,
            )
        num_target_calls += 1

        # ── Vectorised accept / reject ────────────────────────────────────────
        # Verification logit for draft_tokens[i]:
        #   i == 0 → last_target_logit  (target's prediction before draft_tokens[0])
        #   i  > 0 → target_out.logits[:, i-1, :]
        draft_ids = torch.cat(draft_tokens, dim=1)[0].to(draft_device)    # [γ]
        d_probs   = torch.cat(draft_probs_list, dim=0)                    # [γ, vocab]

        all_t_logits = torch.cat(
            [last_target_logit.unsqueeze(1),
             target_out.logits[:, :-1, :].to(draft_device)],
            dim=1,
        )[0]  # [γ, vocab]

        p_draft_v = d_probs[torch.arange(gamma, device=draft_device), draft_ids]  # [γ]

        if temperature == 0.0:
            # T=0: accept iff draft argmax == target argmax
            t_argmax         = all_t_logits.argmax(dim=-1)            # [γ]
            accept_mask      = (draft_ids == t_argmax)                # [γ] bool
            # One-hot probabilities for logging
            p_target_v       = accept_mask.float()
            acceptance_probs_v = accept_mask.float()
        else:
            t_probs          = torch.softmax(all_t_logits / temperature, dim=-1)  # [γ, vocab]
            p_target_v       = t_probs[torch.arange(gamma, device=draft_device), draft_ids]
            acceptance_probs_v = torch.clamp(
                p_target_v / p_draft_v.clamp(min=1e-10), max=1.0,
            )
            # Single batch of randoms — one GPU-CPU sync, not γ
            randoms     = torch.rand(gamma, device=draft_device)
            accept_mask = randoms <= acceptance_probs_v               # [γ] bool

        # First rejection position (None = all accepted)
        rej_pos = (~accept_mask).nonzero(as_tuple=True)[0]
        if len(rej_pos) == 0:
            rejection_idx = None
            n_accepted    = gamma
        else:
            rejection_idx = int(rej_pos[0].item())
            n_accepted    = rejection_idx

        total_draft_tokens += gamma
        total_accepted     += n_accepted

        # EOS within accepted tokens → truncate and stop
        accepted_ids_list = draft_ids[:n_accepted].tolist()
        eos_pos = next(
            (i for i, tid in enumerate(accepted_ids_list) if tid in eos_ids), None
        )
        if eos_pos is not None:
            n_accepted    = eos_pos + 1
            rejection_idx = None
            eos_found     = True
        else:
            eos_found = False

        # Token-level logging — all CPU operations, no GPU syncs
        log_count  = n_accepted + (1 if rejection_idx is not None else 0)
        ids_cpu    = draft_ids[:log_count].tolist()
        pd_cpu     = p_draft_v[:log_count].tolist()
        pt_cpu     = p_target_v[:log_count].tolist()
        ap_cpu     = acceptance_probs_v[:log_count].tolist()
        acc_cpu    = accept_mask[:log_count].tolist()
        for i in range(log_count):
            try:    token_str = draft_tok.decode([ids_cpu[i]])
            except: token_str = f"<id={ids_cpu[i]}>"
            token_level_log.append({
                "position":        n_gen - n_input + i,
                "token_id":        ids_cpu[i],
                "token_str":       token_str,
                "p_draft":         round(pd_cpu[i], 6),
                "p_target":        round(pt_cpu[i], 6),
                "acceptance_prob": round(ap_cpu[i], 6),
                "accepted":        bool(acc_cpu[i]),
            })

        # ── Update buffer and KV caches ───────────────────────────────────────
        n_gen_before = n_gen   # snapshot for KV-slice lengths

        if rejection_idx is not None:
            # Write accepted tokens then corrected replacement
            if n_accepted > 0:
                buf[0, n_gen:n_gen + n_accepted] = draft_ids[:n_accepted]
                n_gen += n_accepted

            # Corrected token from residual distribution max(0, p_target - p_draft)
            rej_t_logit = all_t_logits[rejection_idx]
            rej_d_prob  = d_probs[rejection_idx]
            if temperature == 0.0:
                corrected_tok = rej_t_logit.argmax(keepdim=True).unsqueeze(0)
            else:
                rej_t_prob = torch.softmax(rej_t_logit / temperature, dim=-1)
                adjusted   = torch.clamp(rej_t_prob - rej_d_prob, min=0.0)
                norm       = adjusted.sum()
                corrected_tok = (
                    torch.multinomial(adjusted / norm, num_samples=1).unsqueeze(0)
                    if norm > 1e-10
                    else rej_t_logit.argmax(keepdim=True).unsqueeze(0)
                )
            buf[0, n_gen] = corrected_tok[0, 0]
            n_gen += 1

            # Target KV: crop to n_gen_before + n_accepted, then extend with corrected_tok
            kv_tgt = _slice_past_kv(target_out.past_key_values, n_gen_before + n_accepted)
            with torch.no_grad():
                corr_tgt = target_model(
                    corrected_tok.to(target_device),
                    past_key_values=kv_tgt,
                    use_cache=True,
                )
            target_past_kv    = _to_dynamic_cache(corr_tgt.past_key_values)
            last_target_logit = corr_tgt.logits[:, 0, :].to(draft_device)

            # Draft KV: crop to n_gen_before + n_accepted, then extend with corrected_tok
            kv_dft = _slice_past_kv(draft_kv_running, n_gen_before + n_accepted)
            with torch.no_grad():
                corr_dft = draft_model(
                    corrected_tok.to(draft_device),
                    past_key_values=kv_dft,
                    use_cache=True,
                )
            draft_past_kv    = _to_dynamic_cache(corr_dft.past_key_values)
            last_draft_logit = corr_dft.logits[:, 0, :].to(draft_device)

            if int(corrected_tok[0, 0].item()) in eos_ids:
                stop = True

        elif eos_found:
            # EOS within accepted tokens — write and stop; no bonus
            buf[0, n_gen:n_gen + n_accepted] = draft_ids[:n_accepted]
            n_gen += n_accepted
            target_past_kv = _slice_past_kv(target_out.past_key_values, n_gen)
            stop = True

        else:
            # All γ accepted — write, sample bonus token from target's last logit
            buf[0, n_gen:n_gen + gamma] = draft_ids[:gamma]
            n_gen += gamma

            bonus_logit    = target_out.logits[:, gamma - 1, :].to(draft_device)
            bonus_token, _ = _sample_token(bonus_logit, temperature)
            buf[0, n_gen]  = bonus_token[0, 0]
            n_gen         += 1

            # Target KV: extend with bonus_token (1 call)
            with torch.no_grad():
                bonus_tgt = target_model(
                    bonus_token.to(target_device),
                    past_key_values=_to_dynamic_cache(target_out.past_key_values),
                    use_cache=True,
                )
            target_past_kv    = _to_dynamic_cache(bonus_tgt.past_key_values)
            last_target_logit = bonus_tgt.logits[:, 0, :].to(draft_device)

            # Draft KV: merged 2-token pass — draft_tokens[γ-1] + bonus_token in one call.
            # draft_kv_running covers n_gen_before + (γ-1) positions.  Feeding both tokens
            # extends it by 2, reaching n_gen_before + γ + 1 = n_gen.
            ext_bonus = torch.cat(
                [draft_tokens[gamma - 1].to(draft_device),
                 bonus_token.to(draft_device)],
                dim=1,
            )
            with torch.no_grad():
                merged_dft = draft_model(
                    ext_bonus,
                    past_key_values=draft_kv_running,
                    use_cache=True,
                )
            draft_past_kv    = _to_dynamic_cache(merged_dft.past_key_values)
            last_draft_logit = merged_dft.logits[:, -1, :].to(draft_device)

            if int(bonus_token[0, 0].item()) in eos_ids:
                stop = True

    latency_ms      = (time.perf_counter() - start_time) * 1000
    acceptance_rate = total_accepted / max(total_draft_tokens, 1)
    try:
        generated_text = target_tok.decode(
            buf[0, n_input:n_gen].tolist(), skip_special_tokens=True
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
    temperature: float = 0.0,
) -> dict:
    """
    Autoregressive decoding — used as the speed baseline.
    temperature == 0.0 → deterministic greedy (do_sample=False).
    temperature  > 0.0 → temperature sampling (do_sample=True), matching the
                         stochastic speculative decoding condition for a fair
                         T > 0 latency comparison.
    """
    start_time = time.perf_counter()

    device    = next(target_model.parameters()).device
    input_ids = input_ids.to(device)

    pad_token_id = target_tok.pad_token_id or target_tok.eos_token_id

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=target_tok.eos_token_id,
    )
    if temperature == 0.0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"]    = True
        generate_kwargs["temperature"]  = temperature

    with torch.no_grad():
        output = target_model.generate(input_ids, **generate_kwargs)

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
    checkpoint_path=None,
) -> pd.DataFrame:
    """
    Run decoding over `samples` and return results as a DataFrame.
    mode : "speculative" | "greedy" | "beam"

    temperature     : 0.0 → greedy draft proposals; accept/reject is fully
                      deterministic (accept iff draft argmax == target argmax),
                      σ = 0.000 across seeds.  > 0 → stochastic sampling from
                      both draft and target; seeds produce meaningful variance.
    checkpoint_path : if given, each completed sample is appended to this CSV so
                      results are not lost if the kernel crashes mid-run.

    Failed samples are skipped with a warning rather than crashing the
    entire run — important for long (500–1000 sample) experiments.
    """
    records  = []
    n_failed = 0

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

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

        row = {
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
        }
        records.append(row)

        if checkpoint_path is not None:
            write_header = not os.path.exists(checkpoint_path)
            pd.DataFrame([row]).to_csv(
                checkpoint_path, mode="a", header=write_header, index=False,
            )

    if n_failed:
        warnings.warn(f"[{mode}] {n_failed}/{len(samples)} samples failed and were skipped.")

    if not records:
        return pd.DataFrame(columns=_RESULT_COLS)
    return pd.DataFrame(records)
