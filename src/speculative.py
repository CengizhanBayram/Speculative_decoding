import time
from typing import Optional

import torch
import pandas as pd
from tqdm import tqdm


def speculative_decode(
    input_ids: torch.Tensor,
    draft_model,
    draft_tok,
    target_model,
    target_tok,
    draft_steps: int = 5,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> dict:
    """
    Speculative decoding with full accept/reject algorithm.

    Draft model proposes `draft_steps` tokens using a KV-cached forward pass;
    target model verifies them all in a single forward pass. Each token is
    accepted with probability min(1, p_target / p_draft). Rejected positions
    are replaced by a sample from the corrected residual distribution
    max(0, p_target - p_draft). If all draft tokens are accepted a bonus token
    is sampled from the target. Generation stops immediately when any EOS token
    is encountered (including mid-sequence accepted draft tokens).

    Returns
    -------
    dict with keys: generated_text, acceptance_rate, num_target_calls,
                    latency_ms, token_level_log.
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
    target_prefix_past = None   # target KV cache; covers generated[:-1]

    eos_ids = set()
    if draft_tok.eos_token_id is not None:
        eos_ids.add(draft_tok.eos_token_id)
    if target_tok.eos_token_id is not None:
        eos_ids.add(target_tok.eos_token_id)

    stop = False
    while (generated.shape[1] - n_input) < max_new_tokens and not stop:
        remaining = max_new_tokens - (generated.shape[1] - n_input)
        gamma     = min(draft_steps, remaining)

        # ── Draft phase (KV-cached) ───────────────────────────────────────────
        # First call: full context to build KV cache.
        # Subsequent calls: only the last draft token + accumulated KV cache.
        draft_tokens     = []   # List[Tensor[1,1]]
        draft_probs_list = []   # List[Tensor[1,vocab]]
        past_key_values  = None

        with torch.no_grad():
            for step in range(gamma):
                if step == 0:
                    model_input = generated.to(draft_device)
                else:
                    model_input = draft_tokens[-1].to(draft_device)

                out   = draft_model(model_input,
                                    past_key_values=past_key_values,
                                    use_cache=True)
                logit           = out.logits[:, -1, :]
                past_key_values = out.past_key_values

                if temperature == 0.0:
                    prob  = torch.zeros_like(logit)
                    token = logit.argmax(dim=-1, keepdim=True)
                    prob[0, token.item()] = 1.0
                else:
                    prob  = torch.softmax(logit / temperature, dim=-1)
                    token = torch.multinomial(prob, num_samples=1)

                draft_tokens.append(token)
                draft_probs_list.append(prob)

        # ── Verification phase (target KV-cached) ────────────────────────────
        # We maintain `target_prefix_past`, the target's KV cache for
        # generated[:-1].  Each iteration we feed only [generated[-1:] |
        # draft_tokens] (γ+1 tokens) instead of the full sequence, reducing
        # target attention cost from O(L·γ) to O(γ) per outer step.
        #
        # Logit indexing with prefix cache:
        #   out.logits[:, 0, :] ≡ logit at position (L-1) → predicts d1
        #   out.logits[:, i, :] → predicts d_{i+1}
        #   out.logits[:, γ, :] → bonus token
        L = generated.shape[1]

        if target_prefix_past is None:
            # First iteration: full forward pass (no cache yet); use_cache=True
            # so we capture the KV states for subsequent iterations.
            target_input = torch.cat([generated] + draft_tokens, dim=1).to(target_device)
            with torch.no_grad():
                target_out    = target_model(target_input, use_cache=True)
            target_logits_full = target_out.logits          # [1, L+γ, vocab]
            target_prefix_past = target_out.past_key_values # covers 0..(L+γ-1)
            # Wrap logit getter: same indexing as original code
            def _get_logit(i):
                return target_logits_full[:, L - 1 + i, :].to(draft_device)
        else:
            # Subsequent iterations: feed only [generated[-1:] | d1..dγ]
            verify_input = torch.cat(
                [generated[:, -1:]] + draft_tokens, dim=1
            ).to(target_device)                             # [1, 1+γ]
            with torch.no_grad():
                target_out    = target_model(verify_input,
                                             past_key_values=target_prefix_past,
                                             use_cache=True)
            target_logits_new  = target_out.logits          # [1, 1+γ, vocab]
            target_prefix_past = target_out.past_key_values # covers 0..(L+γ-1)
            # Logit at index i → predicts d_{i+1} (same meaning, different array index)
            def _get_logit(i):
                return target_logits_new[:, i, :].to(draft_device)

        num_target_calls += 1

        # ── Accept / reject ───────────────────────────────────────────────────
        n_accepted    = 0
        rejection_idx = None
        corrected_tok = None
        eos_found     = False  # True when an accepted draft token is EOS

        for i in range(gamma):
            token_id = draft_tokens[i].item()
            total_draft_tokens += 1

            t_logit = _get_logit(i)
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

            token_level_log.append({
                "position":        generated.shape[1] - n_input + i,
                "token_id":        token_id,
                "token_str":       draft_tok.decode([token_id]),
                "p_draft":         round(p_draft,        6),
                "p_target":        round(p_target,       6),
                "acceptance_prob": round(acceptance_prob, 6),
                "accepted":        accepted,
            })

            if accepted:
                n_accepted    += 1
                total_accepted += 1
                if token_id in eos_ids:
                    eos_found = True  # stop immediately after appending this token
                    break
            else:
                rejection_idx  = i
                adjusted       = torch.clamp(t_prob - d_prob, min=0.0)
                norm           = adjusted.sum()
                if norm > 1e-10:
                    adjusted      = adjusted / norm
                    corrected_tok = torch.multinomial(adjusted, num_samples=1).to(draft_device)
                else:
                    corrected_tok = t_logit.argmax(dim=-1, keepdim=True).to(draft_device)
                break

        # ── Append tokens and determine whether to stop ───────────────────────
        if rejection_idx is not None:
            for j in range(n_accepted):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            generated = torch.cat([generated, corrected_tok], dim=1)
            if corrected_tok.item() in eos_ids:
                stop = True

        elif eos_found:
            for j in range(n_accepted):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            stop = True

        else:
            # All gamma draft tokens accepted — append them plus a bonus token.
            for j in range(gamma):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            bonus_logit = _get_logit(gamma)   # logit at index γ → bonus
            if temperature == 0.0:
                bonus_token = bonus_logit.argmax(dim=-1, keepdim=True)
            else:
                bonus_prob  = torch.softmax(bonus_logit / temperature, dim=-1)
                bonus_token = torch.multinomial(bonus_prob, num_samples=1)
            generated = torch.cat([generated, bonus_token], dim=1)
            if bonus_token.item() in eos_ids:
                stop = True

        # ── Update target prefix cache ────────────────────────────────────────
        # target_prefix_past now covers 0..(L+γ-1).
        # After appending (n_accepted + 1) tokens, generated has length new_L.
        # We need the cache to cover generated[:-1] = new_L - 1 positions.
        new_L  = generated.shape[1]
        n_keep = new_L - 1
        target_prefix_past = tuple(
            (k[:, :, :n_keep, :], v[:, :, :n_keep, :])
            for k, v in target_prefix_past
        )

    latency_ms      = (time.perf_counter() - start_time) * 1000
    acceptance_rate = total_accepted / max(total_draft_tokens, 1)
    generated_text  = target_tok.decode(
        generated[0, n_input:].tolist(), skip_special_tokens=True
    )

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

    with torch.no_grad():
        output = target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=target_tok.pad_token_id,
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

    with torch.no_grad():
        output = target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=target_tok.pad_token_id,
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
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Run decoding over `samples` and return results as a DataFrame.

    Parameters
    ----------
    mode : "speculative" | "greedy" | "beam"
    draft_steps : number of speculative draft tokens (γ), ignored for greedy/beam.
    temperature : sampling temperature for speculative decoding (0.0 = greedy).
    """
    records = []

    for sample in tqdm(samples, desc=f"[{mode}] γ={draft_steps}"):
        prompt    = sample["prompt"]
        reference = sample["reference"]
        task      = sample["task"]

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
            raise ValueError(f"Unknown mode: {mode!r}. Use 'speculative', 'greedy', or 'beam'.")

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

    return pd.DataFrame(records)
