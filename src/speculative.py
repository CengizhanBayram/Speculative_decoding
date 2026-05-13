import time
from typing import Optional

import torch
import pandas as pd
from tqdm import tqdm


def _slice_past_kv(past_kv, length: int):
    """
    Trim a KV-cache to the first `length` token positions.
    Handles both the legacy tuple-of-tuples format (GPT-2) and the
    DynamicCache object returned by Llama / Qwen in transformers >= 4.38.
    """
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            dc = DynamicCache()
            for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                dc.key_cache.append(k[:, :, :length, :])
                dc.value_cache.append(v[:, :, :length, :])
            return dc
    except (ImportError, AttributeError):
        pass
    return tuple(
        (k[:, :, :length, :], v[:, :, :length, :])
        for k, v in past_kv
    )


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
    Speculative decoding with target-side KV cache.

    The target model is initialised once on the full prompt (O(L_prompt) cost)
    and thereafter called only on the γ draft tokens per outer iteration (O(γ)).
    This eliminates the O(L) re-encoding cost of the naive implementation and
    makes latency much less sensitive to generation length.

    Invariant maintained throughout: len(target_past_kv) == generated.shape[1].

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

    eos_ids = set()
    if draft_tok.eos_token_id  is not None: eos_ids.add(draft_tok.eos_token_id)
    if target_tok.eos_token_id is not None: eos_ids.add(target_tok.eos_token_id)

    # ── One-time prompt initialisation ───────────────────────────────────────
    # Builds target KV cache for the prompt and gives the logit for the first
    # draft token (d0) without an extra forward pass.
    with torch.no_grad():
        prompt_out = target_model(generated.to(target_device), use_cache=True)
    target_past_kv    = prompt_out.past_key_values
    last_target_logit = prompt_out.logits[:, -1, :].to(draft_device)

    stop = False
    while (generated.shape[1] - n_input) < max_new_tokens and not stop:
        remaining = max_new_tokens - (generated.shape[1] - n_input)
        gamma     = min(draft_steps, remaining)

        # ── Draft phase (KV-cached) ───────────────────────────────────────────
        draft_tokens     = []
        draft_probs_list = []
        past_key_values  = None

        with torch.no_grad():
            for step in range(gamma):
                model_input = generated.to(draft_device) if step == 0 \
                              else draft_tokens[-1].to(draft_device)
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
        # Logit for d0 comes from last_target_logit (carried from previous iter).
        # Logit for di (i >= 1) is target_out.logits[:, i-1, :].
        n_accepted    = 0
        rejection_idx = None
        corrected_tok = None
        eos_found     = False

        for i in range(gamma):
            token_id = draft_tokens[i].item()
            total_draft_tokens += 1

            t_logit_raw = last_target_logit if i == 0 \
                          else target_out.logits[:, i - 1, :]
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
                    eos_found = True
                    break
            else:
                rejection_idx = i
                adjusted      = torch.clamp(t_prob - d_prob, min=0.0)
                norm          = adjusted.sum()
                if norm > 1e-10:
                    corrected_tok = torch.multinomial(
                        adjusted / norm, num_samples=1
                    ).to(draft_device)
                else:
                    corrected_tok = t_logit.argmax(dim=-1, keepdim=True).to(draft_device)
                break

        # ── Append tokens + update target KV cache ────────────────────────────
        if rejection_idx is not None:
            for j in range(n_accepted):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            generated = torch.cat([generated, corrected_tok], dim=1)

            # Slice to accepted drafts, extend with corrected token.
            kv_sliced = _slice_past_kv(target_out.past_key_values, L + n_accepted)
            with torch.no_grad():
                corr_out = target_model(
                    corrected_tok.to(target_device),
                    past_key_values=kv_sliced,
                    use_cache=True,
                )
            target_past_kv    = corr_out.past_key_values
            last_target_logit = corr_out.logits[:, 0, :].to(draft_device)

            if corrected_tok.item() in eos_ids:
                stop = True

        elif eos_found:
            for j in range(n_accepted):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            target_past_kv = _slice_past_kv(target_out.past_key_values, L + n_accepted)
            stop = True

        else:
            # All γ accepted — bonus token from last output logit.
            for j in range(gamma):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)

            bonus_logit = target_out.logits[:, gamma - 1, :].to(draft_device)
            if temperature == 0.0:
                bonus_token = bonus_logit.argmax(dim=-1, keepdim=True).to(draft_device)
            else:
                bonus_prob  = torch.softmax(bonus_logit / temperature, dim=-1)
                bonus_token = torch.multinomial(bonus_prob, num_samples=1).to(draft_device)
            generated = torch.cat([generated, bonus_token], dim=1)

            # Extend KV with bonus token; carry its logit to next iteration.
            with torch.no_grad():
                bonus_out = target_model(
                    bonus_token.to(target_device),
                    past_key_values=target_out.past_key_values,
                    use_cache=True,
                )
            target_past_kv    = bonus_out.past_key_values
            last_target_logit = bonus_out.logits[:, 0, :].to(draft_device)

            if bonus_token.item() in eos_ids:
                stop = True

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
    mode : "speculative" | "greedy" | "beam"
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
            raise ValueError(f"Unknown mode: {mode!r}")

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
