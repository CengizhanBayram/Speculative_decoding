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

    Draft model proposes `draft_steps` tokens; target model verifies them all
    in a single forward pass. Each token is accepted with probability
    min(1, p_target / p_draft). Rejected positions are replaced by a sample
    from the corrected residual distribution max(0, p_target - p_draft).
    If all draft tokens are accepted a bonus token is sampled from the target.

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
    if draft_tok.eos_token_id is not None:
        eos_ids.add(draft_tok.eos_token_id)
    if target_tok.eos_token_id is not None:
        eos_ids.add(target_tok.eos_token_id)

    while (generated.shape[1] - n_input) < max_new_tokens:
        remaining = max_new_tokens - (generated.shape[1] - n_input)
        gamma     = min(draft_steps, remaining)

        # ── Draft phase ──────────────────────────────────────────────────────
        draft_tokens     = []   # List[Tensor[1,1]]
        draft_probs_list = []   # List[Tensor[1,vocab]]

        draft_input = generated.clone()
        with torch.no_grad():
            for _ in range(gamma):
                logit = draft_model(draft_input).logits[:, -1, :]
                if temperature == 0.0:
                    prob  = torch.zeros_like(logit)
                    token = logit.argmax(dim=-1, keepdim=True)
                    prob[0, token.item()] = 1.0
                else:
                    prob  = torch.softmax(logit / temperature, dim=-1)
                    token = torch.multinomial(prob, num_samples=1)
                draft_tokens.append(token)
                draft_probs_list.append(prob)
                draft_input = torch.cat([draft_input, token], dim=1)

        # ── Verification phase ────────────────────────────────────────────────
        # Feed [generated | draft_tokens] → target sees L+gamma tokens.
        # Logit at position (L-1+i) predicts draft_tokens[i].
        # Logit at position (L-1+gamma) is used for the bonus token.
        L            = generated.shape[1]
        target_input = torch.cat([generated] + draft_tokens, dim=1).to(target_device)
        with torch.no_grad():
            target_logits = target_model(target_input).logits  # [1, L+gamma, vocab]
        num_target_calls += 1

        # ── Accept / reject ───────────────────────────────────────────────────
        n_accepted    = 0
        rejection_idx = None
        corrected_tok = None

        for i in range(gamma):
            token_id = draft_tokens[i].item()
            total_draft_tokens += 1

            t_logit = target_logits[:, L - 1 + i, :].to(draft_device)
            d_prob  = draft_probs_list[i]

            if temperature == 0.0:
                t_prob = torch.zeros_like(t_logit)
                t_prob[0, t_logit.argmax().item()] = 1.0
            else:
                t_prob = torch.softmax(t_logit / temperature, dim=-1)

            p_target         = t_prob[0, token_id].item()
            p_draft          = d_prob[0, token_id].item()
            acceptance_prob  = min(1.0, p_target / max(p_draft, 1e-10))
            accepted         = torch.rand(1).item() <= acceptance_prob

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

        if rejection_idx is not None:
            for j in range(n_accepted):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            generated = torch.cat([generated, corrected_tok], dim=1)
        else:
            for j in range(gamma):
                generated = torch.cat([generated, draft_tokens[j]], dim=1)
            # Bonus token from target at position L-1+gamma (= L+gamma-1, last logit)
            bonus_logit = target_logits[:, L - 1 + gamma, :].to(draft_device)
            if temperature == 0.0:
                bonus_token = bonus_logit.argmax(dim=-1, keepdim=True)
            else:
                bonus_prob  = torch.softmax(bonus_logit / temperature, dim=-1)
                bonus_token = torch.multinomial(bonus_prob, num_samples=1)
            generated = torch.cat([generated, bonus_token], dim=1)

        if generated[0, -1].item() in eos_ids:
            break

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
) -> pd.DataFrame:
    """
    Run decoding over `samples` and return results as a DataFrame.

    Parameters
    ----------
    mode : "speculative" | "greedy" | "beam"
    draft_steps : number of speculative draft tokens (γ), ignored for greedy/beam.
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
                temperature=1.0,
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
