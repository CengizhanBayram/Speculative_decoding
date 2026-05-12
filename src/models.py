import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _make_bnb_config(bits: int) -> BitsAndBytesConfig:
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_draft_model(name: str, device: str, bits: int = 0) -> tuple:
    """
    Load a draft model for speculative decoding.

    bits=0  → float16, no quantization (default; suitable for models ≤ ~2 B params)
    bits=4  → 4-bit NF4 via BitsAndBytes (for larger drafts)

    For Llama / Qwen models, trust_remote_code is enabled automatically.
    """
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(trust_remote_code=True)
    if bits in (4, 8):
        kwargs["quantization_config"] = _make_bnb_config(bits)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


def load_target_model(name: str, bits: int = 0) -> tuple:
    """
    Load the target model.

    bits=0  → float16, no quantization (default; suitable for models up to ~2 B params)
    bits=4  → 4-bit NF4 quantization via BitsAndBytes (for 7 B+ models on limited VRAM)
    bits=8  → 8-bit quantization via BitsAndBytes
    """
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if bits in (4, 8):
        kwargs["quantization_config"] = _make_bnb_config(bits)
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer
