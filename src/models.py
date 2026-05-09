import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_draft_model(name: str, device: str) -> tuple:
    """Load a small draft model in float16 for speculative decoding."""
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
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

    if bits in (4, 8):
        if bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer
