"""Utilities for HW2 — provided, do not modify."""

import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

SEED = 0
MODEL_NAME = "Tiny random Llama (2 layers, d_model=2048)"
VOCAB_SIZE = 4096
PROMPT_LEN = 1024
MAX_NEW_TOKENS = 128
PROFILE_STEPS = 12
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_device():
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def synchronize():
    """Device-agnostic barrier (no-op on CPU)."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def empty_cache():
    """Device-agnostic allocator cache release (no-op on CPU)."""
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def build_model(dtype):
    """Create a tiny decoder-only model with real attention and KV cache."""
    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=PROMPT_LEN + MAX_NEW_TOKENS + 64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.to(device=DEVICE, dtype=dtype)
    model.eval()
    return model


def get_input_ids():
    if DEVICE.type == "cuda":
        generator = torch.Generator(device="cuda")
        generator.manual_seed(SEED)
        return torch.randint(
            low=0,
            high=VOCAB_SIZE,
            size=(1, PROMPT_LEN),
            generator=generator,
            device="cuda",
            dtype=torch.long,
        )
    # MPS/CPU: the MPS RNG generator is limited, so draw on CPU then move.
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(1, PROMPT_LEN),
        generator=generator,
        dtype=torch.long,
    )
    return ids.to(DEVICE)


def slow_loop(model, input_ids, n_steps):
    """Reference slow generation loop — do not modify."""
    generated_ids = input_ids.clone()
    generated_tokens = []
    for _ in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_value = next_token_id.item()
        generated_tokens.append(token_value)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens


def time_generation(loop_fn, model, input_ids, label):
    """Time loop_fn for MAX_NEW_TOKENS with proper GPU synchronization."""
    synchronize()
    start = time.perf_counter()
    generated_tokens = loop_fn(model, input_ids, MAX_NEW_TOKENS)
    synchronize()
    elapsed = time.perf_counter() - start

    preview = generated_tokens[:8]
    print(
        f"{label}: {MAX_NEW_TOKENS} tokens in {elapsed:.2f}s "
        f"({MAX_NEW_TOKENS / elapsed:.1f} tok/s)"
    )
    print(f"Token preview: {preview}")
    return elapsed
