import torch
from torch.profiler import ProfilerActivity, profile as torch_profile
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # Optimizations over slow_loop:
    #   1. KV cache: prefill the prompt once, then feed only the newly generated
    #      token each step. The slow loop reprocessed the entire growing sequence
    #      every step (O(L) work per step -> O(L * n_steps)); with the cache each
    #      decode step is O(1) in sequence length. This is the dominant win.
    #   2. No per-step CPU sync: the slow loop called .item() every iteration,
    #      forcing a device->host sync that stalled the async GPU pipeline. Here
    #      we keep next_token_id on the GPU, feed it straight back, and do a
    #      single host transfer at the very end.
    #   3. torch.inference_mode(): skips autograd bookkeeping entirely (eval()
    #      alone does not).
    # (dtype is handled in generate_optimized, which builds the model in bf16.)
    generated_tokens = []
    with torch.inference_mode():
        # Prefill: one full forward over the prompt, populating the KV cache.
        outputs = model(input_ids=input_ids, use_cache=True)
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        generated_tokens.append(next_token_id)
        past_key_values = outputs.past_key_values

        # Decode: feed only the last token; attention reads the rest from cache.
        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token_id)
            past_key_values = outputs.past_key_values

    # Single device->host sync (one .tolist()) instead of one per step.
    return torch.cat(generated_tokens, dim=1).squeeze(0).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    # Run PROFILE_STEPS decode steps under torch.profiler, recording both CPU
    # ops and CUDA kernels, then print the summary table and export a Chrome
    # trace (open at ui.perfetto.dev) to RESULTS_DIR / trace_name.
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace written to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    # Load the model in bfloat16 (the slow baseline uses float32) — L40S has far
    # higher bf16 throughput and bf16 avoids the overflow risk of fp16 here.
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    return time_generation(optimized_loop, model, input_ids, "Optimized")


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
#
# Biggest impact and why:
#
