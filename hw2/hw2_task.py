import torch
from torch.profiler import ProfilerActivity, profile as torch_profile
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    DEVICE,
    empty_cache,
    synchronize,
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
    # Run PROFILE_STEPS decode steps under torch.profiler, recording CPU ops
    # (plus CUDA kernels when on CUDA), then print the summary table and export
    # a Chrome trace (open at ui.perfetto.dev) to RESULTS_DIR / trace_name.
    # On MPS/CPU we record CPU activity only — there is no CUDA kernel stream.
    activities = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"

    with torch_profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
        synchronize()

    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace written to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    # Load the model in bfloat16 (the slow baseline uses float32) — L40S has far
    # higher bf16 throughput and bf16 avoids the overflow risk of fp16 here.
    # bf16 is also supported on recent PyTorch MPS, so this works on Mac too.
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
    empty_cache()

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
# I ran this on an Apple M4 (MPS), not the L40S/H100, so the absolute speedup is
# way bigger than the 4x target - the slow baseline is especially rough on MPS.
# The main run in results/ went 26.53s -> 0.86s (about 31x, 4.8 -> 149 tok/s).
#
# The per-fix numbers below come from a separate run, so the baseline and
# optimized times don't match the results/ run exactly - the baseline came out
# at 22.80s and the fully optimized loop at 0.69s that time. Timings wobble a
# bit between runs; the relative contribution of each fix is the point. Each
# rung adds exactly one change:
#
#   rung                              time      this fix    cumulative
#   baseline (no cache, fp32, .item)  22.80s        -          1.00x
#   + inference_mode                  18.26s      1.25x        1.25x
#   + KV cache                         2.78s      6.57x        8.20x
#   + drop per-step .item()            0.88s      3.18x       26.05x
#   + bf16 (= full optimized)          0.69s      1.26x       32.80x
#
# Changes made and speedup per fix:
#
# 1. KV cache - 6.57x, the big one. The baseline ran a full forward over the
#    whole sequence every step, so each new token got more expensive as the
#    sequence grew. Caching the keys and values means the prompt is processed
#    once up front and every later step only does work for the single new token.
#
# 2. Dropped the per-step .item() - 3.18x. The slow loop pulled the new token
#    back to the CPU every iteration, which forces a sync and stalls the
#    pipeline. The v0 trace shows how much this hurts: aten::item
#    (_local_scalar_dense) is 57% of the CPU time. Keeping the token on the GPU
#    and only converting to a Python list once at the end removes all of that.
#
# 3. inference_mode() - 1.25x. I expected this to be tiny, but on the no-cache
#    loop it actually helps a fair bit, because without it every step builds an
#    autograd graph it never uses. (On top of the cache it would matter less.)
#
# 4. bf16 instead of fp32 - 1.26x. Half the memory traffic and faster math.
#    Once the cache is in, attention over the 1024-token prompt (the prefill) is
#    the biggest single cost in the optimized trace, so making that cheaper is
#    where this shows up.
#
# Biggest impact and why:
#
# The KV cache, clearly - 6.57x on its own, more than everything else combined.
# Dropping .item() is a strong second at 3.18x. That lines up with the traces:
# in v0 the time is dominated by the per-step sync (57%) and full-sequence
# attention, and in v1 the only real cost left is the one-time prefill attention
# (scaled_dot_product_attention, ~51%) - exactly what you'd hope to see once the
# per-step waste is gone. The cache is the structural fix that makes decoding
# incremental instead of quadratic; everything else is trimming what's left.