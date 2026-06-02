import statistics
import time

import torch


def _get_device():
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()


def _synchronize():
    """Device-agnostic barrier (no-op on CPU)."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()  # pure memory copy: read once, write once, ~0 FLOPs


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x  # one FMA-style step: 2 FLOPs/element (mul + add)
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    _synchronize() # CUDA: torch.cuda.synchronize(); MPS: torch.mps.synchronize()

    # L2 cache-flush buffer. Re-zeroing it before each timed run evicts whatever
    # the previous run left resident in L2, so every run pays a realistic memory
    # cost instead of getting an artificial cache hit. This matters for the small
    # matmul inputs (a few MB) that otherwise fit entirely in the H100/L40S L2
    # (~50 MB) and would report inflated bandwidth. The 256 MB element-wise input
    # already exceeds L2, so the flush is a no-op effect there. Same approach as
    # triton.testing.do_bench. (256 MB comfortably exceeds any current L2.)
    # https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    cache_flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEVICE)

    if DEVICE.type == "cuda":
        # Time each of `rep` runs with a dedicated pair of CUDA events. Events are
        # recorded on the GPU stream, so elapsed_time() measures pure device time
        # (excluding Python/CPU launch overhead) once we synchronize at the end.
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

        for i in range(rep):
            cache_flush.zero_()  # evict L2 before this run; queued before the start event, so not timed
            start_events[i].record()
            fn(*args)
            end_events[i].record()

        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    else:
        # MPS/CPU: CUDA events are unavailable, so time each run on the host with
        # perf_counter, synchronizing around it so we capture device execution
        # time rather than just async dispatch. This adds a per-run sync, which
        # is acceptable here since the work per run is large relative to the sync.
        times_ms = []
        for _ in range(rep):
            cache_flush.zero_()
            _synchronize() # MPS: torch.mps.synchronize(); CPU: skip
            t0 = time.perf_counter()
            fn(*args)
            _synchronize() # MPS: torch.mps.synchronize(); CPU: skip
            times_ms.append((time.perf_counter() - t0) * 1e3)

    return statistics.median(times_ms)


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # Each `acc = acc * x + x` iteration does 2 FLOPs per element (one multiply,
    # one add), regardless of how PyTorch schedules the work.
    flops_per_element = 2 * num_ops
    total_flops = num_elements * flops_per_element

    if variant == "compiled":
        # Fused kernel: each element is read once and written once at the kernel
        # boundary; intermediates stay in registers. AI = num_ops / bytes_per_el,
        # so it grows linearly with num_ops and the point moves rightward.
        total_bytes = num_elements * 2 * bytes_per_element
    else:  # eager
        # Eager launches a separate multiply and add kernel each iteration. Each
        # binary element-wise op reads 2 operands and writes 1 result (3 tensor
        # accesses), materializing intermediates to global memory. So traffic
        # scales with num_ops just like the FLOPs do, leaving AI roughly
        # constant -> eager points don't move right on the roofline.
        ops_per_iter = 2  # multiply + add
        accesses_per_op = 3  # 2 reads + 1 write
        total_bytes = num_ops * ops_per_iter * accesses_per_op * num_elements * bytes_per_element

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# ----------------------------------------------------------------------------
# ANSWERS  (numbers from the run on an Apple M4: ~4.3 TFLOP/s FP32,
#           120 GB/s memory, ridge around 36 FLOP/Byte)
# ----------------------------------------------------------------------------
#
# A1. These numbers are from my Apple M4 run. (I assume the question is about an
# H100, where this holds from `1 ops` through `64 ops`. On my M4 the flat region
# ends a bit sooner — around `32 ops` — because its real compute ceiling is well
# below the 4.3 TFLOP/s spec, so it turns compute-bound at a lower intensity.)
# The runtime barely moves across that range (it stays around 5.6 ms) because
# the kernel is still just shuffling memory: it reads the 256 MB input once and
# writes 256 MB back, which is about 5.6 ms at the ~95 GB/s measured. The actual
# multiply-adds are basically free here since they happen while we're waiting on
# memory anyway. So adding more ops just packs more FLOPs into the same ~5.6 ms,
# and FLOP/s (FLOPs divided by time) just keeps going up. Nothing got faster,
# there's simply more useful work riding along on the same memory traffic.
#
# A2. A 1024x1024 matmul is small (~2 GFLOP), so on a big GPU it's over almost
# instantly and the fixed costs take over: launching the kernel, leftover tiles
# that don't evenly fill all the cores, and the fact that plain FP32 doesn't
# touch the tensor cores. There just isn't enough work to get the chip up to
# speed. The 128-op kernel, on the other hand, is one big pass over 256 MB that
# keeps everything busy, so it can report a higher FLOP/s. (Fun fact: my M4 run
# was the other way round - matmul 1024 came out ahead - but all the matmuls
# still only hit ~1.4-1.6 TFLOP/s, nowhere near the 4.3 ceiling, for the same
# reasons.)
#
# A3. It means we've stopped being limited by memory and started being limited
# by the math itself. Up to 32 ops time was flat, but 64 -> 128 ops jumps from
# 7 to 15 ms and the throughput drops off a cliff. The data isn't the
# bottleneck anymore, the arithmetic is, so now doubling the ops roughly
# doubles the time. It also doesn't help that each op depends on the previous
# one (acc = acc*x + x), so they can't overlap and we're partly just waiting on
# that chain.
#
# A4. Same math, very different memory behavior. The compiled version fuses the
# whole loop into one kernel, so it reads and writes once and everything in
# between stays in registers - that's why its arithmetic intensity grows with
# the number of ops and the blue points slide to the right. Eager runs each
# multiply and each add as its own kernel and writes the result back to memory
# every time, so the memory traffic grows right along with the FLOPs and the
# intensity never changes (it's stuck around 0.08). That's why the orange
# points (the eager `ops-K`) just stack up in the same spot on the left and the
# runtime climbs steadily (13 ms up to ~2.2 s) - eager never gets off the
# memory ceiling no matter how many ops you throw at it.
