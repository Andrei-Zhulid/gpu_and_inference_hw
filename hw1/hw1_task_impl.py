import statistics

import torch


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
    torch.cuda.synchronize()

    # L2 cache-flush buffer. Re-zeroing it before each timed run evicts whatever
    # the previous run left resident in L2, so every run pays a realistic memory
    # cost instead of getting an artificial cache hit. This matters for the small
    # matmul inputs (a few MB) that otherwise fit entirely in the H100/L40S L2
    # (~50 MB) and would report inflated bandwidth. The 256 MB element-wise input
    # already exceeds L2, so the flush is a no-op effect there. Same approach as
    # triton.testing.do_bench. (256 MB comfortably exceeds any current L2.)
    # https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
    cache_flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")

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
