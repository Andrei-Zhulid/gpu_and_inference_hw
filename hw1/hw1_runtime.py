import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GPU_SPECS = {
    "H100": {
        "label": "NVIDIA H100 80GB HBM3",
        "peak_flops": 67e12,  # 67 TFLOP/s FP32, without Tensor Cores
        "peak_bw": 3.35e12,  # 3.35 TB/s HBM3 bandwidth
    },
    "L40S": {
        "label": "NVIDIA L40S 48GB GDDR6",
        "peak_flops": 91.6e12,  # 91.6 TFLOP/s FP32
        "peak_bw": 864e9,  # 864 GB/s GDDR6 bandwidth
    },
    "M4_10GPU": {
        "label": "Apple M4 10-core GPU, 120GB/s unified memory",
        "peak_flops": 4.3e12,  # ~4.3 TFLOP/s FP32, estimated; Apple does not publish this
        "peak_bw": 120e9,  # 120 GB/s unified memory bandwidth
    }
}


def _get_device():
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()


def _get_gpu_specs():
    if DEVICE.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        for gpu_key, specs in GPU_SPECS.items():
            if gpu_key in device_name:
                return specs
        raise RuntimeError(
            f"Unsupported GPU '{device_name}'. Please set the roofline GPU settings yourself."
        )

    if DEVICE.type == "mps":
        # No programmatic spec query on MPS; assume the Apple-silicon entry.
        return GPU_SPECS["M4_10GPU"]

    raise RuntimeError(
        "No CUDA or MPS device available. Please set the GPU roofline settings yourself."
    )


GPU_INFO = _get_gpu_specs()
GPU_LABEL = GPU_INFO["label"]
PEAK_FLOPS = GPU_INFO["peak_flops"]
PEAK_BW = GPU_INFO["peak_bw"]
RIDGE_POINT = PEAK_FLOPS / PEAK_BW


def print_header():
    print("=" * 70)
    print(f"HW1: GPU Roofline Model — {GPU_LABEL}")
    print("=" * 70)
    print("\nTheoretical specs (FP32):")
    print(f"  Peak compute:    {PEAK_FLOPS / 1e12:.0f} TFLOP/s")
    print(f"  Peak bandwidth:  {PEAK_BW / 1e12:.2f} TB/s")
    print(f"  Ridge point:     {RIDGE_POINT:.1f} FLOP/Byte")
    print()


def measure_roofline_points(
    lowest_ai_fn,
    make_compute_fn,
    benchmark_fn,
    compute_elementwise_metrics,
):
    """Run operations with varying arithmetic intensity and measure performance."""
    n = 64 * 1024 * 1024  # 64M elements = 256 MB in float32
    x = torch.randn(n, device=DEVICE, dtype=torch.float32)

    bytes_per_element = 4  # float32
    total_transfer_bytes = n * 2 * bytes_per_element  # read + write

    results = []

    # Benchmark lowest-AI baseline function
    ms = benchmark_fn(lowest_ai_fn, x)
    achieved_bw = total_transfer_bytes / (ms * 1e-3)
    results.append(
        {
            "name": "lowest-AI (~0 FLOPs)",
            "series": "baseline",
            "variant": "baseline",
            "num_ops": 0,
            "arithmetic_intensity": 0.01,
            "achieved_flops": achieved_bw * 0.01,
            "achieved_bw": achieved_bw,
            "ms": ms,
        }
    )
    print(f"  lowest-AI:  {ms:.3f} ms | BW: {achieved_bw / 1e12:.2f} TB/s")

    # Benchmark compute functions with varying arithmetic intensity.
    # We compare eager and compiled versions to show how fusion changes the
    # measured roofline position.
    ops_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for num_ops in ops_list:
        eager_fn = make_compute_fn(num_ops, compiled=False)
        compiled_fn = make_compute_fn(num_ops, compiled=True)

        for variant, compute_fn in [("eager", eager_fn), ("compiled", compiled_fn)]:
            ms = benchmark_fn(compute_fn, x)
            total_flops, ai, achieved_flops = compute_elementwise_metrics(
                num_elements=n,
                num_ops=num_ops,
                bytes_per_element=bytes_per_element,
                ms=ms,
                variant=variant,
            )

            results.append(
                {
                    "name": f"{num_ops} ops",
                    "series": "elementwise",
                    "variant": variant,
                    "num_ops": num_ops,
                    "arithmetic_intensity": ai,
                    "achieved_flops": achieved_flops,
                    "achieved_bw": achieved_flops / ai,
                    "ms": ms,
                }
            )
            print(
                f"  {num_ops:>3d} ops ({variant:>8}): {ms:.3f} ms | "
                f"AI: {ai:.3g} FLOP/B | {achieved_flops / 1e12:.2f} TFLOP/s"
            )

    # Benchmark matrix multiplication (very high arithmetic intensity)
    for m in [1024, 2048, 4096]:
        a = torch.randn(m, m, device=DEVICE, dtype=torch.float32)
        b = torch.randn(m, m, device=DEVICE, dtype=torch.float32)
        fn = lambda a=a, b=b: torch.mm(a, b)
        ms = benchmark_fn(fn, warmup=25, rep=100)

        total_flops = 2 * m * m * m
        total_bytes_mm = (2 * m * m + m * m) * bytes_per_element
        ai = total_flops / total_bytes_mm
        achieved_flops = total_flops / (ms * 1e-3)

        results.append(
            {
                "name": f"n={m}",
                "series": "matmul",
                "variant": "library",
                "num_ops": -1,
                "arithmetic_intensity": ai,
                "achieved_flops": achieved_flops,
                "achieved_bw": total_bytes_mm / (ms * 1e-3),
                "ms": ms,
            }
        )
        print(
            f"  matmul {m}×{m}: {ms:.3f} ms | AI: {ai:.1f} FLOP/B | "
            f"{achieved_flops / 1e12:.2f} TFLOP/s"
        )

    return results


def plot_roofline(results):
    """Create a roofline diagram with theoretical ceilings and measured points."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    ai_range = np.logspace(-2, 4, 500)
    mem_ceiling = PEAK_BW * ai_range
    compute_ceiling = np.full_like(ai_range, PEAK_FLOPS)
    roofline = np.minimum(mem_ceiling, compute_ceiling)

    ax.loglog(ai_range, roofline, "k-", linewidth=2.5, label="Roofline ceiling")
    ax.loglog(
        ai_range,
        mem_ceiling,
        "b--",
        linewidth=1,
        alpha=0.5,
        label=f"Memory BW ceiling ({PEAK_BW / 1e12:.2f} TB/s)",
    )
    ax.loglog(
        ai_range,
        compute_ceiling,
        "r--",
        linewidth=1,
        alpha=0.5,
        label=f"Compute ceiling ({PEAK_FLOPS / 1e12:.0f} TFLOP/s)",
    )

    baseline_results = [r for r in results if r.get("series") == "baseline"]
    eager_results = [r for r in results if r.get("series") == "elementwise" and r.get("variant") == "eager"]
    compiled_results = [
        r for r in results if r.get("series") == "elementwise" and r.get("variant") == "compiled"
    ]
    mm_results = [r for r in results if r.get("series") == "matmul" or r["num_ops"] == -1]

    if baseline_results:
        ais = [r["arithmetic_intensity"] for r in baseline_results]
        flops = [max(r["achieved_flops"], 1e6) for r in baseline_results]
        ax.scatter(
            ais,
            flops,
            c="gray",
            s=80,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Lowest-AI baseline",
        )
        for r in baseline_results:
            f = max(r["achieved_flops"], 1e6)
            label = f"{r['name']}\n{r['ms']:.3f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], f),
                textcoords="offset points",
                xytext=(8, -5),
                fontsize=7,
                color="gray",
            )

    if eager_results:
        ais = [r["arithmetic_intensity"] for r in eager_results]
        flops = [max(r["achieved_flops"], 1e6) for r in eager_results]
        ax.scatter(
            ais,
            flops,
            c="orange",
            s=80,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Element-wise eager (estimated AI)",
        )
        eager_offsets = [(0, 8), (0, -16)]
        for i, r in enumerate(eager_results):
            f = max(r["achieved_flops"], 1e6)
            label = f"{r['num_ops']} ops\n{r['ms']:.2f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], f),
                textcoords="offset points",
                xytext=eager_offsets[i % len(eager_offsets)],
                ha="center",
                fontsize=6,
                color="orange",
            )

    if compiled_results:
        ais = [r["arithmetic_intensity"] for r in compiled_results]
        flops = [max(r["achieved_flops"], 1e6) for r in compiled_results]
        ax.scatter(
            ais,
            flops,
            c="blue",
            s=80,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Element-wise compiled (fused AI)",
        )
        for r in compiled_results:
            f = max(r["achieved_flops"], 1e6)
            label = f"{r['num_ops']} ops\n{r['ms']:.3f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], f),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=7,
                color="blue",
            )

    if mm_results:
        ais = [r["arithmetic_intensity"] for r in mm_results]
        flops = [r["achieved_flops"] for r in mm_results]
        ax.scatter(
            ais,
            flops,
            c="red",
            s=100,
            marker="D",
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
            label="Matrix multiply",
        )
        for r in mm_results:
            label = f"{r['name']}\n{r['ms']:.3f} ms"
            ax.annotate(
                label,
                (r["arithmetic_intensity"], r["achieved_flops"]),
                textcoords="offset points",
                xytext=(8, -5),
                fontsize=7,
                color="red",
            )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Performance (FLOP/s)", fontsize=12)
    ax.set_title(f"Roofline Model — {GPU_LABEL} (FP32)", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e9, 2e14)
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    path = RESULTS_DIR / "roofline.png"
    fig.savefig(path, dpi=150)
    print(f"\nRoofline plot saved to {path}")
    plt.close()


def save_roofline_data(results):
    with open(RESULTS_DIR / "roofline_data.json", "w") as f:
        json.dump(results, f, indent=2)
