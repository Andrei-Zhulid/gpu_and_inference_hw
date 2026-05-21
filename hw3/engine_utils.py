"""Utilities for HW3 — provided, do not modify."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import matplotlib

try:
    from dummy_llm import DummyLLM
except ModuleNotFoundError:
    # Allows imports like `from hw3_inference_engine.engine_utils import ...`
    # when running from repository root.
    pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from structs import (
        CacheHandle,
        Request,
        Batch,
        BatchPhase,
        StepMetrics,
        SchedulingPolicy,
        RequestStatus,
    )
except ModuleNotFoundError:
    from hw3_inference_engine.structs import (
        Request,
        StepMetrics,
    )

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Engine constants
BLOCK_SIZE = 16  # tokens per KV-cache block
NUM_BLOCKS = 512  # total physical KV-cache blocks
MAX_SEQS = 64  # maximum concurrent sequences
TOKEN_BUDGET = 2048  # token compute budget per step (prefill + decode)
PREFILL_CHUNK = 256  # maximum tokens per prefill chunk

# Plot palette — kept consistent across both result figures.
# "Cache ON" (in the caching-impact plot) and "Prefill-first" (in the policy
# plot) refer to the same run configuration, so they share a color.
COLOR_NO_CACHE = "steelblue"
COLOR_CACHE_ON = "coral"
COLOR_PREFILL_FIRST = COLOR_CACHE_ON
COLOR_DECODE_FIRST = "seagreen"


# ── Workload generation ───────────────────────────────────────────────────────


def generate_workload(
    num_requests: int = 80,
    prompt_len_range: tuple = (32, 200),
    output_len_range: tuple = (50, 300),
    arrival_rate: float = 1.5,
    shared_prefix_len: int = 128,
    shared_prefix_fraction: float = 0.8,
    seed: int = 42,
) -> list[Request]:
    """
    Synthetic workload: `shared_prefix_fraction` of requests start with the
    same `shared_prefix_len`-token prefix (simulating a shared system prompt).
    """
    rng = random.Random(seed)
    shared_prefix = list(range(9_000, 9_000 + shared_prefix_len))

    requests, step = [], 0
    for i in range(num_requests):
        out_len = rng.randint(*output_len_range)
        if rng.random() < shared_prefix_fraction:
            unique_len = max(1, rng.randint(*prompt_len_range) - shared_prefix_len)
            tokens = shared_prefix + [rng.randint(0, 8_999) for _ in range(unique_len)]
        else:
            tokens = [
                rng.randint(0, 8_999) for _ in range(rng.randint(*prompt_len_range))
            ]
        requests.append(Request(i, step, tokens, out_len))
        step += max(1, int(rng.expovariate(arrival_rate)))
    return requests


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_stats(
    finished: list[Request], metrics: list[StepMetrics], total_steps: int
) -> dict:
    ttfts = [
        r.first_token_step - r.arrival_step
        for r in finished
        if r.first_token_step is not None
    ]
    e2es = [
        r.finish_step - r.arrival_step for r in finished if r.finish_step is not None
    ]
    total_gen = sum(r.num_generated_tokens for r in finished)
    prefix_saved = sum(s.prefix_tokens_saved for s in metrics)
    total_prefill = sum(s.prefill_tokens for s in metrics)
    return {
        "total_steps": total_steps,
        "finished": len(finished),
        "throughput": round(total_gen / total_steps, 2) if total_steps else 0,
        "ttft_mean": round(np.mean(ttfts), 1) if ttfts else 0,
        "ttft_p95": round(np.percentile(ttfts, 95), 1) if ttfts else 0,
        "e2e_mean": round(np.mean(e2es), 1) if e2es else 0,
        "preemptions": sum(r.num_preemptions for r in finished),
        "prefix_saved": prefix_saved,
        "cache_hit_rate": round(prefix_saved / (prefix_saved + total_prefill), 3)
        if (prefix_saved + total_prefill)
        else 0,
    }


def print_stats(label: str, s: dict) -> None:
    print(f"\n  {label}")
    print(f"    Steps                    : {s['total_steps']}")
    print(f"    Throughput (tok/step)    : {s['throughput']}")
    print(f"    TTFT mean / p95 (steps)  : {s['ttft_mean']} / {s['ttft_p95']}")
    print(f"    E2E latency mean (steps) : {s['e2e_mean']}")
    print(f"    Preemptions              : {s['preemptions']}")
    print(
        f"    Prefix cache hit rate    : {s['cache_hit_rate']:.1%}  ({s['prefix_saved']} tokens saved)"
    )


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_results(
    workloads: list[tuple[str, list[StepMetrics], list[StepMetrics], dict, dict]],
) -> None:
    """Plot results for one or more workloads.

    workloads: list of (label, no_cache_metrics, cache_metrics,
                        no_cache_stats, cache_stats)
    """
    n = len(workloads)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5 * n), squeeze=False)
    fig.suptitle(
        "HW3: Mini Inference Engine — Prefix Caching Impact",
        fontsize=14,
        y=0.99,
    )

    for row, (wl_label, nc_met, c_met, nc_st, c_st) in enumerate(workloads):
        runs = [("No cache", nc_met, COLOR_NO_CACHE), ("Cache ON", c_met, COLOR_CACHE_ON)]

        # KV memory pressure
        ax = axes[row, 0]
        for rlabel, ms, color in runs:
            ax.plot(
                [s.step for s in ms],
                [s.kv_blocks_used for s in ms],
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=rlabel,
            )
        ax.axhline(NUM_BLOCKS, color="red", ls="--", alpha=0.4, label="Capacity")
        ax.set(
            xlabel="Time step",
            ylabel="KV Blocks Used",
            title=f"{wl_label} — KV Memory Pressure",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Cumulative prefill tokens (directly shows cache compute savings)
        ax = axes[row, 1]
        for rlabel, ms, color in runs:
            c_prefill = np.cumsum([s.prefill_tokens for s in ms])
            ax.plot(
                [s.step for s in ms],
                c_prefill,
                color=color,
                linewidth=1.8,
                alpha=0.9,
                label=rlabel,
            )
        ax.set(
            xlabel="Time step",
            ylabel="Cumulative Prefill Tokens",
            title=f"{wl_label} — Prefill Compute",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Summary bars
        ax = axes[row, 2]
        keys = ["throughput", "ttft_mean", "e2e_mean", "preemptions"]
        bar_labs = [
            "Throughput\n(tok/step)",
            "Avg TTFT\n(steps)",
            "Avg E2E\n(steps)",
            "Preemptions",
        ]
        x, w = np.arange(len(keys)), 0.35
        ax.bar(
            x - w / 2,
            [nc_st[k] for k in keys],
            w,
            label="No cache",
            color=COLOR_NO_CACHE,
            alpha=0.85,
        )
        ax.bar(
            x + w / 2,
            [c_st[k] for k in keys],
            w,
            label="Cache ON",
            color=COLOR_CACHE_ON,
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labs, fontsize=9)
        ax.set(title=f"{wl_label} — Summary")
        ax.text(
            0.98,
            0.98,
            f"Cache hit rate: {c_st['cache_hit_rate']:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="0.8"
            ),
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out_path = RESULTS_DIR / "hw3_results.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plot → {out_path}")


def plot_policy_results(
    policy_results: list[tuple[str, list[StepMetrics], list[StepMetrics], dict, dict]],
) -> None:
    """Compare prefill-first vs decode-first with prefix cache enabled."""
    n = len(policy_results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5 * n), squeeze=False)
    fig.suptitle(
        "HW3: Scheduling Policy Impact (Prefix Cache ON)",
        fontsize=14,
        y=0.99,
    )

    for row, (wl_label, pf_met, df_met, pf_st, df_st) in enumerate(policy_results):
        runs = [
            ("Prefill-first", pf_met, COLOR_PREFILL_FIRST),
            ("Decode-first", df_met, COLOR_DECODE_FIRST),
        ]

        ax = axes[row, 0]
        for rlabel, ms, color in runs:
            ax.plot(
                [s.step for s in ms],
                [s.kv_blocks_used for s in ms],
                color=color,
                linewidth=1.5,
                alpha=0.9,
                label=rlabel,
            )
        ax.axhline(NUM_BLOCKS, color="red", ls="--", alpha=0.4, label="Capacity")
        ax.set(
            xlabel="Time step",
            ylabel="KV Blocks Used",
            title=f"{wl_label} — KV Memory Pressure",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = axes[row, 1]
        for rlabel, ms, color in runs:
            c_decode = np.cumsum([s.decode_tokens for s in ms])
            ax.plot(
                [s.step for s in ms],
                c_decode,
                color=color,
                linewidth=1.8,
                alpha=0.9,
                label=rlabel,
            )
        ax.set(
            xlabel="Time step",
            ylabel="Cumulative Decode Tokens",
            title=f"{wl_label} — Decode Progress",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = axes[row, 2]
        keys = ["throughput", "ttft_mean", "e2e_mean", "preemptions"]
        bar_labs = [
            "Throughput\n(tok/step)",
            "Avg TTFT\n(steps)",
            "Avg E2E\n(steps)",
            "Preemptions",
        ]
        x, w = np.arange(len(keys)), 0.35
        ax.bar(
            x - w / 2,
            [pf_st[k] for k in keys],
            w,
            label="Prefill-first",
            color=COLOR_PREFILL_FIRST,
            alpha=0.85,
        )
        ax.bar(
            x + w / 2,
            [df_st[k] for k in keys],
            w,
            label="Decode-first",
            color=COLOR_DECODE_FIRST,
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labs, fontsize=9)
        ax.set(title=f"{wl_label} — Policy Summary")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out_path = Path(__file__).parent / "results" / "hw3_policy_results.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Policy plot → {out_path}")
