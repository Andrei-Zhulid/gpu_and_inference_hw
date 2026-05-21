"""Core scheduler/request data structures for HW3 inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SchedulingPolicy(str, Enum):
    PREFILL_FIRST = "prefill_first"
    DECODE_FIRST = "decode_first"


class RequestStatus(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    DONE = "DONE"


class BatchPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class CacheHandle:
    """
    Returned by CacheManager.match_prefix() on a hit. Blocks are NOT yet
    pinned — the caller must call lock(handle) before using matched_blocks,
    and unlock(handle) when the request finishes.
    """

    num_matched_tokens: int
    matched_blocks: list[int]


@dataclass
class Request:
    request_id: int
    arrival_step: int
    prompt_tokens: list[int]
    max_new_tokens: int

    # Set at runtime by the engine / scheduler
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    num_generated_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    num_preemptions: int = 0
    prefix_tokens_saved: int = 0
    # Locked prefix cache handle — must be unlocked when the request finishes.
    cache_handle: CacheHandle | None = None

    # Lifecycle timestamps (engine step units) used for latency stats.
    first_scheduled_step: int | None = None  # WAITING -> RUNNING admission step.
    first_token_step: int | None = (
        None  # Step when first generated token appears (TTFT).
    )
    finish_step: int | None = None  # Step when request is marked DONE (E2E end).

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def is_prefilling(self) -> bool:
        return self.num_computed_tokens < self.prompt_len

    @property
    def remaining_prefill(self) -> int:
        return max(0, self.prompt_len - self.num_computed_tokens)

    @property
    def is_done(self) -> bool:
        return self.num_generated_tokens >= self.max_new_tokens

    def copy(self) -> Request:
        return Request(
            request_id=self.request_id,
            arrival_step=self.arrival_step,
            prompt_tokens=self.prompt_tokens[:],
            max_new_tokens=self.max_new_tokens,
        )


@dataclass
class Batch:
    """One forward pass: either ALL prefill or ALL decode — never mixed."""

    phase: BatchPhase
    # Prefill is chunked, so each item carries both the request and how many
    # prompt tokens to process this step. Decode is always exactly one token
    # per request, so to_decode only needs Request objects.
    to_prefill: list[tuple[Request, int]] = field(default_factory=list)
    to_decode: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    newly_admitted: list[Request] = field(default_factory=list)

    @property
    def is_prefill(self) -> bool:
        return self.phase == BatchPhase.PREFILL

    @property
    def is_decode(self) -> bool:
        return self.phase == BatchPhase.DECODE


@dataclass
class StepMetrics:
    step: int
    decode_tokens: int = 0
    prefill_tokens: int = 0
    num_running: int = 0
    num_waiting: int = 0
    kv_blocks_used: int = 0
    prefix_tokens_saved: int = 0
