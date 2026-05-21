from __future__ import annotations

import torch
import torch.nn as nn


class DummyLLM(nn.Module):
    """Tiny CPU transformer block with explicit paged KV reads/writes.

    The KV tensors are stored as one global `[num_blocks, block_size, d_model]`
    buffer and accessed via a per-request `block_table` (see `_paged_indices`,
    `_scatter_kv`, `_gather_kv`) rather than as a contiguous per-request
    `[seq_len, d_model]` tensor. This is the standard paged-KV layout used by
    real production serving systems (vLLM's PagedAttention, SGLang,
    TensorRT-LLM): it lets many requests share the same physical pool, enables
    prefix sharing across requests, and makes preemption cheap. Keeping the
    same layout here is what gives the CacheManager and Scheduler something
    real to schedule — `DummyLLM` owns the physical tensors, the CacheManager
    owns which block IDs each request may touch, and `block_table` is the
    (deliberately thin) interface between them.

    Separate `prefill` and `decode` entry points mirror the same split in
    production engines, where the two phases have very different compute
    shapes and are typically dispatched to different attention kernels.

    Everything else is simplified: a single layer, a single head, CPU only,
    no fused FlashAttention / varlen kernel, no CUDA graph capture, no tensor
    parallelism, no low-precision dtypes, and no real tokenizer or sampler —
    so the scheduler and cache manager stay the only moving parts worth
    studying.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        vocab_size: int = 10_000,
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.scale = d_model**-0.5

        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.ff2 = nn.Linear(d_model * 4, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.register_buffer(
            "k_cache",
            torch.zeros((num_blocks, block_size, d_model), dtype=torch.float32),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros((num_blocks, block_size, d_model), dtype=torch.float32),
        )
        self.eval()

    def _paged_indices(
        self,
        block_table: list[int],
        start_pos: int,
        n_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map logical token positions to physical (block_id, slot) indices."""
        pos = torch.arange(start_pos, start_pos + n_tokens, dtype=torch.long)
        bt = torch.as_tensor(block_table, dtype=torch.long)
        blk_ids = bt[pos // self.block_size]
        slots = pos % self.block_size
        return blk_ids, slots

    def _gather_kv(
        self,
        block_table: list[int],
        n_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_tokens == 0:
            z = torch.zeros((0, self.d_model), dtype=torch.float32)
            return z, z.clone()
        blk_ids, slots = self._paged_indices(
            block_table, start_pos=0, n_tokens=n_tokens
        )
        return self.k_cache[blk_ids, slots], self.v_cache[blk_ids, slots]

    def _scatter_kv(
        self,
        block_table: list[int],
        offset: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        n = k.shape[0]
        if n == 0:
            return
        blk_ids, slots = self._paged_indices(block_table, start_pos=offset, n_tokens=n)
        self.k_cache[blk_ids, slots] = k
        self.v_cache[blk_ids, slots] = v

    @torch.inference_mode()
    def prefill(
        self,
        prompt_tokens: list[int],
        block_table: list[int],
        start_pos: int,
        chunk_len: int,
    ) -> int:
        tokens = torch.tensor(
            prompt_tokens[start_pos : start_pos + chunk_len],
            dtype=torch.long,
        )
        x = self.emb(tokens)

        q = self.q_proj(x)
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)
        self._scatter_kv(block_table, start_pos, k_new, v_new)

        total = start_pos + chunk_len
        k_all, v_all = self._gather_kv(block_table, total)

        scores = (q @ k_all.T) * self.scale
        q_abs = torch.arange(start_pos, start_pos + chunk_len, dtype=torch.long)
        kv_abs = torch.arange(total, dtype=torch.long)
        causal_mask = kv_abs.unsqueeze(0) > q_abs.unsqueeze(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_out = torch.softmax(scores, dim=-1) @ v_all
        x = self.o_proj(attn_out)
        x = x + self.ff2(torch.relu(self.ff1(x)))

        logits = self.lm_head(x[-1])
        return int(torch.argmax(logits).item())

    @torch.inference_mode()
    def decode(
        self,
        token_id: int,
        block_table: list[int],
        pos: int,
    ) -> int:
        token = torch.tensor([token_id], dtype=torch.long)
        x = self.emb(token)

        q = self.q_proj(x)
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)
        self._scatter_kv(block_table, pos, k_new, v_new)

        k_all, v_all = self._gather_kv(block_table, pos + 1)
        scores = (q @ k_all.T) * self.scale
        attn_out = torch.softmax(scores, dim=-1) @ v_all

        x = self.o_proj(attn_out)
        x = x + self.ff2(torch.relu(self.ff1(x)))

        logits = self.lm_head(x[0])
        return int(torch.argmax(logits).item())
