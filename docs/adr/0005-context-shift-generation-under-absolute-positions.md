# 0005 — Context-shift generation under absolute positional embeddings

**Status**: Accepted
**Date**: 2026-06-11

## Context

E1 (KV-cached generation) makes the sampler process one row per token by
caching each position's K/V vectors per layer. The cache is only valid while
every cached token keeps its position: attn11 uses **learned absolute
positional embeddings** (GPT-2 style), so a token's K/V rows depend on the
position it was embedded at. The 0.6.0 sampler kept a permanently-full window
and slid it by one token per step — under absolute positions that shifts
*every* token's position *every* step, invalidating the entire cache each
token. A KV cache buys nothing for that sampler.

Real absolute-position models face the same problem; relative schemes (RoPE,
ALiBi) were invented partly to dodge it. The constraint here: attn11's
**training** path is frozen scope for E1 ("training untouched"), so the
positional encoding cannot change. The sampler's semantics can.

## Decision

Generation becomes **prefill → incremental decode → context-shift**:

1. The prompt's last `min(plen, T)` bytes occupy positions `0..n-1` — no
   left-padding (0.6.0 padded short prompts with id 0).
2. Each generated token is appended at position `n` and processed as one
   cached row (`model_fwd_row`).
3. When the window is full (`n == T`), the oldest `GEN_SHIFT() = T/2` tokens
   are dropped and the kept `T/2` tokens are **re-primed** (recomputed at
   their new positions `0..T/2-1`) — one window recompute amortized over the
   next `T/2` generated tokens.

An uncached reference path with **identical semantics** (full forward over
the current window every token — `model_eval_window`) is kept permanently,
because the E1 gate is *bit-identity*: cached and uncached logits must match
bit-for-bit at every step. That gate is what makes the cached fast path
trustworthy (any indexing/staleness bug in the cache shows up as a bit
difference).

## Consequences

- **Positive** — 5.9× tokens/sec at the default config (971 → 5 771); the
  speedup grows with `T` (the re-prime amortizes better). The KV cache is now
  a first-class object, which is what makes E2 (GQA) measurable. The
  bit-identity gate doubles as a regression net for every forward kernel.
- **Negative** — sampler output for a given checkpoint differs from 0.6.0
  (no pad tokens; tokens near a shift see `T/2..T-1` of context instead of a
  always-full window). Worst-case per-token cost is lumpy: most tokens cost
  one row, every `T/2`-th costs a window. Two forward implementations must
  stay arithmetically identical — a constraint on all future kernel changes
  (see `docs/architecture/003`).
- **Neutral** — `GEN_SHIFT() = T/2` is a fixed policy, not config; revisit if
  a future preset (E3, ctx 64) wants a different keep/recompute trade-off.

## Alternatives considered

- **Keep the 0.6.0 always-full sliding window, cache anyway** — rejected: the
  cache is invalidated every token (every position shifts), so the "cached"
  path degenerates to the uncached one. A cache that never hits is worse than
  no cache.
- **Shift cached K/V rows in place on slide** — rejected as *unsound*: K/V
  encode the absolute position they were computed at; reusing them at a new
  position changes the model's function (StreamingLLM documents the resulting
  degradation, arXiv:2309.17453). attn11 is a correctness-first reference —
  silent approximation is out.
- **Switch to RoPE / relative positions to make shifts cache-friendly** —
  rejected for E1: it changes the trained architecture (E1's scope is
  "training untouched") and would invalidate every existing checkpoint. A
  positional-encoding experiment is a legitimate future E-track item.
- **Stop at a full window (no shift)** — rejected: `ngen` ≫ `T` is the normal
  case at ctx 16; a sampler that stops after `T - n` tokens is useless.
- **Shift by 1 (recompute window every token once full)** — rejected: that is
  exactly the uncached cost in the steady state; `T/2` amortizes the
  recompute while keeping at least half a window of context (GPT-2's stride
  evaluation makes the same trade).
