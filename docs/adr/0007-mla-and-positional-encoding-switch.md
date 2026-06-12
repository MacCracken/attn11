# 0007 — Multi-head latent attention and a positional-encoding switch

**Status**: Accepted
**Date**: 2026-06-12

## Context

M12 (E7) brings **multi-head latent attention** (MLA, DeepSeek-V2,
arXiv:2405.04434) into attn11 as the next step on the shrink-the-KV-cache axis
that M6 opened (E1 = the cache, E2 = GQA). MLA caches one **low-rank latent**
`c_KV` per token (a down-projection `C → d_c`, `d_c ≪ C`) and up-projects to
per-head K **and** V on read, so the cache stores `d_c` numbers per token instead
of `2·Ckv = 2·nkv·hd` — the cached K-row and V-row that the single latent
replaces (the attn11 KV cache sizes both: `model.cyr` allocs `2·T·Ckv` per
layer, and the X002 bench counts `2·NL·T·Ckv·8` bytes). It is additive 1.x
config, not a fork — the frozen surface stays intact (default run byte-identical).

The real decision is **positions**, and it forces a choice:

- attn11 uses learned **absolute** positional embeddings. Per ADR 0005 these are
  *load-bearing*: a cached row's K/V depend on the absolute position it was
  embedded at, which is exactly what makes the context-shift re-prime correct
  (drop oldest `T/2`, recompute at new positions). The whole KV-cache fast path
  rests on absolute positions.
- Faithful DeepSeek MLA uses **decoupled RoPE**: RoPE is relative, and applying
  it inside MLA naively (rotating the up-projected K — "coupled" RoPE) prevents
  absorbing the K up-projection into the query, which forfeits part of MLA's
  cache-compression benefit. DeepSeek's fix carries position on a *separate*
  small dimension (`d_rope`) that bypasses compression, so the latent stays
  absorbable and the cache stays small.

So three positional schemes are in play — learned-abs (have it), coupled RoPE,
decoupled RoPE — and they are *mutually exclusive* (RoPE is relative and replaces
the learned-abs add; you pick one). The user's framing was explicit: ship "what
we have" first and switch later, without churn. The constraint is the frozen-
surface contract: any of this must be opt-in and must not bump the checkpoint
format more than once across the whole ladder.

## Decision

**Make positions a first-class config axis and ship the lowest-risk rung first.**

1. **Two orthogonal config axes**, both opt-in, both defaulting to today's model:
   `--attn-kind {mha, gqa, mla}` (the attention/KV variant) and
   `--pos-kind {learned, rope, rope-decoupled}` (the positional scheme). Default
   `mha` + `learned` reproduces the current transformer bit-for-bit.

2. **Reference MLA keeps learned-abs positions** (`--attn-kind mla
   --pos-kind learned`): the honest, grad-checkable core is the pure low-rank KV
   compression. RoPE is *not* a prerequisite for MLA here. Decoupled RoPE is an
   optional later rung for a faithful DeepSeek reproduction, not part of MLA
   itself.

3. **Checkpoint v4 reserves the architecture descriptor now**, ahead of any
   math: four header fields — `attn_kind`, `pos_kind`, `latent_dim` (`d_c`),
   `rope_dim` (`d_rope`) — default `mha`/`learned`/`0`/`0`. v1/v2/v3 still load
   (synthesizing the defaults). Only the default descriptor is *accepted* until
   each feature fills its field; relaxing each gate is then pure value-fill with
   **no further format bump** across the entire MLA + RoPE ladder. This is the
   cheap forward-compat move that makes "update later" additive.

4. **Staged, each its own gate** (ONE change at a time): (1) descriptor
   scaffolding [this is increment 1, landed ahead of math] → (2) MLA core at
   `pos-kind learned` → (3) coupled `rope` on dense MHA (independent of MLA;
   may split into its own milestone) → (4) decoupled `rope-decoupled` for MLA.

Out of scope for the decision: GEN_SHIFT policy stays `T/2` (ADR 0005); RoPE
base/θ and any length-extrapolation tricks are deferred to whenever rung (3)
lands.

## Consequences

- **Positive** — "start with what we have, update later" is literally the
  default path; no churn to ship MLA. One format bump (v4) covers the whole
  ladder. MLA's backward is just matmul-backward on linear down/up projections
  (rosnet), so it is tractable to grad-check. RoPE, when it lands, has no learned
  params — its grad-check is only the rotation's backward. The pos-kind switch
  also unblocks future relative-position work (length extrapolation) independent
  of MLA.
- **Negative** — the cached/uncached bit-identity gate (note 003) now multiplies
  across `--attn-kind` × `--pos-kind` values: each accepted combination needs its
  own bit-identity coverage. `model_config_ok`, `model_alloc_bytes`, `_blk`/`_o_*`
  offsets, the attention arena, and `ckpt_expected_np` all gain an attn-kind/`d_c`
  branch when MLA's params land — more invariants to keep in sync (the
  `--layers` heap-OOB class of bug). A coupled-RoPE-in-MLA build would be
  numerically valid but cache-inefficient; documented so nobody ships it thinking
  it is the faithful form.
- **Neutral** — v4 reserves a `d_rope` field for decoupled RoPE that may never be
  filled (rung 4 is optional); a zero i64 costs nothing and avoids a *second*
  format bump within the MLA + RoPE ladder (MoE legitimately bumps to v5 later in
  M13 — this only keeps the whole attention/position ladder inside v4). RoPE
  (rung 3) is genuinely valuable on its own and may be pulled forward as its own
  milestone before MLA if relative positions are wanted first.

## Alternatives considered

- **Force RoPE as an MLA prerequisite (faithful DeepSeek only).** Rejected: it
  couples two large changes, discards the load-bearing learned-abs path (ADR
  0005) and the context-shift machinery built on it, and blocks shipping "what we
  have." The low-rank KV compression is the part worth a reference; RoPE is
  separable.
- **One combined `--arch` flag instead of two axes.** Rejected: attention variant
  and positional scheme are genuinely orthogonal (MLA works with learned-abs *or*
  RoPE; RoPE works with MHA *or* MLA). Two axes keep each combination
  expressible and each gate independent.
- **No format change until MLA's params exist; hand-craft v4 buffers in tests.**
  Rejected: reserving the descriptor now (and writing v4 so round-trip exercises
  it) is the explicitly-requested cheap forward-compat, and it keeps the later
  feature increments pure value-fill rather than format bumps.
- **A single opaque `arch_version` integer instead of named fields.** Rejected:
  attn11's header is explicit-field by convention (nkv, tok_kind, Vb, K); named
  descriptor fields validate independently and read clearly, matching the
  existing hostile-input cascade.
- **Coupled RoPE for MLA (simplest RoPE path).** Not rejected outright but
  demoted: numerically valid, but it forfeits the up-projection absorption and
  thus part of MLA's cache win, so it is the wrong *faithful* form — kept only as
  the easy rung-3 stepping stone on dense MHA, with decoupled RoPE the MLA target.
