# 0011 — A per-layer mixer hybrid (interleave attention with a cheap mixer)

**Status**: Accepted
**Date**: 2026-06-13

## Context

M14 rung (c) is the survey's structural shift: hybrids that interleave a few
full-attention layers among many cheap recurrent ones (the literature's ~10–25%
attention regime) match or beat pure transformers while cutting the decode cache.
attn11 already has four mixers behind the global `attn_kind` descriptor (mha/mla/
lin/ssm). The rung turns that GLOBAL kind into a PER-LAYER kind, so different
blocks run different sequence mixers, and adds a knob to sweep the attention
fraction at our scale.

attn11's invariants constrain the design. The per-block parameter layout is a
UNIFORM stride: `_blk_base(L) = _emb_end() + L·_blk()`, where `_blk()` is one size
computed from the global descriptor. If layers had different parameter footprints,
every within-block offset (`_o_Wo`, the MLP offsets, …) and `g_NP` would become
per-layer — a pervasive, error-prone refactor of the most delicate addressing code.
The dispatch helpers' own design note (1.4.1) anticipated this: the per-layer
hybrid should "flip `g_attn_kind` to a per-layer lookup HERE and nowhere else."
That is only sound if the mixed kinds SHARE a block layout.

## Decision

**A per-layer mixer kind restricted to LAYOUT-COMPATIBLE mixers — {mha/gqa (0),
gated-linear (2)} — so the per-block stride stays uniform.** Gated linear attention
reuses MHA's exact Q/K/V/O projections (ADR 0009), so `_kvw`, `_blk()`, every
`_o_*` offset, and `g_NP` are identical for kinds {0, 2}. The hybrid is therefore
genuinely dispatch-only:

- `g_layer_kind`: 0 = uniform (every block runs `g_attn_kind`, byte-identical to
  today); else a pointer to NL per-layer kinds. `_lk(L)` returns it, read ONLY by
  the three `_attn_block_*` dispatch helpers.
- `_hybrid_kinds_ok` enforces the **uniform-stride invariant**: every layer's
  `_kvw` must equal the base's, and the base must be a {0,2}-layout kind with
  learned-absolute positions (RoPE is softmax/MLA-specific). Expressed as the
  `_kvw`-equality test, not a hardcoded set — a future layout-compatible kind needs
  no change. This naturally rejects MLA/SSM layers (their `_kvw` differs).
- Decode caches are allocated for whatever kinds appear (the MHA K/V arena is held
  for every layer already; `g_lin_state` is held whenever ANY layer is linear).
  `kv_cache_bytes` SUMS the per-layer caches.
- **Checkpoint v6** carries the per-layer kinds — the first model state that can't
  ride the scalar descriptor. A uniform model still writes v5 (byte-identical);
  only a hybrid writes v6 (NL kinds appended after the fixed header, before the
  vocab). v≤5 synthesize the uniform default.
- CLI `--attn-every K`: a full-attention block at every K-th layer (`L % K == 0`),
  the `--attn-kind` base (mha/lin) elsewhere — the attention-fraction knob.

## Consequences

- **Positive** — per-layer interleaving trains and grad-checks: the MIXED backward
  (`attn_bwd` + `attn_lin_bwd` through one residual stack) lands at ~1e-5
  (`test_model_hybrid`), and cached decode is bit-identical per interleaving
  (`test_kv_hybrid`, X012). The hybrid is PARAMETER-free to switch on (same layout)
  — what it buys is a knob on the decode cache: the attention fraction sets how much
  is T-growing K/V vs constant state (1/3 attention ⇒ half of pure-MHA's cache).
  The 1.4.1 dispatch centralization made the model-side change a per-helper one-liner.
- **Negative** — the hybrid is restricted to {mha, gqa, lin}; MLA and SSM (different
  `_kvw`) cannot yet participate, even though SSM is the best single mixer at our
  scale (X011). Admitting them needs a per-layer (or padded) parameter layout — the
  rung-d follow-on. The first checkpoint format bump (v6); the dual v5/v6 save path
  is mild added surface.
- **Neutral** — at reference scale the ratio sweep is within noise (X012): the
  hybrids edge pure-lin and all beat pure-MHA, but the spread on a tiny repetitive
  corpus is not a scaling claim. The survey's hybrid advantage is a long-context
  phenomenon; the deliverable is the infrastructure to run that sweep at any ratio,
  persisted in the checkpoint.

## Alternatives considered

- **A general per-layer parameter layout (admit any mixer per layer)** — rejected
  for 1.4.3: it makes every `_o_*` offset and `g_NP` per-layer (a prefix-sum of
  heterogeneous block sizes) and touches every `PL(L, …)` call site — the project's
  most delicate addressing code — for no extra value at reference scale. Deferred to
  rung (d), where MLA/SSM hybrids actually need it.
- **Padding every block's K/V region to the max participating `_kvw`** — rejected:
  keeps a uniform stride while admitting any mix, but wastes parameters on the
  smaller-region kinds and muddies the param-count/cache story. Reconsidered if
  rung (d) wants MLA/SSM hybrids without the full per-layer layout.
- **A spare-slot bitmask instead of a v6 format bump** — rejected: the v5 header
  is exactly 22 slots with the vocab immediately after (no reserve), so persisting
  a per-layer pattern genuinely needs more storage. A bitmask would also cap the
  hybrid at two kinds; v6's explicit NL-kind array generalizes to rung (d).
- **Requiring the pattern be re-specified on load (don't persist it)** — rejected:
  a checkpoint must round-trip the full architecture (the project invariant); a
  resume that silently changed the mixer layout would mis-load the weights.
