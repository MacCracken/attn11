# 0012 — Padded uniform stride for any-mixer hybrids

**Status**: Accepted
**Date**: 2026-06-13

## Context

ADR 0011 (M14 rung c, v1.4.3) shipped a per-layer mixer hybrid restricted to
LAYOUT-COMPATIBLE kinds `{mha, gqa, lin}` — they share the exact K/V weight region
(`_kvw`), so the per-block stride stayed uniform with no per-layer offset refactor.
But the strongest pairing the survey points at — full attention interleaved with an
SSM — was excluded: MLA's `_kvw` (`3·C·d_c`) and the SSM's (`3·C·N + C`) differ from
MHA's (`2·C·Ckv`), so a hybrid containing them would have a non-uniform per-block
stride. Rung d (this ADR) lifts the restriction so any mix of `{mha, mla, lin, ssm}`
is interleavable — needed so a future scaled bake-off can test attention ⊕ SSM.

The obstacle is the parameter layout. `_blk_base(L) = _emb_end() + L·_blk()` assumes
a single block size. Making it per-layer would turn every within-block offset
(`_o_Wo`, the MLP offsets, the bias offsets) and `g_NP` into per-layer functions —
a pervasive, error-prone rewrite of the most delicate addressing code, threaded
through every `PL(L, _o_X())` call site in model.cyr, attn*.cyr, persist.cyr, and
the tests.

## Decision

**Pad every block's K/V region to the MAX `_kvw` over the kinds present in the
model (`_kvw_hyb`), keeping a UNIFORM per-block stride.** A layer of a smaller kind
tiles its own offsets at the shared region base (`2C + C²`) and leaves a zeroed pad
at the end. Because `t_alloc` zeroes and the pad receives no gradient (the layer's
backward writes only its kind's offsets), the pad stays zero through training and
round-trips cleanly — no explicit zeroing needed.

Consequences of the choice:

- **Only `_kv_weight_size()` changes** to return the padded max for a hybrid (the
  single kind's `_kvw` for a uniform model — byte-identical). Every `_o_*` helper
  stays zero-arg and uniform; `_o_Wo` and everything after it shift by the padded
  size, the same for all layers. No per-layer-offset refactor.
- **A single GLOBAL `latent_dim`** is shared by all MLA/SSM layers in a hybrid (MLA
  `d_c` == SSM state `N`), so the within-region offset helpers stay
  `g_latent_dim`-driven and zero-arg.
- The weight-init loop and the cache allocation branch per layer on `_lk(L)` /
  per-kind presence (`_any_kind`); a kind's caches are held when ANY layer uses it.
- `_hybrid_kinds_ok` drops the `_kvw`-equality requirement (padding removes it) and
  keeps the cross-cutting constraints: learned-absolute positions, full heads
  (`nkv == nh`, required by mla/ssm/lin), and a valid shared latent iff any mla/ssm.
- **Checkpoint v6 is unchanged in shape** — it already carries the per-layer kinds
  (ADR 0011); the loader sizes the padded block via the same `_kvw_hyb`. The base
  descriptor (slot 16) must be a latent-owning kind (mla/ssm) when the latent is
  nonzero, so the CLI `--attn-every K` overrides to MHA over a `--attn-kind {lin,
  mla, ssm}` base.

## Consequences

- **Positive** — completes M14: the full `{mha, mla, lin, ssm}` set is interleavable
  (X013 grad-checks the SSM⊕MHA and MLA⊕MHA mixed backwards at ~1e-4, bit-identical
  decode, v6 round-trip). The change is localized (one helper + per-layer init/cache
  gates), not a per-layer-offset rewrite. The decode cache becomes a continuous knob
  from pure-SSM's constant `C·N` to pure-MHA's `∝T` K/V — the attention fraction
  dials the trade. `{mha,gqa,lin}` hybrids are still exact (shared `_kvw`, max == that
  size, no pad — ADR 0011's behavior preserved).
- **Negative** — the padding wastes parameters on the smaller-kind layers
  (X013: +1 440 / ~4% at reference scale; the SSM layers' region padded up to MHA's).
  A single global latent forces all MLA/SSM layers in one hybrid to share `d_c`/`N`.
  And the base attn_kind (slot 16) is conflated with "the latent-owning kind" for a
  hybrid — a mild descriptor overload (a per-layer `latent_dim` would be the clean
  fix, deferred).
- **Neutral** — at reference scale the ratio sweep is within noise (X013); this is
  mechanism completeness, not a new headline result. The CLI expresses the practical
  hybrids (cheap base + periodic MHA); `model_init_full` supports any mix (e.g.
  lin⊕ssm) for programmatic / future use.

## Alternatives considered

- **Full per-layer parameter layout (exact, no pad)** — rejected for v1.4.4: `_blk_base`
  becomes a prefix-sum and every `_o_*` offset becomes per-layer, touching every
  `PL(L, _o_X())` site — high risk for no reference-scale gain. Padding gets the
  same expressivity for a localized change; the exact layout is a later option if
  the padding waste ever bites (it doesn't at our scale).
- **Keeping rung c's {mha,gqa,lin}-only restriction** — rejected: it permanently
  excludes the survey's strongest hybrid (attention ⊕ SSM) and attn11's best mixer.
- **A per-layer `latent_dim` array (v7)** — deferred: would let MLA and SSM layers
  in one hybrid carry different latent sizes and remove the slot-16 overload, but
  it's a checkpoint-format bump for flexibility no current experiment needs.
