# 0008 — Mixture-of-Experts router: combine, balance, and the dense invariant

**Status**: Accepted
**Date**: 2026-06-12

## Context

M13 (E8) turns each block's dense GELU MLP into N experts with a top-K router,
decoupling parameter count from per-token FLOPs. attn11's whole premise is
hand-derived, finite-difference-grad-checked backward, so the router's discrete
top-K selection — which has no honest gradient — forces three design choices that
the grad-check discipline and the frozen-surface/byte-identical invariants all
bear on:

1. **How the selected experts are combined** (which fixes what the combine
   backward differentiates).
2. **How expert collapse is prevented** (a top-K router left alone routes
   everything to a few experts).
3. **How `--experts 1` behaves** (the no-flag run must stay byte-identical, and
   the dense path is the sweep's baseline).

The selection itself must also be deterministic and bit-reproducible cross-arch,
the same constraint the BPE merge tie-break carries (ADR 0006).

## Decision

**Combine = Mixtral-style renormalized top-K softmax.** Router logits `r = x·Wg`
(bias-free `C→N`); take the top-K by logit; the gate weights are
`softmax(r restricted to the K selected logits)` — equivalently the full
`softmax(r)` renormalized over the selected set (the normalizer `Z` cancels). The
block output is `Σ_{k∈topK} g_k · E_k(x)`. The backward sends gradient **only to
the selected logits** (a K-way softmax Jacobian) — straight-through for the
discrete pick; unselected logits get exactly zero from the combine path.

**Balance = Switch-style auxiliary loss.** `L_aux = α·N·Σ_i f_i·P_i` (α = 0.01),
where `f_i` is the fraction of tokens dispatched to expert i (the discrete count,
held **constant** in backward — straight-through) and `P_i` is the mean full-`softmax(r)`
probability for expert i (differentiated). `L_aux` is added to the cross-entropy
**training** objective and is off the eval/bits-per-byte path. It is the only
gradient that reaches the *unselected* experts' router logits, so it is what keeps
load balanced.

**`--experts 1` is the dense MLP, byte-for-byte.** One expert needs no routing, so
`g_num_experts == 1` takes the original `Wfc/GELU/Wproj` path with **no gate and no
aux loss** — the MLP weight region, parameter count, and arithmetic are identical
to pre-M13. Expert 0's packed offset aliases the dense `_o_Wfc`, so the layout is
unchanged.

**Frozen tie-break.** Top-K scans experts low→high and replaces only on a strict
`>`, so equal logits resolve to the lower index — deterministic and bit-identical
across x86_64 / aarch64 (the logits themselves are bit-identical cross-arch given
the SIMD matmul contract).

## Consequences

- **Positive** — the combine backward is a clean K-way softmax over a fixed set
  plus reused `linear`/`gelu` backward; the aux backward is one more softmax
  Jacobian. Both grad-check tightly in isolation (`test_moe_op` 1e-4, `test_moe_aux`
  1e-5) and the whole thing composes (full-model 1e-3). The dense invariant means a
  no-flag run and every pre-v5 checkpoint stay byte-identical, and the MoE MLP is
  position-independent so the cached decode path reuses the same kernel
  bit-for-bit (`test_kv_moe`). The sweep (X009) confirms balance holds (routing
  entropy 0.99+) at every N.
- **Negative** — two softmaxes share the logits (selected-set for the combine,
  full-N for the aux), computed separately; the aux runs a second gate backward
  rather than folding into the combine's (a tiny `C×N` matmul, accepted for
  clarity over a fused pass). The `_arch` constructors gained a third layer
  (`_moe`) of delegation — a future axis will want a cleaner config struct than
  stacked positional params.
- **Neutral** — α is fixed at 0.01 (no `--aux-alpha` flag); a sweep over α and a
  vidya-corpus bake-off are follow-on X-entries. The MLA up-projection absorption
  optimization (ADR 0007) is orthogonal and still deferred.

## Alternatives considered

- **Switch keep-`p_i` combine** (use the full-softmax probability as the gate
  weight, not renormalized) — rejected: for top-1 it leaves a non-unit gate and
  routes gradient to *all* logits through `Z`, muddying the straight-through story.
  The renormalized form makes the combine gradient flow exactly to the selected
  set, which is the cleaner reference. (For top-2 the two differ only by the `Z`
  normalization; Mixtral's choice is the modern default.)
- **Differentiating the top-K selection** (e.g. Gumbel/soft-top-K) — rejected:
  attn11's contract is hand-derived gradients, and a soft relaxation would change
  the forward and forfeit the bit-identity contract. Straight-through keeps the
  forward exact and the backward grad-checkable.
- **No aux loss, rely on noise/init** — rejected: the sweep shows a top-K router
  needs the balance pressure; without it the load collapses and most experts go
  untrained (and the milestone explicitly is the *grad-checked* load-balance
  backward).
- **`--experts 1` carrying a 1-wide gate** — rejected: it would add `C` params and
  break the byte-identical no-flag run for zero benefit (a single expert is never
  routed).
