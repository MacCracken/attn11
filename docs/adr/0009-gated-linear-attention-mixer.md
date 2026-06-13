# 0009 — Gated linear attention as a sequence-mixer family

**Status**: Accepted
**Date**: 2026-06-12

## Context

M14 (E4) opens a second *sequence-mixer* family: the survey's structural shift
from pure softmax attention toward linear-attention / SSM mixers (constant-state,
linear-time, hybrids that beat pure transformers). The first rung is a gated
linear-attention block. attn11's invariants force several choices:

- The mixer must have a **hand-derivable, grad-checkable backward** (no autodiff).
- It must keep the **cached-decode bit-identity** contract (`docs/architecture/003`).
- It must be **additive and default-preserving** (the no-flag run byte-identical).
- It needs a home in the descriptor/checkpoint scheme without an axis explosion
  (ADR 0008 already flagged the stacked-positional-config smell).

Linear attention can be normalized (Katharopoulos, φ feature map + a denominator)
or unnormalized with a decay (RetNet retention). And it can be a brand-new
descriptor axis or ride the existing one.

## Decision

**Gated retention, unnormalized, fixed per-head decay.** Per head, the causal
recurrence `S_t = γ_h·S_{t-1} + k_t⊗v_t` (an hd×hd state), `out_t = (1/√hd)·S_t^T q_t`,
with `γ_h = 1 − 2^{−(3+h)}` (fixed, parameter-free, clamped against i64 shift
overflow). No softmax, no feature map, no normalizer — the decay supplies the
recency weighting and the whole core is multiply/add. The hand-derived backward
needs **no state caching**: `dq` from a forward S-recompute, `dk`/`dv` from a
reverse `dS` accumulator (`dS_t = Σ_{t'≥t} (1/√hd)·γ^{t'−t}·q_{t'}⊗dout_{t'}`).

**It rides `attn_kind` (value 2), reusing the MHA projections.** Q/K/V/O and their
biases are the MHA layout, so `_kv_weight_size` / `ckpt_expected_np` already size
it (the fall-through MHA path), weight init is unchanged, and **checkpoint v5
carries it in the existing `attn_kind` slot — no format bump**. The cached decode's
"cache" is the constant `nh·hd²` retention state (a per-layer `g_lin_state`), not a
T-growing K/V. Gated to full heads (`nkv == nh`), no latent (`d_c == 0`),
learned-absolute positions only (RoPE is softmax-only).

## Consequences

- **Positive** — the backward is pure multiply/add, so `test_lin_core`
  grad-checks at ~1e-9 (tighter than any softmax path) and is bit-identical
  cross-arch with no trig/exp polyfill concern. The cached single-row step runs
  the same recurrence as the batch scan, so bit-identity (`test_kv_lin`) is
  natural. The decode cache is **constant in T** — the structural win (16× under
  MHA at the preset). Reusing `attn_kind` means zero checkpoint/layout churn.
- **Negative** — unnormalized retention can grow unboundedly without the decay;
  it relies on `γ_h < 1` + the 1/√hd scale + the surrounding LayerNorm/residual
  for stability (fine at reference scale, grad-checked; a normalizer or output
  GroupNorm would be the hardening if a larger run diverges). Overloading
  `attn_kind` with a *mixer* meaning (softmax vs retention) alongside its
  *projection* meaning (mha vs mla) is a mild conflation; a future SSM is
  `attn_kind = 3`, and per-layer interleaving (M14 rung c) will need a per-layer
  kind array regardless of the field's name.
- **Neutral** — the per-head decay schedule is fixed (no `--decay` knob); learning
  it, adding a normalizer, the SSM rung (b), and the hybrid interleave (c) are
  follow-ons. A vidya-scale bake-off across mixers is the next X-entry.

## Alternatives considered

- **Normalized linear attention (Katharopoulos, φ=elu+1 + denominator)** —
  rejected for the first rung: it is more stable but the backward adds the φ
  derivative and a quotient rule, and the roadmap calls for the *easiest
  hand-derivable backward* first. The normalizer is the natural hardening step if
  unnormalized retention proves unstable at scale.
- **A new `g_mixer_kind` descriptor axis + checkpoint v6** — rejected: it would
  add a fourth config-delegation layer and a format bump for no semantic gain;
  linear attention is mechanically an attention variant over the MHA layout, so
  `attn_kind = 2` is the minimal, no-bump home.
- **Learned per-head decay** — rejected for now: fixed RetNet-style decays are
  parameter-free (the no-flag run and the param count stay clean) and sufficient
  to demonstrate the grad-checked mixer; a learned γ is a small additive follow-on.
- **Differentiating a normalized softmax-free attention without decay** —
  rejected: with γ = 1 the unnormalized sum is an unbounded random walk; the decay
  is what makes the retention both stable and a constant-state recurrence.
