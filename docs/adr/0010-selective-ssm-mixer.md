# 0010 — Selective SSM as the third sequence-mixer family

**Status**: Accepted
**Date**: 2026-06-12

## Context

M14 rung (b) adds a selective state-space model — the survey's structural shift
toward SSM/attention hybrids. The Mamba insight is *selectivity*: the SSM
parameters (Δ, B, C) are functions of the input, not fixed (as in S4). attn11's
invariants force the design: the backward must be hand-derivable and
grad-checkable (no autodiff); it must keep the cached-decode bit-identity contract
and the byte-identical no-flag run; and it should not add a descriptor axis or a
checkpoint format bump if it can ride the existing machinery (per ADR 0009's
precedent for gated linear attention).

Full Mamba (the input-dependent conv, SiLU gating, the dt-rank low-rank Δ, the
hardware-aware scan) is far more than a reference needs. The question is which
*minimal* selective SSM captures the essence and grad-checks cleanly.

## Decision

**A minimal selective diagonal SSM, `attn_kind = 3`.** Per channel c, an N-dim
diagonal state. The ZOH discretization with the first-order input term `B̄ = Δ·B`:

```
Δ_t = softplus(a_t·W_dt + b_dt)   B_t = a_t·W_B   C_t = a_t·W_C     (all selective: functions of a_t)
h_t[c,n] = exp(Δ_t[c]·A[c,n])·h_{t-1}[c,n] + Δ_t[c]·B_t[n]·a_t[c]
y_t[c]   = Σ_n C_t[n]·h_t[c,n] + D[c]·a_t[c]            out_t = y_t·W_o + b_o
```

`A` is a learned diagonal (inited to a negative ramp `-(n+1)`, an S4D-style spread
of decay rates); `D` is the identity skip (inited to 1). The hand-derived BPTT runs
a reverse scan accumulating `dh` through the data-dependent `exp(Δ·A)`, sending
gradient to Δ/B/C/A/D and the input.

**Layout reuse (the MLA/`_kvw` pattern, ADR 0007/0009):** `Wq` doubles as `W_dt`
and `Wo` as the output projection (with `bq`/`bo`); the SSM-specific A/W_B/W_C/D
live in the K/V region, so `_kvw` gets the `attn_kind == 3` case (`3·C·N + C`) and
the 1.4.1 dispatch helpers get one branch each. The state size `N` reuses the
`latent_dim` descriptor field — so **checkpoint v5 carries the SSM with no format
bump**. The decode cache is the constant per-layer `C·N` state (a third
constant-cache mixer, with gated linear attention).

## Consequences

- **Positive** — the selective scan's BPTT grad-checks at ~1e-7 (`test_ssm_core`),
  the milestone that the idea is expressible hand-derived in an i64 language. No
  new descriptor axis, no checkpoint bump (rides `attn_kind`/`latent_dim`); the
  1.4.1 dispatch + `_kvw` refactor made wiring a localized addition. Constant
  decode cache; at reference scale it posts the best bits/byte of the four mixers
  (X011). Lands in its own `attn_ssm.cyr` (the one-file-per-mixer pattern).
- **Negative** — the per-channel×state scan is O(T·C·N) and the backward caches the
  full `h` (T·C·N per layer) for the `dh·h_{t-1}`/`dy·h_t` terms — heavier than the
  other mixers, but fine at reference scale. Overloading `attn_kind` (now mha/mla/
  lin/ssm) and `latent_dim` (MLA d_c / SSM N) is a mild conflation; a fifth axis or
  the per-layer hybrid (rung c) will eventually want a cleaner descriptor.
- **Neutral** — this is a *minimal* selective SSM: no input conv, no SiLU gate, no
  low-rank Δ, first-order B̄. Those are faithful-Mamba follow-ons if a scaling
  experiment ever earns them. A vidya-scale bake-off is the next X-entry.

## Alternatives considered

- **Full Mamba (conv + gate + dt-rank + ZOH B̄)** — rejected for the reference:
  far more machinery and a more involved backward for no grad-check or
  expressivity gain at this scale; the minimal selective scan already demonstrates
  the idea and grad-checks.
- **A new descriptor axis + checkpoint v6 for the state size** — rejected:
  `latent_dim` is already "the extra-dimension parameter for the chosen
  attn_kind" (MLA's d_c); reusing it for the SSM state N avoids a format bump, as
  gated linear attention reused `attn_kind` (ADR 0009).
- **A dedicated SSM block layout (own Wq/Wo region)** — rejected: reusing Wq (as
  W_dt) and Wo (output proj) keeps `_o_Wo`/`_o_ln2g` and the bias region unshifted
  (the MLA reuse pattern), so only the K/V region content differs.
- **Non-selective S4 (fixed Δ/B/C)** — rejected: selectivity (input-dependent
  Δ/B/C) is the whole point of the Mamba line and the harder, more interesting
  backward; a fixed SSM would be closer to gated linear attention (rung a).
