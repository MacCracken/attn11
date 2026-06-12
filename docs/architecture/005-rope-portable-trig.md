# 005 — RoPE's portable trigonometry (no native sin/cos)

**What's true**: coupled RoPE (`--pos-kind rope`, 1.2.2) needs `cos`/`sin` of
rotation angles, but `f64_sin`/`f64_cos` are **x86-only compiler builtins** on
this toolchain — aarch64 has no native trig and v5.6.0 does not polyfill it (the
same situation as the `f64_atan` inverse-trig family; see the ganita note). The
project ships on x86_64 **and** aarch64, so RoPE cannot call those builtins.

`rope_apply_fwd`/`rope_apply_bwd` (`src/attn.cyr`) therefore compute the rotation
without any trig builtin, and they must do so **deterministically** — the same
rotation for a given absolute position in the batch path and the cached
single-row path, or the bit-identity contract (note 003) breaks.

## The construction

RoPE never needs a *general* sin/cos. It only needs `cos(m·θ_k)` / `sin(m·θ_k)`
where:

- `θ_k = base^(-2k/hd) ∈ (0, 1]` rad (base 10000, `k = 0 … hd/2-1`), and
- `m` is an **integer** absolute position.

So the angle factors into a base rotation by `θ_k` raised to an integer power:

1. **Base rotation** — `cos θ_k`, `sin θ_k` from a **Maclaurin series**. Because
   `θ_k ≤ 1` rad there is **no range reduction** (the hard, bug-prone part of a
   general trig routine); the series through `θ^16`/`θ^17` is ~1e-15 at `θ = 1`.
   The pair is then renormalised to exactly unit magnitude so step 2 cannot drift
   it (`_rope_unit_cossin`).
2. **Position power** — `(cos m·θ, sin m·θ) = (cos θ + i·sin θ)^m` by **binary
   exponentiation** over complex multiplication, `O(log m)` (`_rope_pow`).

Everything is pure `f64` add/mul/div plus `f64_sqrt` and `f64_pow` (the latter is
`exp(e·ln base)`), all of which **are** polyfilled on aarch64. So the whole path
is available and deterministic on both architectures.

## Why this preserves bit-identity (note 003)

The rotation for absolute position `m` is computed **directly from `m`** — via
`_rope_pow(base, m)` — and never via a row-to-row recurrence. So:

- the **batch path** rotates row `i` by position `i` → `_rope_pow(base, i)`;
- the **cached row path** rotates the single new row at position `pos` →
  `_rope_pow(base, pos)`;
- the **context-shift re-prime** recomputes a kept token at its *new* position
  `p'` → `_rope_pow(base, p')`.

Identical inputs `(base, m)` ⇒ identical bits, so a position's rotation is the
same however it is reached. The forward and backward share `_rope_unit_cossin`
+ `_rope_pow`, so the rotation used to build the score and the one used to undo
it in the gradient match bit-for-bit too. Pinned by `test_kv_rope`
(cached-vs-uncached across shifts) and `test_rope_op` (the rotation backward is
bit-exact; the relative-position invariance holds to rounding).

## What this forbids / constrains

- **Do not** reach for `f64_sin`/`f64_cos` in any code that must build on
  aarch64 — the build hard-errors (`f64_cos is x86-only`). Use the
  `_rope_*` helpers, or extend them.
- **Do not** switch the position power to a per-row recurrence
  (`rot(m) = rot(m-1)·step`) for speed: it accumulates rounding differently from
  the direct power, so the cached single-row path (which computes `rot(pos)`
  directly) would no longer match the batch path bit-for-bit.
- A **decoupled-RoPE** rung for MLA (1.2.3) reuses these same helpers; its only
  new surface is *where* the rotation is applied (a separate `d_rope` channel),
  not *how* cos/sin are produced.

## Cross-arch note

The polyfilled `f64_exp`/`f64_pow` differ from x86 at the ULP level on aarch64
(as `f64_exp` already does for softmax), so RoPE's *cross-arch* agreement is
display-precision, not bit-exact — exactly like the rest of the model. The
*within-arch* cached-vs-uncached contract is bit-exact (same code, same inputs).
The grad-check tolerances already account for the aarch64 transcendental noise
(`test_model_rope` rides the 1e-3 full-model bound, like `test_model_mla`).
