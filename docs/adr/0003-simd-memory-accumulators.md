# 0003 — SIMD via memory-accumulator f64v_fmadd, not value-form vectors

**Status**: Accepted
**Date**: 2026-06-08

## Context

Matmul (`linear_fwd`/`linear_bwd` plus the attention projections) is ~80% of a
training step, so it is the place to vectorize. Cyrius exposes two SIMD surfaces:
value-form vector ops (`f64v2_add`, `f64v2_fmadd`, … returning `f64v2`/`f64v4`
values) and a low-level packed intrinsic `f64v_fmadd(out, a, b, c, lanes)` that
operates on memory pointers (`out` may alias `c`).

A verified toolchain miscompile: **reassigning a SIMD-typed local**
(`accv = f64v2_fmadd(...)` where `accv` is an existing `f64v2`) silently writes
garbage. Loop accumulators inherently reassign, so the value-form path is unsafe
for the exact hot loop we need.

## Decision

Vectorize the matmul inner loops 4-wide using **`f64v_fmadd` with accumulators
and scalar broadcasts in plain byte buffers** (`var acc[32]`, `var xv[32]`),
accumulating in place. Never hold an accumulator in a SIMD-typed var. A scalar
tail handles dimensions that aren't a multiple of 4. Loop order preserves the
scalar reduction sequence so results match scalar exactly (AXPY paths
bit-identical; dot paths within rounding).

## Consequences

- **Positive** — correct *and* fast: ~3.9× on `linear_fwd`, ~2.3× on a full
  fwd+bwd step (see `docs/benchmarks.md`). On x86_64 the builtin lowers to
  `mulpd`+`addpd`, so it is bit-identical to scalar — pinned by
  `test_simd_contract`. Works on aarch64 (fused `fmla`).
- **Negative** — more verbose than value-form (explicit buffers, manual tails),
  and the code is coupled to the `f64v_*` builtins.
- **Neutral** — the rule "never reassign a SIMD-typed var; accumulate in memory"
  is documented in `docs/architecture/001-tensors-and-floats.md`.

## Alternatives considered

- **Value-form `f64v2`/`f64v4` ops** — rejected: the reassignment miscompile
  makes loop accumulators produce garbage, which is precisely the loop we vectorize.
- **Stay scalar** — rejected: leaves ~2.3× on the table for the dominant cost.
- **Hand-written inline assembly** — rejected: unnecessary given the packed
  builtin, and it would break the aarch64 cross-build.
