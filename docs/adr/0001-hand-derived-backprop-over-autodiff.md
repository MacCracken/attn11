# 0001 — Hand-derived backprop over an autodiff engine

**Status**: Accepted
**Date**: 2026-06-08

## Context

Training a transformer needs gradients for every parameter. Two ways to get
them: build a general automatic-differentiation engine (a tape or computation
graph that records ops and replays them in reverse), or hand-derive a backward
pass for each operation and call them in the right order.

Cyrius is an assembly-up, everything-is-i64 systems language with no closures,
no generics, and manual memory management. A dynamic autodiff graph — nodes,
edges, topological replay, per-node gradient buffers — is a large amount of
machinery to express in that substrate, and it adds per-op indirection and
allocation that a from-scratch CPU trainer can't spare.

## Decision

Hand-derive the backward pass for each op (linear, layernorm, GELU, softmax
cross-entropy, attention, the tied head, embeddings) and wire them together
explicitly in `model_backward`. Correctness is gated by **central
finite-difference gradient checks** — every backward op is verified against
`(L(x+ε) − L(x−ε)) / 2ε` before it lands, and a full-model check covers the
assembled graph. No autodiff engine.

## Consequences

- **Positive** — minimal and fast: no graph, no per-node allocation, full
  control of the memory layout and reduction order. Gradients are explicit and
  auditable. The grad-check harness doubles as living documentation of each op's
  math.
- **Negative** — every new op requires a hand-derived backward *and* a grad
  check; there are no free higher-order gradients; a forward refactor forces a
  matching backward re-derivation.
- **Neutral** — the finite-difference grad-check harness is now core
  infrastructure, not just a test (see `tests/attn11.tcyr`).

## Alternatives considered

- **A tape/graph autodiff engine** — rejected: the dynamic-graph machinery is
  heavy in an i64-only language with no closures, and the indirection/allocation
  overhead is exactly what a scalar-f64 CPU trainer must avoid. It is listed as
  explicitly out-of-scope through v1.0 in `docs/development/roadmap.md`.
- **Numerical gradients in the training loop** — rejected: O(params) forward
  passes per step is far too slow; finite differences are for *verification*
  only.
