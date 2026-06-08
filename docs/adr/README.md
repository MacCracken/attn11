# Architecture Decision Records

Decisions about attn11 — what we chose, the context, and the consequences we accept. Use these when a future reader would reasonably ask *"why did we do it this way?"*

## Conventions

- **Filename**: `NNNN-kebab-case-title.md`, zero-padded to four digits. Never renumber.
- **One decision per ADR.** If a decision supersedes a prior one, add a new ADR and set the old one's status to `Superseded by NNNN`.
- **Status lifecycle**: `Proposed` → `Accepted` → (optionally) `Superseded` or `Deprecated`.
- Use [`template.md`](template.md) as the starting point.

## ADR vs. architecture note vs. guide

| Kind | Lives in | Answers |
|---|---|---|
| ADR | `docs/adr/` | *Why did we choose X over Y?* |
| Architecture note | `docs/architecture/` | *What non-obvious constraint is true about the code?* |
| Guide | `docs/guides/` | *How do I do X?* |

## Index

- [0001 — Hand-derived backprop over an autodiff engine](0001-hand-derived-backprop-over-autodiff.md) — each backward op is hand-written and finite-difference grad-checked; no autodiff graph.
- [0002 — Byte-level tokenizer over BPE](0002-byte-level-tokenizer-over-bpe.md) — adaptive ≤256 byte vocab; BPE deferred.
- [0003 — SIMD via memory-accumulator f64v_fmadd](0003-simd-memory-accumulators.md) — matmul vectorized with memory accumulators (never reassign a SIMD-typed var).
- [0004 — Native-endian raw checkpoint blob](0004-native-endian-checkpoint-format.md) — fast same-arch resume; not cross-architecture portable.
