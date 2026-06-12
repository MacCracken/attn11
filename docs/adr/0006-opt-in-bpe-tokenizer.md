# 0006 — Opt-in BPE tokenizer, byte-level default

**Status**: Accepted — amends [0002](0002-byte-level-tokenizer-over-bpe.md)
(adds opt-in BPE; byte-level remains the default 0002 chose)
**Date**: 2026-06-11

## Context

ADR 0002 chose a byte-level adaptive tokenizer and deferred BPE until "corpora
grow past the point where byte-level sequence length hurts". The X001 vidya run
(488 KB of real Cyrius code) reached that point: at ctx 16–64 a byte-level
window covers fragments, not statements, and the frontier survey's
data-efficiency thesis made *byte-vs-BPE at iso-compute* a question worth
answering with numbers (roadmap E3 → M7). Doing that requires both tokenizers
to coexist, be selectable, and round-trip through checkpoints.

## Decision

Add a **simple BPE tokenizer as an opt-in** (`--bpe K`, K ∈ [1, 512] merges);
**byte-level stays the default** — a run without `--bpe` is bit-identical to
one on 0.7.0. Scope and the frozen semantics:

- BPE is **layered on the byte tokenizer**: base ids are the adaptive byte
  vocab (≤ 256); merge `m` mints id `B + m` for one adjacent pair. Max vocab
  768. Merge training is pure i64 (no f64, no PRNG), so it is bit-reproducible
  across architectures.
- **Frozen tie-break**: the pair-count argmax scans row-major ascending with
  strict `>` — highest count, then smallest left id, then smallest right id.
  Replacement is greedy left-to-right non-overlapping; counting includes
  overlaps. Frozen because checkpoints *store* the merges and every encode
  replays them — the learner's ordering choices must never drift.
- **Checkpoints carry the tokenizer** (format v3: `tok_kind`, base vocab,
  merge table). A loaded checkpoint's tokenizer always wins; `--bpe` under
  `--load` is ignored with a warning. The loader validates the merge table as
  a well-founded DAG (every reference strictly below the minting id) with a
  bounded per-token expansion (≤ 64 bytes) — hostile tables cannot loop the
  decoder or expand exponentially.
- Each token's expansion is **precomputed at install time** into a flat
  span table; decode never recurses.

## Consequences

- **Positive** — byte-vs-BPE is now measurable at iso-compute on the same
  binary (`--eval` reports bits-per-byte, comparable across tokenizers); BPE
  shortens vidya-class sequences ~2.3×, so a fixed window covers whole
  statements; the audited byte-level path is untouched by default.
- **Negative** — the checkpoint loader's hostile-input surface grows (merge
  table validation, codes −32…−39); vocab can exceed 256, so the model and
  loader caps had to be re-derived (V ≤ 768) and re-fuzzed; the tied LM head
  costs O(V·C) per token, so a BPE model pays more per step (priced into the
  X003 iso-compute protocol).
- **Neutral** — `corpus_set` retains the raw corpus bytes so a resume can
  re-encode them through a loaded merge table (idempotent by construction);
  the experiments ledger owns the byte-vs-BPE verdict (X003), not this ADR.

## Alternatives considered

- **Replace byte-level with BPE** — rejected: ADR 0002's reproducibility and
  zero-training-step arguments still hold for the default path, and the
  comparison *requires* both.
- **Tokenizer in a sidecar file** — rejected: a checkpoint that doesn't carry
  its tokenizer can silently mis-index every embedding row if paired with the
  wrong merges; one self-consistent file keeps the M2 "validated before
  trusted" discipline.
- **Re-derive merges from the corpus at resume** — rejected for the same
  mis-indexing reason; the merges are model state, exactly like weights.
- **Priority-queue/linked-list BPE (linear-time)** — rejected: the O(K·(n +
  V²)) naive learner costs ~110 ms per 256 KB at K=128 — a one-shot
  pre-training cost that does not justify the extra pointer machinery in an
  arena-only, everything-is-i64 codebase.
