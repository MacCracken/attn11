# 0002 — Byte-level tokenizer over BPE

**Status**: Accepted
**Date**: 2026-06-08

## Context

The model needs a tokenizer to turn a corpus into integer ids. The two common
choices: a char/byte-level tokenizer (vocab is the set of observed byte values,
≤256, assigned ids by first occurrence), or byte-pair encoding (BPE), which
*learns* a merge table to produce a larger subword vocabulary with better
sequence compression.

attn11's goal is a tiny, reproducible, dependency-free reference trainer. BPE
adds a merge-table training step, a serialization format for the merges, and
encode/decode complexity — all of which must round-trip through checkpoints.

## Decision

Use a **byte-level adaptive tokenizer**: scan the corpus, assign an id to each
distinct byte in first-occurrence order, encode the stream. Vocab ≤ 256. BPE is
deferred (noted as optional in the roadmap).

## Consequences

- **Positive** — no tokenizer training, fully deterministic, handles arbitrary
  bytes (any corpus, any encoding), and the implementation is a few dozen lines.
  Matches the "tiny and reproducible" goal and keeps checkpoints simple (the
  vocab is just a ≤256-entry byte table).
- **Negative** — sequences are longer than with subword units, so the fixed
  context window covers fewer "words"; on larger natural-language corpora this
  costs effective context and compute.
- **Neutral** — BPE remains a clean future addition behind the same
  `corpus_set` interface; the checkpoint format already versions its vocab.

## Alternatives considered

- **BPE / subword tokenizer** — rejected for now: the merge-table training and
  serialization are significant code for marginal benefit at this scale.
  Revisit when corpora grow past the point where byte-level sequence length
  hurts (roadmap M2 flagged it as optional).
- **A fixed external vocabulary** — rejected: defeats the self-contained,
  no-data-files design and wouldn't adapt to an arbitrary corpus.
