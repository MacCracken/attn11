# 0004 — Native-endian raw checkpoint blob

**Status**: Accepted
**Date**: 2026-06-08

## Context

Checkpoints persist the full training state — parameters, Adam moments, PRNG
state, step counter, config, and vocab — so a run can resume bit-for-bit. The
tensors are large flat `f64` arrays. Two encodings: a portable format with
explicit byte-order handling (and a schema/serialization layer), or a raw blob
written straight from memory with `store64`/`load64`.

The realistic use is *resume on the same machine*. AGNOS has no folded-in
serialization crate, and `store64`/`load64` round-trip a contiguous region
unchanged on the same architecture.

## Decision

Write checkpoints as a **raw native-endian (little-endian on x86_64/aarch64)
blob**: a fixed i64 header (magic, version, config, NP, step, RNG state), the
vocab table, then `g_params`, `g_adam_m`, `g_adam_v` copied contiguously. The
loader validates magic, version, every config field, a recomputed parameter
count, and the exact total size **before any allocation**. Checkpoints are not
cross-architecture portable.

## Consequences

- **Positive** — trivial and fast (one contiguous `write`/`read`), exact
  round-trip on the same arch (verified by the bit-for-bit resume-determinism
  test). The validated header makes loading hostile-input-safe (fuzzed).
- **Negative** — a checkpoint is not portable across architectures or
  endianness; loading a foreign or untrusted checkpoint is the user's
  responsibility (documented in `SECURITY.md`).
- **Neutral** — a portable format can be added later behind a checkpoint
  version bump; the `version` field already exists for exactly this.

## Alternatives considered

- **Portable explicit-endianness encoding** — rejected: byte-swapping plus a
  schema for a same-machine-resume use case is cost without benefit at this stage.
- **A serialization library** — rejected: none is folded into the AGNOS stdlib,
  and pulling one in would violate the zero-dependency goal.
