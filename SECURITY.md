# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the attn11 version from `VERSION`. Expect an initial response within
one week. Coordinated disclosure is appreciated — do not open a public GitHub
issue with exploit details.

## Threat model

attn11 is a single-process, CPU-only transformer trainer. It has **no
networking**. By default it consumes only an embedded, compile-time corpus, but
since v0.3.0 it can also load a **corpus file / stdin** (`--corpus`, `--stdin`)
and a **checkpoint file** (`--load`) — these are the untrusted-input surfaces. A
realistic attacker is assumed able to:

- supply the build inputs (source / corpus) — i.e. they already control what
  gets compiled, which is equivalent to code execution,
- hand-craft a corpus or checkpoint file placed at the path attn11 will open,
  and/or race attn11 on the filesystem (symlink plant / TOCTOU), and/or
- run the resulting binary with attacker-chosen hyperparameters.

attn11 *does not* defend against:

- an attacker with arbitrary code execution in the build or host process
  (trivially owns the whole address space),
- remote / network attacks (attn11 has no networking),
- side channels (timing, cache),
- adversarial training data once a future release adds file/stdin corpus
  loading — see below.

## Attack surfaces & mitigations

| Surface | Mitigation |
|---|---|
| **Corpus loading** (`--corpus` / `--stdin`) | Opened with `O_NOFOLLOW` (refuses symlinks / TOCTOU at the final path component → `ELOOP`); size-capped via `fstat` to `MAX_CORPUS_BYTES` (4 MB) **before** reading; byte-level tokenizer accepts any of the 256 byte values into a fixed 256-entry table; short reads are looped. A corpus shorter than `ctx+1` is detected and training is skipped rather than reading out of bounds. |
| **Checkpoint loading** (`--load`) | Opened with `O_NOFOLLOW`; the exact file size is `fstat`'d and capped (128 MB) before allocation. `ckpt_load_buf` validates magic, version (v1 and v2 accepted; the v2 header adds `nkv` with `nkv ≥ 1`, `nkv ≤ nh`, `nh % nkv == 0`, plus `step ≥ 0` and `rng ≠ 0`), every config field (range + `C % nh`), recomputes the parameter count (≤ 4M), checks the **total model allocation** the config implies against a 128 MB bound (`model_alloc_bytes`, mirroring `model_init` exactly — so a shape-valid header can't drive the allocator past its cap), and requires the **exact** total size — all **before** any allocation or copy. Truncated / bit-flipped / wild-config / fully random inputs are rejected with a negative error, never a crash (see `tests/attn11.fcyr`: 500 mutated rounds + 100 random corpora). On genuine host OOM, `t_alloc` aborts cleanly rather than writing through a null pointer. |
| **Checkpoint writing** (`--save`) | Crash-atomic: written to a `.tmp` sibling (`O_NOFOLLOW`), `fsync`'d, then `rename`'d over the target (the rename is the only mutation of the target and is atomic). A failed/interrupted save leaves the prior checkpoint intact; `O_NOFOLLOW` refuses writing through a symlink. |
| **Numeric / memory layout** | All training buffers (parameters, optimizer state, activation caches, attention arena) are allocated once at fixed, dimension-derived sizes; no allocation or unbounded growth in the training loop. Indices are computed from validated dims. |
| **PRNG** | Deterministic xorshift64 with splitmix64 seeding — used for weight init and sampling only, **not** for any security purpose. Its state is part of a checkpoint solely for reproducible resume. |

## Residual notes

- Checkpoint save writes to `<path>.tmp` then renames over `<path>`; the temp
  uses `O_TRUNC` (a stale `.tmp` from a prior crash is overwritten) and
  `O_NOFOLLOW`. Don't point `--save` at an attacker-controlled directory.
- **The table above describes the Linux targets.** On the AGNOS target
  (v0.6.0+) three guarantees are weaker, inherent to the frozen agnos ABI:
  no `O_NOFOLLOW` (the `AO_*` flag set has no nofollow bit), sizes come from
  path-`stat` rather than `fstat` (a benign race — the size only caps the
  allocation, and reads stop at EOF), and durability uses the global `sync()`
  (no per-fd `fsync`). See `docs/guides/agnos.md` and
  `docs/audit/2026-06-10-agnos-audit.md`.
- Cross-architecture checkpoints are not portable: tensors are raw native
  little-endian `f64` bit patterns (no endianness header). A checkpoint is
  validated for shape/size but its payload is trusted once the header passes —
  load only checkpoints you produced.
