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
| **Checkpoint loading** (`--load`) | Opened with `O_NOFOLLOW`; the exact file size is `fstat`'d and capped before allocation. `ckpt_load_buf` validates magic, version, every config field (range + `C % nh`), recomputes the parameter count, and requires the **exact** total size — all **before** any allocation or copy. Truncated / bit-flipped / wild-config / fully random inputs are rejected with a negative error, never a crash (see `tests/attn11.fcyr`: 500 mutated rounds + 100 random corpora). |
| **Checkpoint writing** (`--save`) | Opened `O_WRONLY\|O_CREAT\|O_TRUNC\|O_NOFOLLOW` (refuses writing through a symlink); short writes looped. |
| **Numeric / memory layout** | All training buffers (parameters, optimizer state, activation caches, attention arena) are allocated once at fixed, dimension-derived sizes; no allocation or unbounded growth in the training loop. Indices are computed from validated dims. |
| **PRNG** | Deterministic xorshift64 with splitmix64 seeding — used for weight init and sampling only, **not** for any security purpose. Its state is part of a checkpoint solely for reproducible resume. |

## Residual notes

- Checkpoint save uses `O_TRUNC` without `O_EXCL` (it intentionally overwrites
  an existing checkpoint at the chosen path); `O_NOFOLLOW` still refuses a
  symlinked target. Don't point `--save` at an attacker-controlled directory.
- Cross-architecture checkpoints are not portable: tensors are raw native
  little-endian `f64` bit patterns (no endianness header). A checkpoint is
  validated for shape/size but its payload is trusted once the header passes —
  load only checkpoints you produced.
