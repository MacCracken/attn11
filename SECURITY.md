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
- adversarial *content* in `--corpus`/`--stdin` training data: the bytes are
  validated and safely tokenized (see below), but their *meaning* is untrusted —
  attn11 learns whatever you feed it.

## Attack surfaces & mitigations

| Surface | Mitigation |
|---|---|
| **Corpus loading** (`--corpus` / `--stdin`) | Opened with `O_NOFOLLOW` (refuses symlinks / TOCTOU at the final path component → `ELOOP`); size-capped via `fstat` to `MAX_CORPUS_BYTES` (64 MB, 1.5.3) **before** reading; byte-level tokenizer accepts any of the 256 byte values into a fixed 256-entry table; short reads are looped. A corpus shorter than `ctx+1` is detected and training is skipped rather than reading out of bounds. |
| **Checkpoint loading** (`--load`) | Opened with `O_NOFOLLOW`; the exact file size is `fstat`'d and capped (128 MB) before allocation. `ckpt_load_buf` validates magic, version (**v1 through v8 all accepted/loadable**; the v2 header adds `nkv` with `nkv ≥ 1`, `nkv ≤ nh`, `nh % nkv == 0`, plus `step ≥ 0` and `rng ≠ 0`; the v3 header adds the tokenizer triple `tok_kind ∈ {0,1}`, `base_vocab ∈ [1,256]`, `n_merges ∈ [0,512]` with `V = base + merges` enforced; v4 adds the architecture descriptor `attn_kind`/`pos_kind`/`latent_dim`/`rope_dim` with codes `-40..-43`; v5 adds the MoE descriptor `num_experts`/`expert_topk`, codes `-44`/`-45`; v6 adds the NL per-layer mixer kinds for a hybrid, each validated against the uniform-stride invariant, code `-46`; v7 adds the diffusion `objective` field, code `-47`; v8 adds the M16 `ternary` flag ∈ {0,1}, code `-48`, and rejects a ternary image paired with a non-scope axis (mla/lin/ssm/rope/MoE/diffusion), code `-49`), every config field (range + `C % nh`), recomputes the parameter count (≤ 4M; a hybrid's padded per-block size via `_kvw_hyb`), checks the **total model allocation** the config implies against a 128 MB bound (`model_alloc_bytes`, mirroring `model_init` exactly — so a shape-valid header can't drive the allocator past its cap), and requires the **exact** total size — all **before** any allocation or copy. Truncated / bit-flipped / wild-config / fully random inputs are rejected with a negative error, never a crash (see `tests/attn11.fcyr`: 500 byte-level + 500 BPE mutated rounds + 100 random corpora). On genuine host OOM, `t_alloc` aborts cleanly rather than writing through a null pointer. |
| **BPE merge table** (`--load`, v3 and later) | A hostile merge table is the classic decompression-bomb / cyclic-reference shape, so it is validated structurally before install: every merge's left/right id must lie in `[0, base + j)` for merge `j` — a **well-founded DAG** (rejects negative ids, self-reference, forward references, and therefore all cycles: the decoder cannot loop), and a length recurrence bounds every token's byte expansion to 64 (rejects exponential expansion chains, which double per merge and would reach 2^512 bytes). Both run on fixed stack scratch — hostile input drives **zero** heap allocation. Decode never recurses: expansions are precomputed into a flat span table at install time. Fuzzed with dedicated merge-slot, triple-inconsistency, and expansion-bomb mutation modes. |
| **Checkpoint writing** (`--save`) | Crash-atomic: written to a `.tmp` sibling (`O_NOFOLLOW`), **`fdatasync`'d** (flushes data + the size/block metadata a read needs — not `fsync`, which qemu-user aarch64 mis-emulates; `fdatasync` is portable and sufficient for the temp-write-then-rename pattern), then `rename`'d over the target (the rename is the only mutation of the target and is atomic). A failed/interrupted save leaves the prior checkpoint intact; `O_NOFOLLOW` refuses writing through a symlink. |
| **Numeric / memory layout** | All training buffers (parameters, optimizer state, activation caches, attention arena) are allocated once at fixed, dimension-derived sizes; no allocation or unbounded growth in the training loop. Indices are computed from validated dims. CLI numbers (`_atoi`) saturate, so a garbage-huge arg cannot wrap to a small/negative value. |
| **PRNG** | Deterministic xorshift64 with splitmix64 seeding — used for weight init and sampling only, **not** for any security purpose. Its state is part of a checkpoint solely for reproducible resume. |
| **Supply chain / CI** | GitHub Actions are **SHA-pinned** (no floating tags — the retag-compromise vector); the release job passes the git tag to `awk` as data, not interpolated code; `contents: write` is scoped to the release job only. The cyrius toolchain version is pinned in `cyrius.cyml` (single source of truth). Deferred (rationale in the M8 audit): installer `curl\|sh` SHA-pinning, release-artifact signing/provenance, a `lib/`-closure lockfile. |

## Residual notes

- Checkpoint save writes to `<path>.tmp` then renames over `<path>`; the temp
  uses `O_TRUNC` (a stale `.tmp` from a prior crash is overwritten) and
  `O_NOFOLLOW`. Don't point `--save` at an attacker-controlled directory.
- **The table above describes the Linux targets.** On the AGNOS target
  (v0.6.0+) three guarantees are weaker, inherent to the frozen agnos ABI:
  no `O_NOFOLLOW` (the `AO_*` flag set has no nofollow bit), sizes come from
  path-`stat` rather than `fstat` (a benign race — the size only caps the
  allocation, and reads stop at EOF), and durability uses the global `sync()`
  (no per-fd `fdatasync`). See `docs/guides/agnos.md` and
  `docs/audit/2026-06-10-agnos-audit.md`.
- Cross-architecture checkpoints are not portable: tensors are raw native
  little-endian `f64` bit patterns (no endianness header). A checkpoint is
  validated for shape/size but its payload is trusted once the header passes —
  load only checkpoints you produced.
- The **M8 security sweep** (`docs/audit/2026-06-11-m8-security-sweep-audit.md`)
  surveyed six vulnerability classes against recent CVEs and recorded a
  per-class disposition for each — including the headline negative result that
  the flat-i64 checkpoint is structurally immune to the model-file
  deserialization-RCE genre, and the accepted residuals (leaf-only
  `O_NOFOLLOW`, no `S_ISREG` gate on user-supplied corpus paths) with rationale.
