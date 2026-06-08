# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.1] - 2026-06-08

First-party standards conformance (docs/process; no model behavior change).

### Added
- `CODE_OF_CONDUCT.md` (required root file) and a `Makefile` convenience wrapper.
- `docs/sources.md` — academic citation map for every algorithm (Transformer,
  LayerNorm, GELU, Adam, weight tying, GPT-2 init scaling, cosine LR, grad
  clipping, xorshift64, splitmix64, Marsaglia polar, finite-difference checks),
  plus inline citation comments on the declaring functions.
- `docs/audit/2026-06-08-audit.md` — consolidated security/correctness audit
  (the M2/M3/M4 adversarial-review findings, by severity, all fixed).
- ADRs `0001`–`0004` (hand-derived backprop, byte-level tokenizer, SIMD memory
  accumulators, native-endian checkpoint) + index.
- `scripts/bench-history.sh` — commit-stamped CSV + Markdown bench trail.
- `docs/examples/minimal_train.cyr` — runnable minimal-API example.

### Changed
- Toolchain pin `6.1.5` → `6.1.6`.
- `CLAUDE.md` restructured to the first-party template (genesis + shared-crates
  links, fixed the stale `applications/` standards path → `first-party/`,
  standard Rules block, Cyrius Conventions, CI/Release, P(-1)/work-loop process).
- Architecture note renumbered to `docs/architecture/001-tensors-and-floats.md`
  and indexed; `.gitignore` aligned to the standard (release artifacts).
- `bench-history.csv` schema is now a commit-stamped trail (date/commit/branch).

## [0.5.0] - 2026-06-08

Portability & robustness (roadmap M4).

### Added
- **aarch64 validation**: the model cross-builds (`cyrius build --aarch64`) and
  runs under `qemu-aarch64`; the grad-check suite passes on aarch64 (where
  `f64_exp`/`f64_ln` are polyfills and the FMA is a fused NEON `fmla`), and a
  250-step training run matches x86_64 to display precision. New CI lane
  cross-builds + qemu-runs the suite and a binary smoke.
- **NaN/inf guard**: `f64_is_finite` (bit-pattern check) + a training-loop guard
  that stops cleanly on a non-finite loss or grad-norm rather than poisoning the
  weights. Covered by `test_nan_guard`.
- **Soak test**: `test_soak` asserts `alloc_used()` is identical across many
  steps (no per-step allocation → no leak) and that training reaches its target
  step (loss stayed finite).
- **Crash-atomic checkpoint save**: `secure_write_atomic` writes to a `.tmp`
  sibling (`O_NOFOLLOW`), `fsync`s, then `rename`s over the target; a failed
  write/fsync aborts and cleans up, leaving the prior checkpoint intact.
- Arch-aware test tolerances: attention grad checks stay 1e-5 on x86 and relax
  to 1e-4 on aarch64 (exp-polyfill finite-difference noise); `test_simd_contract`
  asserts bit-exact on x86 / within-rounding on aarch64 (fused FMA). **52 checks.**

### Changed
- Toolchain pin `6.1.5`.

## [0.4.0] - 2026-06-08

Performance — SIMD matmul (roadmap M3). **~2.27× faster training** (1939 →
4396 tokens/sec on the default config), gradients and convergence unchanged.

### Added
- **4-wide SIMD vectorization** of the matmul hot paths (`linear_fwd`,
  `linear_bwd`, and the attention per-head score/AV/`dQ`/`dK`/`dV` loops) via
  the packed `f64v_fmadd` builtin with memory accumulators and a scalar tail for
  non-multiple-of-4 dims. `linear_fwd` alone is **3.88×**; a full fwd+bwd step
  **2.27×**.
- Real benchmark harness `tests/attn11.bcyr` (now_ns timings of linear /
  forward / backward / step / Adam + tokens/sec); results in
  [`docs/benchmarks.md`](docs/benchmarks.md) and tracked in
  [`bench-history.csv`](bench-history.csv).
- Grad checks now cover head dims 6/8/10 (the production `hd=8` two-chunk SIMD
  path) and `test_simd_contract` pins `f64v_fmadd` == scalar `mul+add` so a
  future toolchain emitting a fused single-rounding FMA is caught. **47 checks.**

### Changed
- Toolchain pin `6.1.5`.

### Notes
- The vectorization is numerically faithful: AXPY paths (forward `y`, `dW`,
  attention AV/`dQ`/`dK`/`dV`) are **bit-identical** to scalar on this toolchain
  (verified); the dot paths (`dx`, attention scores/`dP`) use a 4-lane tree
  reduction and so differ only at floating-point rounding (~1e-16, far within
  the grad-check tolerance). All 47 grad checks pass unchanged.

## [0.3.0] - 2026-06-08

Data & persistence (roadmap M2).

### Added
- **Corpus loading** from a file (`--corpus PATH`) or stdin (`--stdin`), with a
  byte-level tokenizer that adapts its vocab to whatever bytes occur. Opens with
  `O_NOFOLLOW` and caps size via `fstat` (4 MB) before reading. Falls back to
  the embedded corpus when no source is given. (`src/fileio.cyr`,
  `corpus_set`/`corpus_load_file`/`corpus_load_stdin` in `src/train.cyr`.)
- **Checkpoints** (`src/persist.cyr`): `--save PATH` writes magic + version +
  config + vocab + params + Adam moments + PRNG state + step; `--load PATH`
  validates and restores. Validation (magic, version, config ranges, recomputed
  param count, exact size) happens **before any allocation** — hostile inputs
  are rejected, not crashed.
- **Deterministic resume**: resumable training via a global step counter and a
  fixed schedule horizon, so `train(N)` == `train(K)` → checkpoint → `train(N)`
  **bit-for-bit**. Verified by `test_resume_determinism`.
- **CLI flags**: `--corpus`, `--stdin`, `--load`, `--save`, `--steps`,
  `--gen-only` (argv parsing via stdlib `args`).
- **Fuzz harness** (`tests/attn11.fcyr`): 500 mutated-checkpoint rounds
  (truncated / bit-flipped / wild-config / random) + 100 random corpora; loaders
  must reject without crashing. Plus a `test_ckpt_reject` smoke in the suite.
- `SECURITY.md` updated for the new file surfaces.

### Changed
- Toolchain pin `6.1.5`.
- `train` is now `train(run_until, total, batch, base_lr, min_lr, warmup, clip,
  log_every)` and advances the global `g_step` (enables resume); the LR
  schedule uses the fixed `total` horizon.
- `stdlib` deps gain `args`.

### Fixed
- `ckpt_load_file` allocates the exact (capped) file size instead of the 2 GB
  max bound, which the bump allocator couldn't satisfy.
- Generation prompts containing bytes absent from the vocab no longer produce a
  negative token id (out-of-bounds embedding lookup) — they fall back to id 0.

### Hardened (adversarial review)
- **Resume corpus/vocab consistency**: loading a checkpoint with a corpus whose
  byte-level vocab differs from the checkpoint's is now a hard error (`-15`)
  instead of silently mis-indexing the encoded corpus against the restored
  embeddings.
- **Dropout validation**: the checkpoint's dropout field is bit-pattern-checked
  to a finite `[0,1)` before use (rejects `NaN`, `±Inf`, `>=1.0`, negatives →
  `-14`). Validated against the bit pattern because this toolchain's f64
  comparisons are not NaN-correct.
- **I/O errors vs EOF**: `secure_read_file` / `ckpt_load_file` / `read_stdin`
  now propagate a negative `read(2)` error instead of treating it as a clean
  short read; oversize stdin (> cap) is rejected (`-2`) like the file path.
- **Sampler index**: `sample_window` masks the sign bit instead of negating
  (negating `INT64_MIN` stayed negative → potential OOB corpus read).
- A note is printed when sampling from an untrained model.
- Documented that checkpoint save is non-atomic (atomic save deferred to M4).

## [0.2.0] - 2026-06-08

Depth & training quality (roadmap M1).

### Added
- **Stacked transformer blocks** — configurable `n_layers`. The model now runs
  a residual stream through `N` pre-norm blocks; parameters are packed per-layer
  at a computed block stride, and forward/backward generalize over the stack.
- GPT-2 **residual-projection init scaling** (`1/sqrt(2·n_layers)`) on the
  attention output and MLP projection weights.
- **Gradient clipping** by global L2 norm (`model_clip_grads`).
- **LR schedule** — linear warmup then cosine decay to a floor (`lr_at`, with a
  Taylor cosine since Cyrius has no `f64_cos`).
- **Config-gated attention biases** (`bq/bk/bv/bo`) threaded through
  `attn_fwd`/`attn_bwd`; appended to the block layout only when enabled
  (default on). The MLP already carried biases.
- **Config-gated residual dropout** (inverted, per-branch masks) on the
  attention and MLP outputs; auto-disabled outside training (`g_training`) so
  generation and grad checks stay deterministic. Default off (tiny corpus).
- Full-model gradient check extended to a 2-layer model with biases (both
  blocks, attention bias, shared embeddings, head, final LayerNorm); added
  standalone dropout and attention-bias grad checks. **20 checks total.**

### Changed
- `model_init` now takes `n_layers`; `train` takes `(steps, batch, base_lr,
  min_lr, warmup, clip, log_every)`. Per-layer weights/grads are addressed via
  `PL`/`GL` + `_o_*` offset helpers (replacing the single-block `P_*`/`G_*`
  globals); shared weights via `P_tokemb()`/`P_posemb()`/`P_lnfg()`/`P_lnfb()`.
- Default config is now 3 layers; training logs `lr` and grad-norm per checkpoint.
- Toolchain pin `6.1.3` → `6.1.4`.

## [0.1.0]

### Added
- A from-scratch, dependency-free **GPT-style transformer that trains** in
  Cyrius — forward pass, hand-derived backprop, and Adam, all on raw `f64`
  arrays (no BLAS, no libc, no autodiff).
- `src/tensor.cyr` — f64-array helpers, deterministic PRNG (xorshift64 +
  splitmix seeding, Marsaglia-polar normal sampling), float printing.
- `src/ops.cyr` — linear, LayerNorm, GELU (tanh approx), softmax cross-entropy
  (forward + backward).
- `src/attn.cyr` — causal multi-head self-attention (forward + backward) over a
  single pre-allocated arena.
- `src/model.cyr` — packed parameter vector, token/positional embeddings,
  weight-tied LM head, full forward/backward wiring, Adam optimizer.
- `src/train.cyr` — char-level tokenizer, corpus, batch sampling, training
  loop, autoregressive generation (greedy + temperature sampling).
- `tests/attn11.tcyr` — 16 finite-difference gradient checks (every op,
  attention, and the full model); all passing.
- Architecture notes in `docs/architecture/001-tensors-and-floats.md`.
- Roadmap to v1.0 with versioned milestones and acceptance gates
  (`docs/development/roadmap.md`).
- CI/release process aligned to the patra/sigil model: lint warn-gate, DCE
  build + ELF verify, gradient-check suite, fuzz, benchmarks, security scan,
  and a docs/version-consistency gate; release builds the binary + source
  tarball + `SHA256SUMS`, extracts the changelog, and marks `0.x` prereleases.
- `CONTRIBUTING.md`, `SECURITY.md`, and `scripts/version-bump.sh`.

### Changed
- Toolchain pin `6.1.2` → `6.1.3`.
- `cyrius.cyml` package version now derives from `VERSION` via `${file:VERSION}`
  (VERSION is the single source of truth).
- Corpus is assembled from short segments at runtime (keeps src lint-clean
  under the 120-column gate).
