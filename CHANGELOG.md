# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
- Architecture notes in `docs/architecture/tensors-and-floats.md`.
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
