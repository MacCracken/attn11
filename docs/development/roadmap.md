# attn11 — Roadmap

> Milestone plan through v1.0. State lives in [`state.md`](state.md);
> this file is the sequencing — what ships, in what order, against
> what gates. Every milestone keeps the invariant: **all backward passes
> stay grad-checked** (`cyrius test` green) and **src lints clean**.

## Versioning

`VERSION` is the single source of truth (`cyrius.cyml` derives it via
`${file:VERSION}`). Bumps go through `scripts/version-bump.sh`. While the major
is `0`, releases are cut as **prereleases** (the release workflow sets this
automatically for `0.x` tags). v1.0.0 is the first stable, API/CLI-frozen tag.

## v1.0 criteria

v1.0.0 ships when **all** of these hold:

- [ ] **Frozen surface** — model config + CLI flags documented and stable; no
      planned breaking changes.
- [ ] **Correctness** — per-op and full-model gradient checks green for the
      final architecture (including stacked blocks); a long soak run shows no
      NaN/inf and no memory growth.
- [ ] **Coverage** — tests for tokenizer, checkpoint round-trip, and each op;
      fuzz harnesses for every external input path.
- [ ] **Benchmarks** — step time + tokens/sec captured in
      [`benchmarks.md`](benchmarks.md), with a tracked history.
- [ ] **Portability** — builds and trains on x86_64 **and** aarch64; results
      match within tolerance.
- [ ] **One consumer green** — at least one downstream user (or a documented
      example pipeline) builds against a tagged attn11.
- [ ] **CHANGELOG complete** from 0.1.0 onward; **security audit** recorded in
      `docs/audit/YYYY-MM-DD-audit.md`.

## Milestones

### M0 — Working model (v0.1.0) — ✅ shipped 2026-06-08

- One pre-norm Transformer block trains end-to-end: token+positional
  embeddings → LayerNorm → causal multi-head self-attention → residual →
  LayerNorm → GELU MLP → residual → final LayerNorm → weight-tied LM head →
  softmax cross-entropy.
- Hand-derived backprop + Adam; char tokenizer; greedy + temperature sampling.
- 16 finite-difference gradient checks (every op, attention, full model), green.
- CI/release process aligned to the patra/sigil model.

### M1 — Depth & training quality (v0.2.0)

- **Stacked blocks**: configurable `n_layers`; refactor the block into a
  reusable forward/backward over a per-layer cache stack.
- GPT-2 residual-projection init scaling (`1/sqrt(2·n_layers)`).
- Attention + MLP biases (optional, config-gated).
- Gradient clipping (global-norm); LR warmup + cosine decay.
- **Gates**: per-op grad checks unchanged; full-model grad check green with
  `n_layers ≥ 2`; deeper model trains to lower loss than M0 on the same corpus.

### M2 — Data & persistence (v0.3.0)

- Corpus loading from a file/stdin (validated: size cap, `O_NOFOLLOW`,
  bounds-checked reads) instead of only the embedded string.
- Larger vocab; optional byte-level / simple BPE tokenizer.
- **Checkpoint save/load** of the flat parameter vector (+ optimizer moments +
  RNG state) for deterministic resume.
- **Gates**: train→save→load→resume reproduces the loss curve bit-for-bit;
  every loader has a fuzz harness; `SECURITY.md` updated for the new surface.

### M3 — Performance (v0.4.0)

- SIMD (`f64v2`/`f64v4`) hot paths for matmul / attention / Adam.
- Benchmarks captured in [`benchmarks.md`](benchmarks.md) with a
  `bench-history.csv` (step time, tokens/sec, build size).
- **Gates**: documented `before → after` speedup per bench; grad checks and
  training curve unchanged within tolerance.

### M4 — Portability & robustness (v0.5.0)

- Validate on **aarch64** (where `f64_exp`/`f64_ln` are polyfills): grad checks
  and a short training run must match x86_64 within tolerance.
- NaN/inf guards on loss; a soak target (long run) proving no leak / no blowup.
- **Gates**: cross-build + native aarch64 CI lane green; soak run clean.

### M5 — Freeze & consumer (v0.9.0 → v1.0.0)

- Freeze the config/CLI surface; finalize docs (guides + a runnable example).
- Land one downstream consumer or example pipeline against a tagged build.
- Security audit (`docs/audit/`); tag **v1.0.0** (first non-prerelease).

## Out of scope (for v1.0)

- GPU / accelerator backends — attn11 is a CPU, scalar-f64 (then SIMD)
  reference implementation.
- Distributed / multi-process training.
- A general autodiff engine — gradients stay hand-derived and grad-checked.
- Windows / macOS as first-class training targets (cross-build only, if at all).
