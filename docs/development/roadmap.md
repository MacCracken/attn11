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
      [`benchmarks.md`](../benchmarks.md), with a tracked history.
- [ ] **Portability** — builds and trains on Linux x86_64 **and** aarch64, and
      runs under the **AGNOS kernel** (`--agnos`); results match within tolerance.
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

### M1 — Depth & training quality (v0.2.0) — ✅ shipped 2026-06-08

- ✅ **Stacked blocks**: configurable `n_layers`; the block became a reusable
  forward/backward over a per-layer cache stack (residual stream array).
- ✅ GPT-2 residual-projection init scaling (`1/sqrt(2·n_layers)`).
- ✅ Attention biases (config-gated; MLP biases already present); residual
  dropout (config-gated, auto-disabled in eval).
- ✅ Gradient clipping (global-norm); LR warmup + cosine decay.
- **Gates met**: per-op grad checks unchanged; full-model grad check green at
  `n_layers = 2` with biases; 20 grad checks total; the 3-layer default trains
  to ~0.11 loss, below M0's single-layer ~0.15.

### M2 — Data & persistence (v0.3.0) — ✅ shipped 2026-06-08

- ✅ Corpus loading from a file/stdin (`O_NOFOLLOW`, `fstat` size cap, looped
  reads) with fallback to the embedded string.
- ✅ Byte-level tokenizer with an adaptive vocab. (Simple BPE remains optional
  / deferred — byte-level covers the gate.)
- ✅ **Checkpoint save/load** of the flat parameter vector + Adam moments + RNG
  state + step, with header validation before any allocation.
- **Gates met**: `train(N)`→save→load→`train` reproduces params bit-for-bit
  (`test_resume_determinism`); the loaders have a fuzz harness (500 mutated
  checkpoints + 100 random corpora); `SECURITY.md` updated for the new surface.

### M3 — Performance (v0.4.0) — ✅ shipped 2026-06-08

- ✅ 4-wide SIMD (`f64v_fmadd`) on the matmul hot paths — `linear_fwd`/
  `linear_bwd` and the attention per-head score/AV/`dQ`/`dK`/`dV` loops. (Adam
  is <1% of a step; GELU's `f64_tanh` has no packed form — both left scalar.)
- ✅ Benchmarks in [`benchmarks.md`](../benchmarks.md) + [`bench-history.csv`](../../bench-history.csv).
- **Gates met**: documented `before → after` — `linear_fwd` 3.88×, fwd+bwd step
  2.27×, **tokens/sec 1939 → 4396**; grad checks unchanged (47 pass, incl. the
  production `hd=8` path) and the SIMD is bit-identical (axpy) / within-rounding
  (dot) to scalar, so training converges identically.

### M4 — Portability & robustness (v0.5.0) — ✅ shipped 2026-06-08

- ✅ Validated on **aarch64** via cross-build + qemu: grad checks pass (with
  arch-aware tolerances for the `f64_exp` polyfill and the fused NEON FMA), and
  a 250-step training run matches x86_64 to display precision. New CI lane.
- ✅ NaN/inf training guard (stops cleanly, doesn't poison weights); soak test
  proving no per-step allocation (no leak) and no blow-up.
- ✅ Crash-atomic checkpoint save (temp + fsync + rename; prior checkpoint
  preserved on any failure) — the item deferred from M2.
- **Gates met**: aarch64 CI lane (cross-build + qemu) green; soak clean; both
  arches pass 52 checks. (Native aarch64 hardware runs are emulated via qemu.)

### M5 — Portability: AGNOS kernel (v0.6.0)

Adapt attn11 to run as a ring-3 application under the **AGNOS kernel** (the
sovereign OS Cyrius writes), not just Linux. The stdlib already ships the target
layer (`syscalls_x86_64_agnos.cyr`, `alloc_agnos.cyr`, …); `cyrius build --agnos`
selects it via `CYRIUS_TARGET_AGNOS`.

- **De-Linux the syscalls**: replace the raw Linux syscall numbers baked into
  the source (`syscall(60)` exit, `syscall(1, 1, …)` write in `_putc`/`fmt`
  paths) with the portable stdlib wrappers that dispatch per target — so the
  same source compiles for Linux, aarch64, and AGNOS.
- Build `cyrius build --agnos src/main.cyr` clean; resolve any
  AGNOS-userland-ABI gaps (see `agnos/docs/development/agnos-userland-abi.md`).
- Confirm the AGNOS allocator path (`alloc_agnos.cyr`) backs the one-shot
  arena allocation the training loop relies on.
- **Gates**: `--agnos` build is green in CI; the binary trains + samples under
  AGNOS and reproduces the Linux run bit-for-bit for a fixed seed; grad-check
  suite builds for AGNOS. Document the AGNOS build/run in a guide.

### M6 — Freeze & consumer (v0.9.0 → v1.0.0)

- Freeze the config/CLI surface; finalize docs (guides + a runnable example).
- Land one downstream consumer or example pipeline against a tagged build.
- Security audit (`docs/audit/`); tag **v1.0.0** (first non-prerelease).

## Out of scope (for v1.0)

- GPU / accelerator backends — attn11 is a CPU, scalar-f64 (then SIMD)
  reference implementation.
- Distributed / multi-process training.
- A general autodiff engine — gradients stay hand-derived and grad-checked.
- Windows / macOS as first-class training targets (cross-build only, if at all).
