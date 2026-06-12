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
- [x] **Portability** — builds and trains on Linux x86_64 **and** aarch64, and
      runs under the **AGNOS kernel** (`--agnos`); results match within tolerance
      (aarch64: display-precision; AGNOS: bit-for-bit at fixed CPU — M5).
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

### M5 — Portability: AGNOS kernel (v0.6.0) — ✅ shipped 2026-06-11

- ✅ **De-Linuxed every raw syscall site** (exit/write in `_putc`, harness
  epilogues, and the fileio fstat/fsync/unlink/rename Linux-isms) behind
  portable wrappers + `#ifdef CYRIUS_TARGET_AGNOS` bridges — one source tree
  compiles for Linux x86_64, Linux aarch64, and AGNOS.
- ✅ `cyrius build --agnos` clean for the binary AND the grad-check suite
  (static x86_64 ELF64); new build-only `agnos` CI lane. The agnos allocator
  (`alloc_agnos.cyr`, chunked mmap bump) backs the one-shot arena fine.
- ✅ Toolchain pin `6.1.6` → `6.1.31` (old pin predated the 6.1.13/6.1.14
  HIGH-sev agnos codegen fixes); found + worked around a remaining cycc gap
  (argv capture emitted after call-bearing gvar inits → statement-call entry
  epilogues; filed upstream, `docs/architecture/002`).
- ✅ `scripts/agnos-smoke.sh` — boots the real kernel (gnoboot + agnos 1.44.15
  + agnsh 1.6.x) in QEMU, drives `run /bin/attn11 --steps N --save /ck.ckpt`
  over the emulated keyboard, extracts the checkpoint from the ext2 image.
- **Gates met**: the binary **trains + samples under AGNOS**; the saved
  checkpoint is **bit-for-bit identical** to the Linux run (948,008 bytes,
  fixed seed, CPU implementation held constant — Linux reference under
  `qemu-x86_64`, mirroring the aarch64 method); 1000-step loss/lr/grad-norm
  serial output matches native Linux to every displayed digit; grad-check
  suite builds for AGNOS; documented in `docs/guides/agnos.md`.

### M6 — Inference efficiency (v0.7.0) — ✅ shipped 2026-06-11

Frontier-track items E1 + E2 graduated (both freeze-safe and additive; landing
them before the M7 freeze means the frozen surface includes them).

- ✅ **E1 — KV-cached generation**: per-layer K/V caches + a single-row cached
  forward; generation runs one row per token. Learned absolute positional
  embeddings pin cached rows to positions, so a full window context-shifts
  (drop oldest T/2, re-prime) instead of sliding per token (ADR 0005).
- ✅ **E2 — GQA/MQA**: `n_kv_heads` config; K/V projections at `C × Ckv`;
  checkpoint v2 (+`nkv` header field, v1 still loads); default `nkv = nh`
  keeps training bit-identical to 0.6.0.
- **Gates met**: cached and uncached generation **bit-identical** (every
  prefix, across context-shifts, greedy + temperature, hd ∈ {4,6,8,10} ×
  nkv ∈ {1,2,nh}); generation **6.2× faster** (951 → 5 868 tok/s, default
  config, x86_64); GQA grad-checked at `nkv ∈ {1, 2, nh}` + full-model
  `nkv < nh` + MQA resume-determinism; KV bytes accounted in the bench
  (24 576 → 6 144 at `nkv = 1`); 161 checks green on x86_64 and
  aarch64/qemu; the AGNOS run gate **PASS** (bit-for-bit checkpoint under
  the booted kernel, pin 6.1.33); adversarial multi-agent review passed
  (9 confirmed findings, all fixed — `docs/audit/2026-06-11-kv-gqa-audit.md`).
- Bonus discovery: the K bias has exactly zero gradient (softmax
  shift-invariance) — pinned by a dedicated check.

### M7 — Scale preset + BPE (v0.7.1) — ✅ shipped 2026-06-11 (frontier E3 graduated)

Straight from what the vidya run (X001) exposed — small, additive, freeze-safe:

- ✅ **`--preset`** for ctx 64 / d_model 64 (8 heads, 4 layers; the other
  shapes reachable via new `--heads`/`--kv-heads`/`--layers` overrides).
  Re-benched as predicted: the context-shift re-prime amortizes over 32
  tokens at T=64, so KV-cached generation widens from 6.2× to **23×** over
  uncached (64 → 1 486 tok/s).
- ✅ **Simple BPE tokenizer** (`--bpe K`, K ≤ 512; revisits ADR 0002 → ADR
  0006; byte-level stays the default). Pure-i64 deterministic merge training
  (frozen tie-break, pinned cross-arch by an exact-merge-sequence test);
  decode via a precomputed span table (no recursion); ~110 ms one-shot cost
  at 256 KB / K=128.
- ✅ **Checkpoint v3** records the tokenizer (kind + base vocab + merge
  table); v1/v2 still load (byte-level); the loader validates hostile merge
  tables as a well-founded DAG with bounded expansion (-37/-38) before any
  allocation — same discipline as v1→v2.
- ✅ **`--eval`** (deterministic, RNG-neutral): CE/token + **bits-per-byte**,
  the tokenizer-comparable metric for X003.
- **Gates met**: grad checks + the bit-identity gate green at the preset
  config (and at V=300, past the old byte cap) — 242 checks on x86_64 AND
  aarch64/qemu; checkpoint v3 with v1/v2 compat covered by round-trip +
  rejection tests and 500 dedicated fuzz rounds (merge-slot clobber, triple
  inconsistency, expansion bombs); byte-vs-BPE at iso-compute logged as
  **X003** in [`experiments.md`](experiments.md). An adversarial review of the
  diff caught + fixed a `--layers` heap-OOB (the fresh-config path had skipped
  the loader's alloc cap) — see [`../audit/2026-06-11-m7-bpe-audit.md`](../audit/2026-06-11-m7-bpe-audit.md).

### M8 — Security sweep (v0.8.0) — ✅ shipped 2026-06-11

A research-driven hardening release — surveyed the world first, then repaired.
Six vulnerability classes web-researched against recent CVEs, then
adversarially mapped onto attn11's surfaces (survey→map workflow, 12 agents);
full per-class dispositions in
[`../audit/2026-06-11-m8-security-sweep-audit.md`](../audit/2026-06-11-m8-security-sweep-audit.md).

- ✅ **Headline (negative) result**: the checkpoint is a flat native-endian
  i64 array — no opcode interpreter, callable revival, or embedded path — so
  it is **structurally immune** to the pickle/Keras/numpy model-file
  deserialization-RCE genre. Integer-overflow, size-vs-shape, alloc-bomb, and
  merge-DAG vectors all confirmed mitigated against their exact code guards.
- ✅ **Two real bugs fixed**: a dropped `_file_size` path arg that crashed
  every AGNOS `--load` (`strlen` on a garbage register), and — exposed by the
  new file-path coverage — checkpoint *save* broken on the whole aarch64 lane
  (qemu mis-emulates `fsync`; switched the durability barrier to `fdatasync`).
- ✅ **Hardening**: `_atoi` overflow saturation; the `lens` merge-scratch
  pinned to its buffer; **CI supply-chain** — every GitHub Action SHA-pinned,
  the `GITHUB_REF_NAME` awk-injection closed, `contents: write` scoped to the
  release job. Deferred infra-bound items (installer pinning, artifact
  signing, `lib/` lockfile) documented with rationale.
- ✅ **Coverage**: `test_ckpt_file_roundtrip` (the file loader was untested),
  the AGNOS run gate now `--load`s, two new fuzz boundary modes.
- **Gates met**: 247 checks green on x86_64 AND aarch64/qemu (incl.
  checkpoint save+load on both); fuzz extended; all repairs regression-tested;
  the dated CVE-survey audit with per-class dispositions is in `docs/audit/`.

### M9 — Performance (v0.8.x)

The accumulated perf levers, landed one at a time against
[`bench-history.csv`](../../bench-history.csv) (numbers or it didn't happen):

- ✅ **Vectorize the tied LM head** — **0.8.1**: `head_fwd_row` scalar → 4-wide
  `f64v_fmadd`, **2.7×** at V=768 (`head_fwd` 9.7 → 3.59 ms); win scales with
  the vocab (negligible at V=25, ~17% of the forward at BPE-scale). Bit-identity
  gate unaffected (shared kernel); a C=6 tail test (mutation-verified) covers
  the new `C % 4 ≠ 0` path. See [`benchmarks.md`](../benchmarks.md).
- ❌ **Packed `tanh` for GELU** — profiled, **NOT shipped** (X004): the exact
  one-exp `tanh` is only ~15% faster per call (noisy) — `f64_exp` is cheap on
  this toolchain — and GELU is ~8% of a step, so the win is ~1–2%, below
  step-bench noise.
- ❌ **Matmul cache-blocking / register-tiling** — profiled, **NOT shipped**
  (X004): an m-blocked `linear_fwd` is bit-identical but **~15% slower** at the
  preset shape — attn11's weight matrices are cache-resident, so blocking only
  adds accumulator traffic. (Attention Q·K/P·V were already 4-wide; the head
  was the lone scalar kernel.)
- ❌ **Batched prefill** — prototyped, **NOT shipped** (X004): a batched n-row
  window forward vs the `keep` single-row re-prime benched **~1% (within
  noise)** at n=32. The re-prime's cost is irreducible *work* (the n-row causal
  forward — same MACs either way), not per-call overhead, so batching saves
  nothing at this scale.
- **Gates per item**: documented before → after in
  [`benchmarks.md`](../benchmarks.md) + the CSV; grad checks and the
  bit-identity gate stay green (`docs/architecture/003` — kernel changes
  must land in BOTH forward paths or neither).

**M9 concluded (0.8.1).** The LM head was the one clean win; all three other
levers were measured and rejected (GELU-tanh marginal, matmul-blocking slower,
batched prefill no win — X004). The residual matmul gap to SIMD peak is
structural (the SIMD-var-reassign rule + 2-wide `f64v_fmadd` lowering) and
needs toolchain support, not a v0.8.x code change. Next: **M10**.

### M10 — Freeze, docs & cleanup (v0.9.0) — ✅ shipped 2026-06-11

Everything required so the v1.0.0 cut is a tag, not a scramble:

- ✅ **Froze the config/CLI surface** — [`STABILITY.md`](../STABILITY.md):
  12 additive-only CLI flags, the compile-time `CFG_*` knobs with their frozen
  values, checkpoint v3 with permanent v1/v2 load-compat, the magnitude caps,
  and an explicit not-in-the-contract list. The CLI gained `--help`/`--version`
  and now rejects unknown args + missing flag values (CI-gated `--version`).
- ✅ **Docs audit** (5-dimension multi-agent sweep): ADR 0005 perf figure,
  note 001 SIMD-head listing, the missing dropout citation, benchmarks↔X004,
  the 0006↔0002 bidirectional link, README/getting-started caps. Pin moved
  6.1.34 → **6.1.37** (`lib/` resynced).
- ✅ **Vidya example pipeline** — [`examples/vidya-pipeline.md`](../examples/vidya-pipeline.md):
  preset + the 488 KB corpus → loss **1.089** (4000 steps), bits/byte **1.760**
  → ~5 MB checkpoint (reloads bit-for-bit) → sample, with a BPE variant.
- ✅ **Cleanup**: dead code removed (`secure_write_file`/`f_println_lbl`/
  `CFG_NKV`); no TODOs/FIXME remain in `src`/`tests`/`docs`. Loose end left:
  the cross-repo `agnos/scripts/stage-tools.sh` attn11 row (maintainer's edit).

### v1.0.0 — the clean cut — ✅ ready 2026-06-11

Every v1.0 criterion held. The final audit — 5 adversarial dimensions
(hostile-input, math correctness, memory safety, frozen surface, release
integrity) — returned **go on all five, zero blockers**
([`../audit/2026-06-11-v1.0-final-audit.md`](../audit/2026-06-11-v1.0-final-audit.md));
release-hygiene fixes (`--save` exit code, `version-bump.sh` CFG_VERSION,
SECURITY wording) landed. 248 tests green on x86_64 + aarch64, all gates pass,
version surfaces consistent. Cut as **v1.0.0** — the first non-prerelease. No
features ride this tag; the surface is frozen ([`../STABILITY.md`](../STABILITY.md)).
Anything past here is v1.x (additive) or the v2-track frontier below.

## Beyond v1.0 — the frontier track (experiments)

> Informed by the June-2026 frontier survey
> (`ai-ml-frontier-2026-expanded.docx`, repo root): data quality > volume,
> the KV cache as the central inference object, SSM/attention hybrids
> displacing pure transformers, diffusion LMs as a maturing generation
> paradigm, and the precision ladder (FP8→FP4→ternary). attn11 adapts by
> building **reference implementations of the ideas**, not by chasing the
> hardware: every item below keeps the project invariants — hand-derived
> backward, finite-difference grad checks, CPU f64, no deps. Results are
> logged in [`experiments.md`](experiments.md); an experiment graduates into
> a milestone only when it earns one.

Ordered by (value ÷ risk), each independently shippable:

- ~~**E1 — KV-cached generation.**~~ ✅ Graduated into **M6 (v0.7.0)** — gates
  met (bit-identity + 6.2× tokens/sec).
- ~~**E2 — GQA/MQA config.**~~ ✅ Graduated into **M6 (v0.7.0)** — gates met
  (grad checks at `n_kv_heads ∈ {1, 2, nh}`; KV bytes in the bench).
- ~~**E3 — Scale preset + BPE.**~~ ✅ Graduated into **M7 (v0.7.1)** — gates
  met (preset bit-identity + 23× cached generation; BPE deterministic
  cross-arch; byte-vs-BPE at iso-compute logged as X003).
- **E4 — A second sequence-mixer family (linear attention → selective SSM).**
  The survey's structural shift: hybrids with ~10–25% attention layers beat
  pure transformers. Ladder: (a) a gated linear-attention/RetNet-style block
  (easiest hand-derivable backward — constant state, linear compute), then
  (b) a minimal selective-SSM block (BPTT through the scan), then (c) a
  per-layer `kind` config to interleave mixer types and sweep the **hybrid
  ratio** at our scale. Gate: every new backward grad-checked; perplexity vs
  iso-param transformer on the vidya corpus.
- **E5 — Char-diffusion objective (dLLM at reference scale).** Same trunk,
  different training model: masked-denoising objective (drop the causal
  mask, predict masked bytes, confidence-aware parallel decode). Tests the
  survey's "diffusion LMs are super data learners" thesis honestly at tiny
  scale: AR vs diffusion on repeated epochs of the same small corpus. Gate:
  grad check the masked CE path; matched-compute comparison logged.
- **E6 — Ternary (BitNet-style) training.** Weights in {−1, 0, +1} with a
  straight-through estimator — the precision ladder's algorithmic endpoint,
  and a *natural fit* for an everything-is-i64 language: ternary matmul
  collapses to integer adds. Gate: STE backward documented + grad-checked
  where defined; accuracy vs f64 baseline; the i64-add matmul benched
  against SIMD f64.

Sequencing intent: E1–E2 shipped pre-freeze as M6 (v0.7.0); E3 shipped as M7
(v0.7.1). The remaining pre-freeze ladder is fixed: **0.8.0 security sweep →
0.8.x performance → 0.9.0 freeze/docs/cleanup → v1.0.0 clean cut** (M8–M10
above). E4–E6 are v2-track (new model families), may fork the architecture
config, and do not ride any 0.x release.

## Out of scope (for v1.0)

- GPU / accelerator backends — attn11 is a CPU, scalar-f64 (then SIMD)
  reference implementation. (The survey's engine/serving layer — vLLM/SGLang,
  FP4 tensor cores, photonics — is *observed, not chased*; only its
  algorithmic ideas translate here.)
- Distributed / multi-process training.
- A general autodiff engine — gradients stay hand-derived and grad-checked.
- Windows / macOS as first-class training targets (cross-build only, if at all).
