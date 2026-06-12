# attn11 — Roadmap

> Milestone plan through v1.0 and the 1.x architecture arc (M11+). State lives
> in [`state.md`](state.md);
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
Anything past here is v1.x (additive) — the architecture arc below (M11+); there
is no v2 fork.

### M11 — The extraction (v1.1.0) — ✅ shipped 2026-06-12

The first post-1.0 minor — additive/internal, the frozen surface untouched. The
reusable numeric core was lifted out of attn11 into two sovereign sibling
libraries, which attn11 now consumes and dogfoods:

- ✅ **Tensor storage + BLAS-1 + dense matmul (and its gradient) →
  [rosnet](https://github.com/MacCracken/rosnet) 0.1.0** — `t_alloc`/`t_zero`/
  `tget`/`tset`, `t_axpy`/`t_scale`/`t_sum`, `f64_is_finite`, `t_randn`, and
  `linear_fwd`/`linear_bwd`. Matmul + its backward are pure linear algebra
  (`dx = dy·Wᵀ`, `dW = xᵀ·dy`), reusable beyond ML — so they belong in the
  tensor lib, not the model.
- ✅ **Deterministic statistical PRNG →
  [tyche](https://github.com/MacCracken/tyche) 0.1.0** — `rng_seed`/`rng_u64`/
  `rng_uniform`/`rng_normal` + the `_rng_state` stream (attn11 still reads/writes
  it directly for crash-atomic save + bit-for-bit resume).
- ✅ Model-specific differentiable layers (LayerNorm/GELU/dropout/softmax CE)
  stay in `src/ops.cyr`; `src/tensor.cyr` keeps only attn11-local float printing.
- **Gates met**: byte-identical training/sampling, same CLI, v3 checkpoint
  unchanged, **248** grad-check/property tests green — attn11's grad-checks now
  double as rosnet's gradient validation and tyche's RNG-state validation. Still
  **no BLAS/libc/autodiff** (both libs are pure-Cyrius `f64`-in-`i64`,
  sovereign-ecosystem). attn11 is the **reference consumer**, satisfying the
  v1.0 "one consumer green" criterion. Pin held at 6.1.37.

## The 1.x architecture arc (M12+)

Past 1.0 the surface is **additive-only**, so the frontier experiments below
ship as opt-in 1.x minors — **not** a v2 fork. Each adds new `--flags` and a new
checkpoint version (prior versions always load), leaves the default run
byte-identical, and keeps every invariant: hand-derived backward,
finite-difference grad-checked, CPU `f64`, no deps. This is exactly how
KV-cache/GQA (M6) and BPE (M7) landed additively pre-freeze — the same
freeze-safe pattern, now applied past the 1.0 line. Each milestone graduates a
frontier experiment (E-series below), is independently shippable, and bumps the
checkpoint version with permanent back-compat. The build order is a value÷risk
call, **not** a dependency chain — the axes (attention, FFN, sequence-mixer,
objective, precision) are orthogonal and re-orderable.

### M12 — Multi-Head Latent Attention + positional-encoding switch (v1.2.0) — E7

The **KV-cache evolution**. MLA caches one **low-rank latent** `c_KV` per token
(a down-projection `C → d_c`, `d_c ≪ C`) and up-projects to per-head K/V on
read, instead of caching full per-head K/V (MHA) or shared K/V heads (GQA). It
is the next step on the **shrink-the-KV-cache** axis M6 opened (E1 cache +
E2 GQA), and the survey's "KV cache is the central inference object" thesis made
concrete. Decision recorded in **ADR 0007**.

**Two orthogonal config axes** land here, both opt-in, both defaulting to the
current model so a no-flag run is byte-identical:

- `--attn-kind {mha, gqa, mla}` — the attention/KV variant (`gqa` already
  exists via `--kv-heads`; `mha` is today's default; `mla` is new).
- `--pos-kind {learned, rope, rope-decoupled}` — the positional scheme. attn11
  uses learned **absolute** embeddings (`learned`, the default), which pin
  cached rows to positions and enable the context-shift re-prime (ADR 0005).
  RoPE is **relative** and mutually exclusive with learned-abs (you pick one).
  *Coupled* `rope` rotates per-head K directly — simplest, but in MLA it breaks
  the up-projection absorption, so it forfeits part of the cache win. *Decoupled*
  `rope-decoupled` (DeepSeek-V2) carries position on a separate small dimension
  that bypasses compression — the faithful, cache-efficient MLA combo.

**Staged increments** (each additive, each its own grad-check / bit-identity
gate; ONE change at a time):

1. ✅ **Descriptor scaffolding** (1.1.x groundwork) — checkpoint **v4** reserves
   `attn_kind`, `pos_kind`, `latent_dim` (`d_c`), `rope_dim` (`d_rope`),
   defaulting to `mha`/`learned`/`0`/`0`; v1/v2/v3 still load. Pure forward-compat
   ahead of the math; no further format bump across the whole MLA+RoPE ladder.
2. ✅ **MLA core** (**v1.2.0**) — `--attn-kind mla --latent-dim d_c` at
   learned-abs positions: low-rank KV factorization (down `C→d_c`, up `d_c→C`),
   full heads. A shared `attn_core_fwd`/`attn_core_bwd` was extracted so MHA and
   MLA run the identical softmax/PV kernel; the MLA backward composes from
   `linear_bwd` + the core (no novel math). Grad-checked per-op (tight) +
   full-model; checkpoint round-trips; trains (loss ↓) and samples.
3. **Latent KV-cache decode** (**M12.2**, the deferred gate) — a cached single-row
   MLA path that stores the `d_c` latent per token (not full K/V), with the
   cached-vs-uncached **bit-identity** gate and the **KV-cache-bytes table** vs
   GQA/MQA (the headline compression number). 1.2.0 generates MLA via the
   uncached reference path; this adds the inference win on top.
4. **Coupled RoPE** (optional) — `--pos-kind rope` on dense MHA; RoPE has no
   learned params, so the grad-check is just the rotation's backward. Valuable on
   its own; could split into its own milestone.
5. **Decoupled RoPE** (optional) — `--pos-kind rope-decoupled` for MLA, the
   faithful cache-efficient form, built on (4).

- **Gates**: latent down/up-projection backward grad-checked; RoPE rotation
  backward grad-checked; cached vs uncached generation **bit-identical** across
  context-shifts (the M6 KV bit-identity gate, extended per `--attn-kind` /
  `--pos-kind` value); **KV-cache-bytes table** vs GQA/MQA (the headline
  compression number, mirroring X002); perplexity vs iso-param GQA on the vidya
  corpus. Logged as an X-series entry.

### M13 — Mixture of Experts (v1.3.0) — E8

**Sparse FFN.** The dense GELU MLP in each block becomes **N experts** with a
top-`K` router, decoupling parameter count from per-token FLOPs. The deliverable
is the **expert-density sweep** so density is a chooseable knob, not a hardcode.

- New `--experts N` for **N ∈ {1 (dense baseline), 8, 16, 32, 64, 128, 256}**,
  `--expert-topk K` (active experts/token, default top-2), each expert reusing
  the `F = 4·C` width. A linear gate `C → N` + softmax selects the top-K; expert
  outputs are gate-weighted so the router is differentiable through the combine
  weights.
- **The hard grad-check (why this earns a milestone):** the router. Top-K
  selection is discrete — differentiate the soft gate weights (straight-through
  for the hard pick), and the **load-balancing auxiliary loss** (Switch-style,
  to stop expert collapse) gets its own finite-difference check. A router
  backward without a passing grad check is incomplete.
- Deterministic routing: the argmax / top-K tie-break is a **frozen** rule
  (bit-reproducible cross-arch, same discipline as the BPE merge tie-break,
  ADR 0006).
- Checkpoint **v5** (router gate + per-expert weights + `experts`/`topk`); prior
  versions load.
- **Gates**: router + aux-loss backward grad-checked; the **density-sweep
  experiment** (X-series) reports, for each N, bits/byte on vidya, active vs
  total params, and **expert utilization** (routing-entropy / load-balance) — so
  "choose the expert density" is backed by numbers. Honest caveat: at attn11's
  reference scale (C = 32–64, tiny corpus) N = 256 is wildly over-parameterized;
  the value is the **grad-checked reference** that sparse routing learns + the
  sweep's density/quality/utilization curve, not a quality win at this scale —
  the same framing as GQA's value here (X002 #2).

### M14 — A second sequence-mixer family (v1.4.0) — E4
### M15 — Char-diffusion objective (v1.5.0) — E5
### M16 — Ternary (BitNet) training (v1.6.0) — E6

The original frontier ladder, now sequenced as 1.x minors after MLA + MoE — full
specs in the E4–E6 catalog below. These are the biggest architectural departures
(a non-attention mixer, a non-AR objective, sub-`f64` weights), so they ride
later in the arc; each keeps the same opt-in-flag + new-checkpoint-version +
grad-check discipline.

### M17 — Reinforcement learning (v1.7.0) — E9

**Last in the chain, by design.** RL is an orthogonal *training-objective* layer,
not an architecture: it runs on whatever trunk exists (any `--attn-kind` /
mixer / precision), so it graduates after the architecture families are in
place and can fine-tune the best of them. The reference target is **policy
gradient (REINFORCE)** at char scale: sample a rollout from the current policy,
score it with a deterministic reward function, and weight the log-prob gradient
by `(R − b)` (advantage, `b` a moving-average baseline). The gradient
`∇ log π(a)·(R − b)` **reuses the existing softmax-CE backward** reweighted per
token — so the hand-derived backward is a small, grad-checkable delta over the
supervised path, exactly attn11's wheelhouse. Reward is a simple in-process
function at this scale (e.g. "is the sample valid-Cyrius / matches a target
pattern / hits a length target"), not a learned reward model. Gate: the
reward-weighted backward grad-checked against finite differences; a documented
RL-vs-SFT comparison (does the policy move toward the reward) logged as an
X-series entry. PPO/GRPO (clipped ratio, group baselines) noted as a heavier
follow-on only if REINFORCE earns it.

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
- **E7 — Multi-Head Latent Attention + positional-encoding switch.** Graduates
  into **M12 (v1.2.0)**; decision in **ADR 0007**. The attention/KV-cache axis
  (continues E1 + E2): cache a low-rank latent `c_KV` per token, up-project to
  per-head K/V on read — plus a `--pos-kind {learned, rope, rope-decoupled}`
  switch (learned-abs default; coupled/decoupled RoPE for the faithful MLA).
  Gate: latent-projection + RoPE-rotation backward grad-checked;
  cached-vs-uncached bit-identity; KV-cache-bytes vs GQA; perplexity vs iso-param
  GQA on vidya.
- **E8 — Mixture of Experts (sparse FFN).** Graduates into **M13 (v1.3.0)**. The
  FFN axis: an N-expert top-K router replacing the dense MLP, swept over
  **N ∈ {8, 16, 32, 64, 128, 256}**. Gate: router + load-balance-aux backward
  grad-checked; deterministic frozen tie-break; density sweep (bits/byte, active
  vs total params, expert utilization) logged.
- **E9 — Reinforcement learning (policy gradient).** Graduates into
  **M17 (v1.7.0)** — last in the chain. An orthogonal *objective* layer (runs on
  any trunk): REINFORCE with a deterministic reward, the log-prob gradient
  reweighted by advantage `(R − b)` — a small grad-checkable delta over the
  softmax-CE backward. Gate: reward-weighted backward grad-checked; RL-vs-SFT
  reward-movement logged.

Sequencing intent: E1–E2 shipped pre-freeze as M6 (v0.7.0); E3 as M7 (v0.7.1);
the v0.8–v1.0 ladder (security → perf → freeze → clean cut) shipped as M8–M10.
Past 1.0, the remaining experiments ship as **additive 1.x minors** (not a v2
fork — see "The 1.x architecture arc" above): **E7 → M12 (v1.2.0, MLA + pos-kind),
E8 → M13 (v1.3.0, MoE), E4 → M14 (v1.4.0, mixers), E5 → M15 (v1.5.0, diffusion),
E6 → M16 (v1.6.0, ternary), E9 → M17 (v1.7.0, RL)**. The order is value÷risk —
MLA continues shipped KV-cache infra (lowest risk), MoE is the requested
scale-up knob, the mixer/objective/precision departures ride later, and RL is
last because it is an objective layer over a finished trunk — and is
re-orderable, since the axes are orthogonal. Each is opt-in config + a new
checkpoint version with permanent back-compat, so the frozen 1.0 surface is
never broken.

## Out of scope (for v1.0)

- GPU / accelerator backends — attn11 is a CPU, scalar-f64 (then SIMD)
  reference implementation. (The survey's engine/serving layer — vLLM/SGLang,
  FP4 tensor cores, photonics — is *observed, not chased*; only its
  algorithmic ideas translate here.)
- Distributed / multi-process training.
- A general autodiff engine — gradients stay hand-derived and grad-checked.
- Windows / macOS as first-class training targets (cross-build only, if at all).
