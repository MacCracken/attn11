# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.8.0] - 2026-06-11

**Security sweep (roadmap M8).** A research-driven hardening release: six
vulnerability classes web-researched against recent CVEs, then adversarially
mapped onto attn11's surfaces (survey→map workflow, 12 agents); per-class
dispositions — negative results included — in
[`docs/audit/2026-06-11-m8-security-sweep-audit.md`](docs/audit/2026-06-11-m8-security-sweep-audit.md).
The headline result is a **negative** one: the checkpoint is a flat
native-endian i64 array with no opcode interpreter, callable revival, or
embedded path, so it is **structurally immune** to the pickle/Keras/numpy
model-file deserialization-RCE genre. Two real bugs and a batch of hardening
fell out of the map + the new file-path coverage.

### Security
- **Fixed a crash on every AGNOS `--load`**: `ckpt_load_file` called
  `_file_size(fd)` but the function is `_file_size(fd, path)`; on AGNOS the
  path-stat branch ran `strlen()` on the garbage register left by the dropped
  arg → OOB read / SIGSEGV before any content was examined (masked on Linux's
  `fstat` branch; the arity mismatch compiled silently). Now passes `path`.
- **`_atoi` saturates at ~1e9**: a garbage-huge `--steps`/`--layers`/… can no
  longer wrap mod 2⁶⁴ to a plausible small or negative value (defense-in-depth
  atop `model_config_ok`'s caps).
- **Merge-table scratch pinned to its buffer**: the `lens[6144]` validation
  array sat at the exact `BPE_VMAX` boundary; an explicit `(Vb+j) ≥ 768 → -37`
  bound stops a future cap bump from silently overflowing the stack buffer.
- **CI supply-chain hardening**: every GitHub Action **SHA-pinned**
  (`actions/checkout`, `softprops/action-gh-release`) against the floating-tag
  retag-compromise vector; the `GITHUB_REF_NAME` **awk-injection** in
  `release.yml` closed (tag passed as `awk -v` data); `contents: write`
  **scoped** to the release job only (CI gate runs read-only). Deferred items
  (installer `curl|sh` pinning, release-artifact signing/provenance, a `lib/`
  closure lockfile) are documented with rationale in the audit.

### Fixed
- **Checkpoint *save* was broken on the entire aarch64 CI lane.** The new
  file-path round-trip test (below) exposed that `secure_write_atomic`'s
  durability barrier used `fsync`, which **qemu-user aarch64 mis-emulates**
  (returns `EFAULT`; it works on real aarch64). Every `ckpt_save_file`
  returned `-2` and wrote nothing under qemu. Switched the barrier to
  **`fdatasync`** (75 x86_64 / 83 aarch64) — sufficient for the
  temp-write-then-rename crash-atomic guarantee (flushes data + size, skips
  only mtime/atime) and emulated correctly. Real aarch64 binary now saves and
  loads under qemu.

### Added
- **`test_ckpt_file_roundtrip`** (5 checks, 247 total): the file-path loader
  (`ckpt_save_file`/`ckpt_load_file`) was untested — only the in-memory
  `ckpt_load_buf` was. Drives a save→load→bit-compare round-trip; this is what
  surfaced the two `Fixed`/`Security` findings above.
- **`agnos-smoke.sh` now `--load`s on AGNOS**: the run gate saved a checkpoint
  and byte-compared it to Linux but never *loaded* it — exactly the surface the
  AGNOS crash lived on. It now loads and asserts the "resumed from checkpoint"
  marker.
- **Two fuzz modes** (`tests/attn11.fcyr`): a boundary-combination checkpoint
  mutation (every size field at/over its cap at once) and a max-vocab triple
  mode (`V=768, Vb=256, K=512` — the `BPE_VMAX`/`lens` boundary).

### Changed
- `_fsync` → `_fdatasync` (`src/fileio.cyr`); `CKPT_MAX_MODEL_BYTES` already
  lives in `model.cyr` (see `docs/architecture/004`). Training, generation,
  checkpoint format, and CLI surface are unchanged — a 0.7.1 binary and an
  0.8.0 binary produce bit-identical checkpoints and output.

## [0.7.1] - 2026-06-11

**Scale preset + BPE (roadmap M7 — frontier E3 graduated).** A `--preset` for
ctx 64 / d_model 64 (whole statements instead of 16-char fragments — the
quality lever the X001 vidya run exposed), an opt-in simple BPE tokenizer
(`--bpe K`; byte-level stays the default — ADR 0006), checkpoint format v3
(records the tokenizer; v1/v2 still load), and `--eval` (bits-per-byte, the
tokenizer-comparable metric). Byte-vs-BPE at iso-compute measured on the
vidya corpus (X003 in the experiments ledger). A run without the new flags is
behaviorally identical to 0.7.0.

### Added
- **`--preset`**: ctx 64 / d_model 64 / 8 heads / 4 layers (205 760 params at
  the embedded corpus). At T=64 the context-shift re-prime amortizes over 32
  tokens, so KV-cached generation is **23× faster** than uncached (vs 6.1×
  at the default config) — 64 → 1 486 tok/s greedy.
- **`--heads N` / `--kv-heads N` / `--layers N`**: config overrides for fresh
  models (last-wins with `--preset`; ignored under `--load`); invalid
  combinations abort cleanly in `model_init`.
- **Simple BPE tokenizer (`--bpe K`, K ∈ [1, 512])**: learns K most-frequent
  adjacent-pair merges over the byte-level base vocab (Sennrich et al. 2016;
  byte-level layering per GPT-2). Pure i64 — bit-reproducible across arches
  (pinned by an exact-merge-sequence test on both x86_64 and aarch64/qemu).
  Deterministic tie-break frozen: row-major ascending argmax, greedy
  left-to-right non-overlapping replace, overlap-inclusive counting
  (ADR 0006). Decode via a precomputed flat span table — no recursion.
  `bpe_learn` benched: ~110 ms for 256 KB at K=128 (one-shot, pre-training).
- **Checkpoint format v3**: 16-field header adds `tok_kind`/`base_vocab`/
  `n_merges` + the merge table after the base vocab; saves always write v3;
  **v1 and v2 checkpoints still load** (as byte-level; pure header shifts of
  the same body). The loader's vocab cap rises to 768 for v3 (`= 256 base +
  512 merges`); v1/v2 keep the 256 cap verbatim.
- **`--eval`**: one deterministic, RNG-neutral pass over the corpus at stride
  T; prints CE/token and **bits-per-byte** (BPE targets weighted by their
  byte expansion) so byte and BPE runs are directly comparable. Runs after
  `--save`, so checkpoints are identical with or without it.
- **X003** (experiments ledger): byte vs BPE at iso-compute (analytic MACs,
  `12C² + 2TC + CV` per token) on the vidya corpus at the preset config.
- 81 new checks (161 → 242): BPE known-merge/round-trip/determinism pins,
  v3 round-trip + the full rejection matrix (-32…-39 incl. the forgery
  cascade), BPE resume determinism (through the corpus rebuild), BPE
  generation bit-identity, eval determinism/RNG-neutrality, the preset-shape
  KV bit-identity gate, V=300 generation bit-identity + alloc-accounting
  pins, and the **config-magnitude-cap pin** (`model_config_ok` rejects
  out-of-range V/C/T/NL — the `--layers` heap-OOB regression below). Fuzz:
  +500 BPE-image rounds (merge-slot clobber, (V,Vb,K) triple inconsistency,
  expansion-bomb rewrite) + a BPE round-trip property over 100 random
  corpora.

### Security
- **Hostile merge tables cannot loop or bomb the decoder**: the v3 loader
  validates the merge table as a well-founded DAG (every reference strictly
  below its minting id — rejects self/forward/negative refs and all cycles,
  `-37`) with a length recurrence capping every token's expansion at 64
  bytes (rejects exponential chains, `-38`) — on fixed stack scratch,
  BEFORE any allocation. New codes: `-32` tok_kind, `-33` base vocab, `-34`
  merge count, `-35` V ≠ Vb+K, `-36` byte-kind with merges, `-39` BPE
  resume without retained corpus bytes (defensive).
- **Fixed a `--layers` heap-OOB (M7 adversarial review, confirmed 3/3).** The
  new `--heads`/`--kv-heads`/`--layers` flags made the fresh-model config
  CLI-controllable, but `model_config_ok` bounded `NL` only by `≥ 1` and the
  fresh path skipped the checkpoint loader's pre-allocation cap. A crafted
  `--layers` value overflowed `NL · _blk()` in `model_init`, wrapping the
  allocation size to a small positive that `t_alloc` accepted — then
  weight-init wrote past the undersized buffer (SIGSEGV / heap corruption,
  reachable on the default corpus with no `--load`). `model_config_ok` now
  caps V/C/T/NL to the same magnitudes the loader enforces (`-4/-5/-7/-8`),
  and `model_init` runs the `model_alloc_bytes` pre-flight against the 128 MB
  cap before any allocation — the two config gates (file and CLI) now share
  one invariant. Pinned by `test_config_caps`.
- Fixed a latent OOB read the BPE work exposed: `ckpt_serialize`'s vocab
  loop ran to `g_V` over the 256-entry `g_vocab` table — correct while
  V ≤ 256 always held, an OOB read once BPE pushes V past it. The loop now
  runs to the base-vocab count.
- `_atoi(0)` null guard: a value flag given as the last CLI arg (e.g.
  `--steps` with no value) read through `argv()`'s out-of-range null instead
  of crashing.

### Changed
- Toolchain pin `6.1.33` → `6.1.34` (with the matching `lib/` snapshot —
  the pin and snapshot move together).
- Banner now prints the active tokenizer (`tokenizer=byte` /
  `tokenizer=bpe(merges=K)`) and labels the corpus length in **tokens**
  (post-merge ids), not chars.
- `generate()`'s prompt path branches: byte mode keeps the 0.7.0 loop
  verbatim; BPE prompts encode through the learned merges (last ≤ 8 192
  bytes), then take the last ≤ T ids.

## [0.7.0] - 2026-06-11

**Inference efficiency (roadmap M6 — frontier E1 + E2 graduated).** KV-cached
generation makes sampling **6.2× faster** (951 → 5 868 tok/s at the default
config), and grouped-query attention (GQA/MQA) makes the KV cache's size a
config knob (up to 4× smaller at `nkv=1`). Training at the default config is
unchanged (same 39 488 params, same init draws, same loss curve).

### Added
- **KV-cached generation (E1)**: per-layer K/V caches + a single-row cached
  forward (`attn_fwd_row`, `model_fwd_row`) — generation processes one row per
  token instead of recomputing the whole window. **Bit-identity gate**: the
  cached path's logits match the uncached reference (`model_eval_window`)
  bit-for-bit at every prefix length, across context-shifts, greedy and
  temperature, MHA and MQA (`test_kv_generation`). Bench: 1 050 579 →
  170 392 ns/token greedy (x86_64, default config, pin 6.1.33).
- **Grouped-query / multi-query attention (E2)**: `n_kv_heads` config
  (`nkv ≤ nh`, must divide it; default `nkv = nh` = classic MHA). K/V
  projections shrink to `C × Ckv` (`Ckv = nkv·C/nh`); each group of `nh/nkv`
  query heads shares one K/V head. Backward derives grouped `dK`/`dV`
  accumulation; **grad-checked at `nkv ∈ {1, 2, nh}`** (attention level) and
  `nkv < nh` (full model), plus MQA resume-determinism. KV bytes are
  accounted in the bench: 24 576 (`nkv=4`) → 6 144 (`nkv=1`) at the default
  config.
- **Checkpoint format v2**: 13-field header adds `nkv` (field 12); new
  validations — `nkv ≥ 1`, `nkv ≤ nh`, `nh % nkv == 0` (reject `-16`) and
  `step ≥ 0` (reject `-17`). **v1 checkpoints (≤ 0.6.0) still load**, as
  `nkv = nh` (with `nkv = nh` the v2 parameter layout is identical to v1's);
  saves always write v2. Covered by `test_ckpt_v1_compat`,
  `test_ckpt_gqa_roundtrip`, and the extended fuzz header mutations.
- Generation benchmarks (`gen uncached` / `gen kv-cached` / tokens/sec /
  KV bytes per `nkv` / MQA training + generation timings) in
  `tests/attn11.bcyr`.
- 97 new checks (64 → 161): GQA/MQA grad checks incl. `dWk`/`dWv`/`dbv`, the
  KV bit-identity suite (hd ∈ {4,6,8,10} — SIMD tails live — × nkv ∈
  {1,2,nh} × odd-T shifts), checkpoint v1-compat/GQA round-trip/rejection
  tests, the parameter-layout tiling pin, and the alloc-accounting pin.

### Security (M6 adversarial review — see `docs/audit/2026-06-11-kv-gqa-audit.md`)
- **Pre-allocation bound on checkpoint loads**: a shape-valid header whose
  model would blow the allocator (e.g. the `nh·T·T` attention-arena term,
  independent of `np`) is rejected (`-18`) via `model_alloc_bytes()` — an
  exact, test-pinned mirror of `model_init`'s allocations — BEFORE anything
  is allocated. Caps tightened to the allocator's reality: `CKPT_MAX_NP`
  64M → 4M params, `CKPT_MAX_BYTES` 2 GB → 128 MB, new
  `CKPT_MAX_MODEL_BYTES` 128 MB. Previously such a checkpoint SIGSEGV'd in
  `t_alloc`'s zero-fill (alloc() returns 0 past its cap).
- `model_init` now enforces the config invariants itself (`nh | C`,
  `nkv | nh`, `nkv ≤ nh`) — an invalid in-process config aborts cleanly
  instead of silently corrupting arena/KV memory.
- Checkpoint `rng_state = 0` (the xorshift64 fixed point — bricks the PRNG
  stream) is rejected (`-19`).
- `t_alloc` aborts cleanly on allocation failure (was: zero-fill from
  address 0); `ckpt_load_file`/`ckpt_save_file`/corpus loaders null-check
  their buffers (`-31`/`-21`/`-4`).

### Changed
- **Toolchain pin `6.1.31` → `6.1.33`** (`lib/` re-synced). The 0.7.0 AGNOS
  run gate caught the drift: cycc 6.1.32 fixed the argv-capture issue attn11
  filed (init rsp parked in r15 at the entry landing; the
  `_agnos_init_rsp` global and `_agnos_capture_rsp` are GONE), so a 6.1.33
  compiler against the stale 6.1.31 `lib/args_agnos.cyr` gave `argc()==0`
  under the booted kernel — CLI flags silently ignored, Linux unaffected.
  At pin ≥ 6.1.32 the `docs/architecture/002` statement-call epilogue
  workaround is no longer load-bearing (entries keep it; it is harmless).
- **Generation semantics**: the sampler no longer left-pads short prompts with
  id 0 — the prompt's last `min(plen, T)` bytes occupy positions `0..n-1` and
  the context grows incrementally. When the window fills, the oldest `T/2`
  tokens are dropped and the kept context re-primed at its new positions
  (context-shift; required because learned absolute positional embeddings pin
  each cached row to its position — ADR 0005). Sample output for a given
  checkpoint therefore differs from 0.6.0's sliding-window sampler.
- `model_init` gained an `nkv` parameter (after `nh`); `attn_fwd`/`attn_bwd`/
  `attn_arena_size` take `nkv`. Banner prints `kv_heads=`.

### Fixed
- `docs/examples/minimal_train.cyr` still used the pre-M5 `var r = main();`
  entry epilogue (banned by `docs/architecture/002`) and the pre-0.7.0
  `model_init` signature — both updated; the example builds and runs again.

### Discovered
- **The K-projection bias has exactly zero gradient** — a constant bias added
  to every K row shifts each attention score row by `q_i·bk`, constant over
  the softmax dimension, and softmax is shift-invariant; GPT-2's K bias is a
  no-op parameter. Found when the new `dbk` grad check compared two
  rounding-noise vectors. The suite now FD-checks `dbv` instead and pins
  `|dbk| < 1e-10` (the backward must *respect* the invariance).

## [0.6.0] - 2026-06-11

**AGNOS kernel port (roadmap M5).** attn11 now runs as a ring-3 application
under the AGNOS kernel: it **trains, checkpoints, and samples under the booted
kernel**, and the saved checkpoint is **bit-for-bit identical** to the Linux
run (948,008 bytes, fixed seed, CPU implementation held constant). One source
tree compiles for Linux x86_64, Linux aarch64, and AGNOS. No model behavior
change on Linux (52 checks, fuzz, and benchmarks unchanged).

### Added
- **AGNOS cross-build** (`cyrius build --agnos src/main.cyr build/attn11_agnos`)
  compiles clean — a static x86_64 ELF64, the shape agnos exec-from-disk
  (`elf_load_from_file`) requires. New build-only `agnos` CI lane (binary +
  grad-check suite + static-ELF verify).
- `scripts/agnos-smoke.sh` — the M5 run gate as a one-command harness, **PASS**
  on agnos 1.44.15 + agnsh 1.6.x: boots the real kernel in QEMU (gnoboot +
  ext2 rootfs with `/bin/agnsh` + `/bin/attn11`), drives
  `run /bin/attn11 --steps N --save /ck.ckpt` over the emulated keyboard,
  extracts the checkpoint from the ext2 image with `debugfs` (post-boot
  `e2fsck` clean) and `cmp`s it against the Linux reference. The reference
  runs under `qemu-x86_64` when the guest is TCG — x87 transcendentals are
  implementation-defined, so silicon-vs-TCG differs by ULPs; holding the CPU
  constant isolates the software stack, mirroring the aarch64 method. A
  1000-step run under AGNOS also matches native serial output (loss/lr/
  grad-norm) to every displayed digit.
- `docs/guides/agnos.md` — how the AGNOS build works, what the target lacks
  (no `fstat`/`fsync`, explicit-length paths, `AO_*` flags), how attn11
  bridges each gap, and how to run the gate.
- `docs/audit/2026-06-10-agnos-audit.md` — the M5 delta audit (adversarial
  review + run-gate findings, both fixed/worked-around; the two documented
  AGNOS security deltas).
- `docs/architecture/002-agnos-entry-epilogue.md` + the upstream issue filing
  (see Fixed below).

### Fixed
- **agnos argv: entry epilogues converted to the statement-call shape**
  (`var r = 0; r = main();` — all five entry files). With the scaffold's
  `var r = main();` initializer shape, cycc emits the `main()` call inside the
  gvar-init block, *before* the v6.1.14 `_agnos_capture_rsp` emission — so on
  agnos `argc()` returned 0 inside `main` and every CLI flag was silently
  ignored (Linux unaffected). Diagnosed by disassembly + a minimal argv probe
  under the booted kernel; upstream cycc gap filed in
  `docs/development/issues/2026-06-10-cyrius-agnos-capture-after-gvar-init-call.md`,
  rule recorded in `docs/architecture/002-agnos-entry-epilogue.md`.

### Changed
- **Toolchain pin `6.1.6` → `6.1.31`** — flagged by the M5 adversarial review:
  6.1.6 predates two HIGH-sev agnos codegen fixes (6.1.13: indirect calls
  returned 0 on the agnos target; 6.1.14: `argc()`/`argv()` returned 0/null
  because the init-stack capture ran after top-level code moved rsp — exactly
  attn11's `var r = main()` shape), so a 6.1.6-built agnos binary silently
  ignores every CLI flag. `lib/` re-synced to the 6.1.31 snapshot. Stdlib deps
  follow the 6.1.31 reshuffle: `ganita` added (`f64_tanh`/`f64_pow` moved
  there from `math.cyr`), unused `matrix` dropped (its `mat_*` now duplicate
  ganita's, and attn11 has its own SIMD matmul).
- **De-Linuxed every raw syscall site** so one source tree compiles for Linux
  x86_64, Linux aarch64, and AGNOS:
  - `tensor.cyr` `_putc`: raw `syscall(1, 1, …)` → portable `sys_write`.
  - `main.cyr`/`test.cyr` + the test/bench/fuzz harness epilogues: raw
    `syscall(60, …)` exit / `syscall(1, 2, …)` stderr → `sys_exit`/`sys_write`.
  - `fileio.cyr`: `_file_size` now takes `(fd, path)` — Linux keeps `fstat(fd)`
    (st_size @ 48); AGNOS has no fstat, so it path-stats (ABI §4.1, size @ 16).
    New `_unlink`/`_rename` shims bridge the explicit-path-length AGNOS ABI
    (§3.2) vs the NUL-terminated Linux wrappers. `_fsync` falls back to the
    global `sys_sync()` on AGNOS (no per-fd fsync in the frozen ABI).

### Removed
- The unused `random` stdlib dep (kernel-CSPRNG `random_bytes`; attn11's PRNG
  is deliberately deterministic) — it was also the one undefined-symbol
  (`sys_getrandom`) hold-out in the AGNOS build.
- **`lib/` is no longer committed** (patra model): the old
  `lib/*.cyr`-with-whitelist gitignore shipped a tracked-but-incomplete lib
  that shadowed the pinned snapshot — the new-at-6.1.31 `ganita.cyr` was
  silently ignored and CI's `cyrius deps` failed on the clean checkout. The
  whole dir is generated now: `cyrius deps` materializes the closure from the
  pin (verified: the full CI matrix passes from a lib-less checkout);
  `cyrius lib sync` refreshes a local copy.

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
