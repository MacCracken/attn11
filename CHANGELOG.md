# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.7.4] - 2026-06-19

**Toolchain realign (cyrius 6.2.27 → 6.2.29) + M18 GPU-arc unblock verified (mabda 3.4.1).**
A maintenance cut with one milestone-status finding. The installed cycc had rolled ahead of
the pin (local builds warned of drift at 6.2.27 vs 6.2.29); bumped `cyrius.cyml` and resynced
the gitignored `lib/` snapshot + `cyrius.lock`. The 6.2.27 → 6.2.29 stdlib snapshot differs in
**only 5 files**, four off attn11's compile path entirely (`bayan`, `fdlopen`, `hashmap_fast`,
`tls_native_conn`); the fifth, `syscalls_aarch64_linux.cyr`, is a genuine upstream **bugfix** —
`SYS_FACCESSAT` renumbered 48 → 269 to dodge an ESYSXLAT collision that made native-aarch64
`sys_access()` return -EBADF (it remapped to `setsockopt`). attn11 calls no `sys_access`, so the
fix is in uninvoked code: the **x86_64 binary is byte-identical** across the bump (proven —
`1f8683b0…` ≡ `1f8683b0…`, 373,504-byte static ELF), and the aarch64 binary differs *only* in
that one constant with a **byte-identical default run**. **1056** checks green (x86_64 +
aarch64/qemu); lint + fuzz + `make smoke` exit 0. `src/*.cyr` unchanged except `CFG_VERSION`.

The milestone finding (X026): the **M18 GPU arc is unblocked**. An isolated-worktree
integration probe against cyrius 6.2.29 + **mabda 3.4.1** proved the one real blocker — the
`F64_HALF`/`F64_TWO` symbol collision with the `math` stdlib that statically zeroed those
constants → NaN in ganita tanh/GELU — is **fixed**: mabda 3.4.1 renames its copies to
`MABDA_F64_*`, so `include "lib/mabda.cyr"` now builds clean and the no-`--gpu` CPU path is
byte-identical (finite loss, no NaN). `gpu_probe` (src/gpu.cyr) brings the **native-AMD Cezanne
f64 SPIR-V device online** (launcher-free, pure-Cyrius; X025). The previously-listed second
item — cyrius `dep-module-call` — is **reclassified as a transient binary-size cost, not a
gate**: without it, including mabda amalgamates its ~1 MB dist (binary 373,504 → 1,338,344 B
measured) even when `--gpu` is unused; it is slated for the 6.2.x line and slims back with zero
attn11 change. So M18 is **GO**; mabda stays staged in the default build only to keep it lean
until the landing approach is chosen. Binary unchanged this cut (mabda not yet included).

## [1.7.3] - 2026-06-19

**Toolchain realign (cyrius 6.2.6 → 6.2.27) + dependency bump (tyche/rosnet 0.1.0 → 0.1.1).**
A maintenance cut. The installed cycc had rolled well ahead of the pin (local builds warned
of toolchain drift at 6.2.6 vs 6.2.27); bumped `cyrius.cyml` and ran `cyrius update` to
resync the gitignored `lib/` snapshot + `cyrius.lock`. Both consumed AGNOS crates moved with
it: **tyche 0.1.0 → 0.1.1** and **rosnet 0.1.0 → 0.1.1** — each is itself a pure
toolchain-realign (6.2.11 pin + `dist/` regen, no API change), so attn11's call surface is
untouched. The default training run is **byte-identical** to 1.7.2 (verified: baseline binary
built pre-bump, diffed against the post-bump binary). **1056** checks unchanged green; no pin
drift warning. `src/*.cyr` unchanged except `CFG_VERSION`.

## [1.7.2] - 2026-06-14

**Competitor benchmarks (the B-series, B0) + toolchain realign (cyrius 6.2.5 → 6.2.6).**
The wrap-up cut before the GPU work pauses on the mabda 3.x Vulkan-f64 dependency. Two
binary-neutral threads:

1. **B-series B0 — the competitor-benchmark harness.** New `scripts/compete-bench.sh`
   (+ `make compete-bench`) adds the axis the self-bench lacks: attn11 vs external
   references (llm.c / nanoGPT / llama2.c / micrograd) on the training + decode surfaces,
   plus the normalized **zero-deps** story. It clones + builds competitors at a **recorded
   upstream commit** into a gitignored `bench/` (no vendoring), enforces the fairness rules
   (matched config with the **param count asserted**, `taskset` + warmup, attn11
   single-thread first, competitor thread/deps recorded), and — per the "no silent caps"
   discipline — emits an **explicit status row** (built-only / skip + reason) for any
   competitor it can't match/build/run, never a fabricated number. New
   `competitor-bench.csv` (append-per-run, like `bench-history.csv`); the self-bench track
   is untouched. **B3 (zero-deps) is complete + real**: attn11 ships as **one 372,896-byte
   static ELF, `ldd` = "not a dynamic executable"** (no BLAS/libc/CUDA) vs every
   competitor's runtime stack. The harness validated here by cloning + **building llm.c and
   llama2.c** at recorded refs (`f1e2ace` / `350e04f`) + emitting attn11's real row
   (~4 393 tok/s, 39 488 params); the **matched-config comparison runs (B1/B2) are
   harness-ready but unpopulated** — they need each competitor's stack + GPT-2 data-prep on
   a bench machine (this host has no PyTorch; nanoGPT skipped with a reason). **B4 (GPU)**
   rides M18. Write-up: `docs/benchmarks.md` ("Competitor benchmarks").
2. **Toolchain realign: cyrius pin 6.2.5 → 6.2.6** (cycc rolled forward again, faster than
   the pin). `cyrius.cyml` bumped + `cyrius update` resynced the `lib/` snapshot + lock; the
   default / RL / ternary runs are **byte-identical** to 1.7.1 (verified), grad-checks green
   on the realigned snapshot.

**1014 → unchanged: 1056** checks green x86_64 + aarch64/qemu (no new src tests — the B0
harness is tooling/data/docs); lint + fuzz + `make smoke` green; `make release` exit 0.
`src/*.cyr` unchanged except `CFG_VERSION`.

### Added
- **`scripts/compete-bench.sh`** + **`make compete-bench`** (B0 harness) and
  **`competitor-bench.csv`** (the data track).
- **`docs/benchmarks.md`** — "Competitor benchmarks (the B-series, 1.7.2)" section: the
  fairness rules, the zero-deps table (B3), attn11's baseline, and the honest competitor
  coverage (built vs harness-ready vs skip).
- `.gitignore` += `bench/` (the gitignored competitor clone dir).

### Changed
- `cyrius.cyml` pin **6.2.5 → 6.2.6** + `cyrius.lock`/`lib/` resynced (`cyrius update`); no
  source/behavior change beyond `CFG_VERSION`.

## [1.7.1] - 2026-06-14

**Toolchain realignment (maintenance): cyrius pin 6.2.2 → 6.2.5.** The installed `cycc`
had moved ahead of the pin (local builds warned "cyrius.cyml pins 6.2.2 but cycc is
6.2.5 — toolchain drift"). Bumped the pin in `cyrius.cyml` and ran `cyrius update` to
resync the vendored `lib/` snapshot + `cyrius.lock` to 6.2.5. `lib/` is gitignored (CI
regenerates it from the pin via `cyrius deps`), so the tracked diff is **`cyrius.cyml` +
`cyrius.lock`** only. The 6.2.5 snapshot does differ from 6.2.2 — but **only in files
attn11 does not use on its Linux training/inference path**: networking / TLS
(`net`, `tls_native*`, +6 new TLS files), threading (`thread`, `thread_local`,
`thread_agnos`), `chrono`, and `syscalls_x86_64_agnos`. attn11's core libs are
**unchanged** (`rosnet`, `tyche`, `alloc`, `fmt`, `string`, `io`, `math`, `simd`,
`bounds`, …), so the binary and every run are **byte-identical** to 1.7.0 (verified:
default / preset / BPE / diffusion / ternary / RL). The pin + snapshot + lock move
together per the standing rule (a new-compiler / old-lib mismatch can reproduce the AGNOS
`argc()==0` trap — and `syscalls_x86_64_agnos` is among the changed files), so the agnos
target was rebuilt and the full gate re-run on the realigned snapshot. **1056** checks
unchanged, green x86_64 **and** aarch64/qemu; lint + fuzz + `make smoke` green; `--agnos`
build OK; `make release` exit 0. `src/*.cyr` is unchanged except `CFG_VERSION`.

### Changed
- `cyrius.cyml`: `[package].cyrius` pin **6.2.2 → 6.2.5** (the single source of truth; CI
  and the release gate read it). `cyrius.lock` resynced via `cyrius update` to the 6.2.5
  hashes; `lib/` (gitignored) regenerated from the new pin. The 6.2.5 lib diff touches
  only networking/TLS/threading/chrono/agnos-syscall files — none on attn11's Linux path
  — so the binary and all runs are byte-identical. No `src/` change beyond `CFG_VERSION`.

## [1.7.0] - 2026-06-14

**M17 — Reinforcement learning (REINFORCE), E9; the last milestone in the chain.**
`--objective rl` trains the policy toward a *reward* instead of a corpus, via on-policy
**REINFORCE** (Williams 1992): each step samples `batch` rollouts from the current policy
(temperature 1), scores each with a deterministic reward, and weights its log-prob
gradient by the advantage `(R − b)` (`b` = an EMA of past mean rewards). The key
simplification — and why RL is a *small* milestone — is that for a softmax policy
`∇log π(a) = −∇CE(a)`, so the policy gradient IS the existing softmax-CE backward over the
sampled rollout **scaled by `(R − b)`**: no new forward, no new backward op, no autodiff.
The model stays a plain AR transformer and an RL-trained image is a **normal v5 checkpoint**
(no format bump). The reward is a simple in-process function: the count of a target char
(`--rl-target C`, default space) per rollout. **Gate met (X024):** the policy moves
decisively toward the reward — target-char frequency rises **9–19% (SFT) → ~99.7% (RL)**
across common/rare/space targets — at the cost of the SFT→RL alignment tax (corpus
bits/byte 0.24 → ~13, honest reward hacking from the naive count reward). The advantage
scale is gated on `g_rl`, so AR and diffusion training are untouched and the no-flag run is
**byte-identical** to 1.6.5. **1045 → 1056** checks green x86_64 + aarch64/qemu; lint +
fuzz + `make smoke` (valid `--objective rl` + error cases) green. See
[ADR 0015](docs/adr/0015-reinforcement-learning.md), [X024](docs/development/experiments.md),
and `docs/sources.md` (Williams 1992). PPO/GRPO + richer rewards are the documented
follow-on.

### Added
- **`--objective rl`** (`src/main.cyr`, `src/train.cyr`, `src/model.cyr`): REINFORCE
  policy-gradient training over the plain AR model. **`--rl-target C`** sets the reward
  target char (default space).
- **`rl_train` / `rl_rollout` / `rl_reward` / `rl_prompt` / `rl_eval`** (`src/train.cyr`):
  the RL training loop (rollout → reward → advantage → reward-weighted backward → EMA
  baseline), the on-policy rollout sampler (lays a sampled sequence out as AR
  `A_tokens`/`A_targets`), the count reward, the random-corpus-token prompt, and a
  rollout-frequency report (the X024 metric, run after `--save`).
- **`g_rl` / `g_reinforce_scale`** (`src/model.cyr`): the advantage scale on the seeded
  logit gradient `D_logits` in `model_backward` (gated on `g_rl`).
- **`test_rl_op` / `test_rl_rollout`** (`tests/attn11.tcyr`): the RL backward grad-checked
  three ways (RL grad == advantage × AR grad to rounding; FD vs the numeric gradient of
  `advantage × CE`; sign-flip + zero-advantage limits), plus the rollout layout /
  reward / prompt structural checks. **+11 checks** (1045 → 1056), dead-last.
- **`scripts/m17-rl.sh`**: the reproducible X024 runner (SFT vs RL per target char).
- **[ADR 0015](docs/adr/0015-reinforcement-learning.md)** and a Williams-1992 citation in
  `docs/sources.md`. **Smoke**: valid `--objective rl` (+ `--rl-target`) runs and the
  `--rl-target`-without-`--objective rl` / bad-objective rejections.

### Changed
- `model_backward` (`src/model.cyr`) scales `D_logits` by `g_reinforce_scale` when
  `g_rl != 0` (the policy-gradient term); the AR/diffusion paths (`g_rl == 0`) are
  untouched and byte-identical. The `--objective` flag now accepts `rl` alongside
  `ar`/`diffusion`.

## [1.6.5] - 2026-06-14

**X021 — the data-volume held-out win (experiment; binary unchanged).** The run X019
wanted and X020 unblocked but couldn't reach: does training on MORE clean data generalize
better to a THIRD disjoint held-out set? It only shows once the model **overfits** the
small corpus (X020 found <1.3% own→held at sub-epoch — nothing to convert), so X021 forces
that regime — a 256 KB train slice (many epochs) vs a disjoint 4 MB slice at matched
compute (4000 steps, BPE 256, seed 1337), both scored on a third disjoint 512 KB slice of
one curated 12 MB C4-en pool. Rides the shipped `--eval-corpus` + `c4_sample.py`; **no new
binary surface** — the binary is byte-identical to 1.6.4 (CFG_VERSION bump only), and the
new reproducible runner is `scripts/x021-heldout.sh`. **Result (held-out bits/byte):** at
**preset** capacity, training on 4 MB beats 256 KB by **−14.2%** (2.383 vs 2.776) — the
data-volume generalization win; at **default** (tiny) capacity it is a **tie** (−0.06%), so
the win is a **capacity lever** (the X017/X019 thesis, now on held-out text). The overfit
regime is reached and exposes the headline caveat: preset·256 KB has a **+40.4%** own→held
gap (own 1.977 ≪ held 2.776) — on its own corpus it looks *better* than the 4 MB model
(1.977 vs 2.366), but that is memorization; on held-out the 4 MB model wins decisively. So
**own-corpus bits/byte inverts the truth above the overfit threshold** — exactly when a
held-out metric earns its keep — while remaining trustworthy below it (validating X020's
read of the 1.5.x numbers). Closes the data-ingestion story (X016→X021). Full write-up:
[X021](docs/development/experiments.md).

### Added
- **`scripts/x021-heldout.sh`**: the reproducible X021 runner — curate a C4 pool, cut three
  disjoint slices (byte-offset gaps), train {default, preset} × {small, large} at matched
  compute, and report own vs held-out bits/byte + the own→held gap. Deterministic; the
  slice sizes / step budget are env-overridable.

### Changed
- Nothing in the binary — `src/*.cyr` is unchanged except `CFG_VERSION` (the default run,
  every checkpoint, and all 1045 grad-checks are byte-identical to 1.6.4).

## [1.6.4] - 2026-06-14

**BPE GB-scale shards (`--stream-encode`); the second 1.6.2 fast-follow.** The in-RAM
`--encode-shard` path caps the corpus at 64 MB (it loads it whole), so BPE shards were
limited to that. **`--corpus FILE --bpe K --encode-shard OUT --stream-encode`** now
pre-encodes an arbitrarily large file to a BPE (u16) token-shard in **bounded RAM**
(~the 64 MB learn budget, independent of total corpus size). Two passes over the file:
(0) learn the byte vocab + K merges from a bounded prefix; (1) re-read the whole file in
1 MB chunks, apply the tokenizer with `tok_encode`, and emit only the **stable** prefix
of each chunk's ids while carrying the trailing raw bytes. The header's `ntokens` (not
known until the end) is written as a placeholder and patched via `lseek` before the
atomic rename. Byte-level `--stream-encode` works too (no merges → no carry).

The hard part is **chunk-boundary exactness**: BPE merges cross chunk boundaries, so a
naive chunked encode diverges from a whole-corpus encode. The fix rests on a property of
greedy left-to-right BPE: a final token spans ≤ `BPE_MAX_TOKLEN` bytes, and a token
ending ≥ `BPE_MAX_TOKLEN` from the right edge can never change as more bytes arrive (it
cannot be absorbed into a boundary-crossing token, and greedy-LTR never re-pairs to its
left). So carrying the trailing `2*BPE_MAX_TOKLEN` raw bytes — always re-encoded from a
**true token boundary** — makes the chunked encode byte-identical to a single-buffer
encode. Verified: `--stream-encode` produces a shard **byte-identical to in-RAM
`--bpe K --encode-shard`** on the same corpus, across real 1 MB chunk boundaries (a
2.6 MB BPE-128 corpus) and across hundreds of boundaries at a 7-byte forced chunk
(`test_stream_bpe_encode`). The default run stays byte-identical to 1.6.3.
**1039 → 1045** checks green x86_64 + aarch64/qemu; lint + fuzz + `make smoke` (new
`--stream-encode` BPE round-trip + error cases) green.

### Added
- **`--stream-encode`** (`src/main.cyr`, `src/stream.cyr`): a one-shot mode that, with
  `--corpus FILE` + `--encode-shard OUT` (+ optional `--bpe K`), streams a large corpus
  into a token-shard in bounded RAM. Requires a seekable file (not `--stdin`); excludes
  `--load` / `--stream-corpus`.
- **`shard_stream_encode`** (`src/stream.cyr`): the bounded-RAM encoder — bounded-prefix
  tokenizer learn, chunked `tok_encode` with a `2*BPE_MAX_TOKLEN` raw-byte carry, a
  streaming shard writer with an `lseek` header patch for `ntokens`, and a crash-atomic
  rename. `g_stream_enc_chunk` overrides the 1 MB chunk for tests; `_enc_write` loops
  short writes.
- **`test_stream_bpe_encode`** (`tests/attn11.tcyr`): the chunked encode (forced 7-byte
  chunk, ~6 KB synthetic corpus) is byte-identical to the whole-corpus in-RAM encode, at
  BPE and byte width. **+6 checks** (1039 → 1045), dead-last (RNG-neutral).
- **Smoke** (`Makefile`): a `--stream-encode` BPE round-trip (encode → train from the
  shard) plus the `--stream-encode` error cases (no `--encode-shard`, no `--corpus`,
  with `--stdin`).

## [1.6.3] - 2026-06-14

**Resume-from-stream (`--load` + `--stream-corpus`); the first 1.6.2 fast-follow.**
A checkpoint saved mid-stream can now be reloaded against the same token-shard to
continue training — the shard supplies the corpus, the checkpoint supplies the
weights / optimizer moments / step / RNG. Resume is **bit-for-bit deterministic**:
`train(K)` streamed in one go equals `train(K1)` → checkpoint → `train(K)` streamed,
verified at byte **and** BPE width (the streamed path touches no RNG, and the schedule
horizon is held fixed, exactly like the in-memory resume). Because a streamed corpus
holds the tokens on disk (`g_data == 0`) and never re-encodes, the loader requires the
shard to carry the **EXACT** tokenizer the checkpoint trained with — kind, base size,
merge count, every base-vocab byte, and every merge pair — not just the base vocab the
in-memory resume path checks (which adapts via re-encode). This closes a real hole the
naive check would miss: a BPE shard whose base vocab happens to match a byte checkpoint
would otherwise load and then stream ids ≥ Vb straight past the byte model's embedding
table; now both directions reject cleanly with `-15`. The in-memory resume path is
unchanged (its grad-checks stay green), and the default run is byte-identical to 1.6.2.
**1032 → 1039** checks green x86_64 + aarch64/qemu; lint + fuzz + `make smoke` (new
resume-from-stream round-trip + vocab-mismatch cases) green.

### Added
- **`--load` + `--stream-corpus`** (resume-from-stream): supported in `src/main.cyr`
  (the prior rejection is removed); the corpus streams from the shard while the
  checkpoint restores the model.
- **`test_resume_stream`** (`tests/attn11.tcyr`): `train(K)` streamed == `train(K1)` →
  in-memory checkpoint round-trip → `train(K)` streamed, params bit-for-bit; plus the
  tokenizer-mismatch rejection (`-15`). **+7 checks** (1032 → 1039), dead-last.
- **Smoke** (`Makefile`): a valid `--stream-corpus --save` → `--stream-corpus --load`
  resume round-trip, and a BPE-shard-against-byte-checkpoint mismatch reject.

### Changed
- `ckpt_load_buf` (`src/persist.cyr`): the corpus-vocab consistency check splits into
  an in-memory branch (`g_data != 0`, base-vocab match, the existing re-encode path)
  and a **streamed branch** (`g_stream != 0`, full-tokenizer identity: kind + Vb + K +
  base vocab + all merge pairs, exempt from the `-39` retained-bytes requirement since
  the shard reads directly). The in-memory branch is behaviorally identical to 1.6.2.

## [1.6.2] - 2026-06-14

**Streaming token-shard ingestion (1.6.x group; the RAM-independent large-corpus
path).** Two new flags decouple corpus size from RAM. **`--encode-shard PATH`**
pre-encodes the loaded corpus (byte-level or `--bpe`) to a self-describing
**token-shard** — a small header + the tokenizer (byte vocab + BPE merges, exactly
as a checkpoint carries it) + the packed token ids — then exits.
**`--stream-corpus PATH`** trains / evals *from* a shard **without loading the
tokens into RAM**: `gd_ld` pulls corpus windows by offset through a single 64 K-token
chunk cache (file `lseek`+`read`, or a memory blob in the test), so RAM is
O(model + chunk) regardless of corpus size — a GB shard trains in ~13 MB. The key
design property: `gd_ld` is the *only* corpus-token reader, so streaming is a
read-through cache inside it and **`sample_window` / `eval_corpus` / the model are
untouched** — byte-identity is structural, not re-verified per call site. Verified
bit-for-bit: a streamed run reproduces the in-memory run's params at byte + BPE
width, across chunk boundaries, on x86_64 **and** aarch64/qemu (the real
`open`/`lseek`/`read` path). `scripts/c4_sample.py --emit-shard` is the GB-scale
producer: it streams C4 and writes a byte-level shard in bounded RAM, **byte-identical
to `attn11 --encode-shard`** on the same text (matching adaptive first-appearance
vocab). Hostile shards are rejected before the tokenizer is installed (a distinct
`-60..` code range, mirroring the checkpoint loader's tokenizer/merge-DAG checks)
and validated against the real file size, so a window seek can never pass EOF; the
fuzz harness adds 500 shard-metadata rounds. The **default run stays byte-identical**
to 1.6.1 (default / preset / BPE / ternary / diffusion / ssm-hybrid all verified).
Streaming is Linux-only (`lseek`); the agnos target rejects `--stream-corpus` (-73).
Resume-from-stream (`--load` + `--stream-corpus`) and BPE GB-scale shards are
documented fast-follows. **1014 → 1032** checks green x86_64 + aarch64/qemu; lint +
fuzz + `make smoke` (new encode/stream + bad-shard cases) green. See
[`docs/architecture/007-streaming-token-shards.md`](docs/architecture/007-streaming-token-shards.md).

### Added
- **`--encode-shard PATH`** (`src/main.cyr`, `src/stream.cyr`): write the in-memory
  corpus (byte or BPE) to a token-shard, then exit — a one-shot pre-encode.
- **`--stream-corpus PATH`** (`src/main.cyr`, `src/stream.cyr`, `src/train.cyr`):
  train / eval from a token-shard with RAM independent of corpus size.
- **`src/stream.cyr`**: the token-shard format — `SHARD_MAGIC` "ATTNSH01", the
  serializer (`shard_serialize` / `shard_save_file`), the hostile-input-safe reader
  (`shard_open`), and the shared metadata validator (`shard_validate_buf` /
  `shard_validate_tok`, the merge table as a well-founded DAG with bounded expansion).
- **Streaming chunk cache** (`src/train.cyr`): `g_stream*` state + `stream_tok` (a
  read-through single-chunk cache) behind a `gd_ld` branch; `STREAM_CHUNK()` = 65536.
- **`file_seek`** (`src/fileio.cyr`): a raw `lseek`(SEEK_SET) wrapper, arch-dispatched
  like `_fdatasync` (the stdlib ships none).
- **`scripts/c4_sample.py --emit-shard PATH`**: emit a byte-level GB-scale shard in
  bounded RAM (two streaming passes), byte-identical to `attn11 --encode-shard`.
- **`test_stream_ingest`** (`tests/attn11.tcyr`): pins streamed-vs-in-memory training
  bit-for-bit (byte + BPE), round-trip fidelity across chunk boundaries (tiny forced
  cap), and hostile-metadata rejection. **+18 checks** (1014 → 1032), dead-last
  (RNG-consuming + `model_init`-reseeding, X019/X015 discipline).
- **Fuzz** (`tests/attn11.fcyr`): 500 shard-metadata rounds (wild triples, clobbered
  merges, truncation, random headers) — `shard_validate_buf` must reject, never crash.
- **Smoke** (`Makefile`): a valid `--encode-shard` → `--stream-corpus` round-trip plus
  missing / non-shard / conflicting-flag cases (reject cleanly, never crash).

### Changed
- `gd_ld` (`src/train.cyr`) routes through the chunk cache when `g_stream != 0`
  (never taken on the default path — the no-flag run is byte-identical); the
  training/eval guards (`eval_corpus`, `eval_diffusion`, `main`) accept a streamed
  corpus; `eval_corpus_buf` disables streaming for the held-out pass so it reads the
  in-memory held-out buffer, then restores.

## [1.6.1] - 2026-06-14

**M16 — the i64-add ternary matmul + bench, increment 2 (E6); closes the M16 gate.**
The BitNet realization of `--ternary`: `x·W_eff = γ·(x·t)`, `t ∈ {−1, 0, +1}`, collapses
the contracted-dim multiply to **add / subtract / skip** with one γ-scale per output
(M·N multiplies vs the dense M·K·N). Two attn11-local reference kernels
`ternary_matmul_fwd` / `ternary_matmul_dx` (in `ops.cyr`) implement the collapse and are
**grad-checked** (`test_ternary_matmul`: the forward and `dx` both pinned against the
SIMD-f64 `W_eff` path at **maxrel 0** — exact to rounding — and `dx` FD-checked at 1e-5;
`dW` is the unchanged STE pass-through already pinned in `test_ternary_quant`), then
**benched head-to-head** against the 1.6.0 SIMD-f64 path. **Honest result (X023):** on
x86_64 the collapse is **~3.0× slower** (matmul) / **~2.4× slower** (end-to-end) than the
SIMD-f64 path — `f64v_fmadd` does 4 fused multiply-adds per instruction while the collapse
is scalar add/sub/skip, so the wide-SIMD f64 multiply is already cheaper per element than
a scalar add (the integer-add win needs **activation quantization** and/or hardware
without wide FMA — the documented heavier follow-on, ADR 0014). So the **default ternary
forward keeps the SIMD-f64 path**; the collapse ships as the grad-checked + benched
reference kernel, wired into no run — **ternary *and* default runs stay byte-identical to
1.6.0** (default, preset, BPE all verified). **1010 → 1014** checks, green x86_64 **and**
aarch64/qemu; lint + fuzz + `make smoke` green. See ADR 0014 (updated consequence) and X023.

### Added
- **`ternary_signs` / `ternary_matmul_fwd` / `ternary_matmul_dx`** (`src/ops.cyr`).
  `ternary_signs(W, t, K, N)` fills `t` with the i64 signs (−1/0/+1) of `W` and returns
  `γ = absmean(W)` (γ·t reproduces `ternary_quant`'s `W_eff` value-for-value);
  `ternary_matmul_fwd` computes `y = b + γ·(x·t)` with the contracted dim as add/sub/skip;
  `ternary_matmul_dx` computes `dx = γ·(dy·tᵀ)` the same way. Reference kernels — the
  default ternary forward (`qlinear_fwd`/`qlinear_bwd`) is unchanged, so no run's numerics
  move.
- **`test_ternary_matmul`** (`tests/attn11.tcyr`, registered dead-last after the other
  ternary tests — the shared-RNG-stream discipline). Pins `signs·γ == W_eff` (value-exact),
  the collapse forward == SIMD-f64 `W_eff` forward, the collapse `dx` == `W_eff` `dx`, and
  FD-checks `dx` through the collapse. **+4 assertions (1010 → 1014).**
- **Ternary matmul bench** (`tests/attn11.bcyr`). Head-to-head at T×C×F = 16×32×128:
  SIMD-f64 vs i64-add, matmul-only and end-to-end (quant + matmul), with the `(x100)`
  ratio printed.

### Notes
- **X023** (the increment-2 bench): the honest negative above — the collapse is exact and
  grad-checked but ~2.4–3× slower than wide-SIMD f64 at this scale/hardware; the integer-add
  advantage lives at activation-quantized / non-FMA regimes (and the orthogonal *memory*
  win, 1.58 bits/weight). See `experiments.md`.
- ADR 0014's "Neutral" consequence is updated: increment 2 is shipped as a benched
  reference kernel with the measured result, not an assumed speedup.

## [1.6.0] - 2026-06-14

**M16 — ternary (BitNet-style) training, increment 1 (E6).** `--ternary` trains with
weights quantized to **{−1, 0, +1}** (BitNet b1.58, arXiv:2402.17764): each quantized
weight matrix `W` is replaced in the forward by `W_eff = γ·clamp(round(W/γ),−1,+1)`,
`γ = absmean(W)`, while the **master weights stay f64** (fake-quant). The backward is a
**straight-through estimator** that costs no new gradient math — `dx` flows through the
fixed `W_eff` (a normal linear backward) and `dW = xᵀ·dy` is the pass-through, which
rosnet's `linear_bwd` already computes without reading `W`. Scope for v1.6.0 is narrow —
**MHA + dense MLP + uniform + AR + learned-abs** (the other axes are documented
fast-follows) — so it lands one idea behind a tight grad-check, the way M15 diffusion
did. The no-flag/default run is **byte-identical** to 1.5.5/1.5.6 (default, preset, and
BPE runs all verified bit-for-bit). The **i64-add ternary matmul + a bench vs the
SIMD-f64 path** (the remaining M16 gate) is deferred to increment 2 — increment 1 reuses
the f64 `linear_fwd` for correctness first. See ADR 0014 and X022.

### Added
- **`--ternary`** (`src/main.cyr`, `src/ops.cyr`, `src/model.cyr`). Two attn11-local
  wrappers `qlinear_fwd`/`qlinear_bwd` (in `ops.cyr`, since the vendored
  `lib/rosnet.cyr` may not be modified) quantize `W` into a pre-allocated scratch
  `g_qscratch` (sized `C·F`, reused sequentially) and run the matmul/backward against
  `W_eff`; with `g_ternary == 0` they are exact passthroughs (byte-identical default).
  Wired into the 16 MHA + dense-MLP call sites (train / cached-decode / eval). Gated to
  mha + dense + uniform + AR + learned-abs at the CLI, with a `model_init_full` backstop.
- **Checkpoint v8** (`src/persist.cyr`). One `g_ternary` field at slot [23] (objective at
  [22], both written; a v8 image is AR). A ternary model writes v8; non-ternary AR still
  writes v5, hybrid v6, diffusion v7 — all **byte-identical**. v≤7 synthesize
  `ternary = 0`. Code `-48` rejects `ternary ∉ {0,1}`, `-49` a ternary image paired with
  mla/lin/ssm/rope/MoE/diffusion (validated before any allocation). The weight blob is
  unaffected — master weights serialize as f64 verbatim.
- **`test_ternary_quant`, `test_model_ternary`, `test_ckpt_ternary`** (`tests/attn11.tcyr`,
  registered dead-last — the shared-RNG-stream discipline). The op test FD-checks the
  `dx` path (smooth in `x` through the fixed `W_eff`, 1e-5) and the full-precision bias,
  pins `W_eff ∈ {−γ,0,+γ}` and `γ = absmean`, and pins the STE `dW` **bit-for-bit**
  against a plain `linear_bwd` (its definition — a naive FD of `dW` through the
  quantizer is meaningless). The full-model test FD-checks the smooth params (embeddings,
  final LN, biases) end-to-end through the quantized stack. The ckpt test round-trips v8
  and pins the `-48`/`-49` rejections. **986 → 1010** checks, green x86_64 **and**
  aarch64/qemu.
- **`make smoke`** gains ternary cases: hostile `--ternary --experts/--objective/--attn-kind`
  reject cleanly, a valid `--ternary` run builds + trains.

### Notes
- **X022** (the ternary-vs-f64 accuracy run): at reference scale ternary is **competitive
  with f64** (default config, 2000 steps: f64 **0.254** bits/byte — bit-for-bit X015 —
  vs ternary **0.228**; the ~1.58-bit constraint acts as a regularizer on the memorizable
  tiny corpus). Read honestly as "the grad-checked ternary trains and is competitive at
  this scale", NOT a general win — BitNet's real advantage is the memory/compute of the
  i64-add matmul at large scale (increment 2 + M16's scale work). See `experiments.md`.
- ADR 0014 documents the STE design and why `dW` is pinned, not FD'd.

## [1.5.6] - 2026-06-13

**Held-out (cross-corpus) eval — the X019 generalization follow-on.** A new
`--eval-corpus PATH` flag scores the model on a **disjoint** corpus: re-encode the
file through the **currently loaded tokenizer** (the same byte vocab / BPE merges,
no vocab-order check) and run the active objective's eval over it. This is the
apples-to-apples generalization metric X019 called for — bits/byte-on-own-corpus
penalizes a higher-entropy training set in absolute terms, so it can't show "more
clean data → better generalization"; a held-out pass can. Additive and surgical:
the no-flag/default run is **byte-identical** to 1.5.5; no new backward op, so the
grad-check count is structurally unchanged (the new test adds eval-only assertions).

### Added
- **`--eval-corpus PATH`** (`src/main.cyr`, `src/train.cyr`). Re-encodes a disjoint
  corpus through the loaded tokenizer into a packed buffer at the active token width
  (`g_data_w`: u8 byte-level / u16 BPE), temporarily swaps it in as the corpus, runs
  the objective-appropriate eval (AR `eval_corpus` → CE/token + bits/byte; diffusion
  → the masked-CE denoising grid + ELBO bound), and restores `g_data`/`g_datalen`.
  RNG-neutral (eval sets `g_training = 0`), so it never perturbs subsequent sampling.
  Combines with `--eval` (own-corpus then held-out both print) and works under
  `--gen-only --load` (score a saved checkpoint on unseen text). Unknown bytes (absent
  from a byte-level vocab) map to id 0; BPE simply replays its merges — the same
  idempotent re-encode the checkpoint loader already relies on. An explicitly-requested
  held-out eval that can't run (missing/empty/too-short file) sets a **non-zero exit**
  (mirrors `--save`), so a script can detect it; a hostile/missing path **never
  crashes** (new `make smoke` case).
- **`test_eval_held`** (`tests/attn11.tcyr`, registered dead-last — RNG-neutral, so the
  earlier tests' shared-stream positions are byte-for-byte unchanged). Pins the core
  invariant: fed the training corpus's **own** raw bytes, the held-out path reproduces
  `eval_corpus()` **bit-for-bit** (same bytes + same tokenizer ⇒ same id stream ⇒ same
  nats/bytes) for byte-level **and** BPE, restores `g_data`/`g_datalen`, and rejects a
  too-short buffer (−1) without indexing out of bounds. **977 → 986** checks, green on
  x86_64 **and** aarch64/qemu.

## [1.5.5] - 2026-06-13

**Hardening / audit / security pass (P(-1)) — closes the data-ingestion & curation
1.5.x arc.** A security/correctness audit of the surface the 1.5.x arc added: the
packed `g_data` token store + raised 64 MB cap (1.5.3), the `c4_sample.py` curation
script ingesting untrusted shards (1.5.1/1.5.2/1.5.4), and the char-diffusion path
(1.5.0). Adversarial multi-agent review — five read-only dimensions, each medium+
finding adversarially verified against the committed code. **One** confirmed finding,
fixed; four dimensions clean. The Cyrius binary is **unchanged** (the fix is
Python-side; CFG_VERSION bump only) — **977** checks unchanged.

### Fixed
- **Gzip-bomb / unbounded line buffering in `scripts/c4_sample.py`** (the one confirmed
  finding; medium). The script ingests untrusted gzipped JSON-lines shards and iterated
  them with `for line in gz:`, which materializes a whole line before yielding — a
  malformed/MITM'd member with a giant no-newline run (DEFLATE ~1000:1) decompresses to
  hundreds of MB in memory (**OOM**), and the `--max-bytes` guard runs only *after* the
  line is buffered. Fix: a bounded `iter_lines()` reading fixed 1 MB chunks, splitting
  on `\n` locally, **dropping** any runaway line over an 8 MB cap (real C4 docs are a
  few KB) and resyncing at the next newline — memory capped at ~chunk+max_line, the drop
  count reported (`oversized=N`), a valid doc after a bomb still recovered. Regression-
  verified: normal extraction unchanged (`oversized=0`); a 12 MB bomb completes (exit 0,
  no OOM, `oversized=1`); **re-curating the 4 MB sample is byte-identical to before**
  (well-formed shards yield identical text → X016/X017/X019 reproducibility preserved).

### Audit
- **`docs/audit/2026-06-13-1.5.x-hardening-audit.md`** (the ninth audit). Verdict
  **GO**, 0 residual blockers. Four dimensions returned zero findings with explicit
  coverage: the **packed store** (no integer overflow, no OOB, width-decision sound,
  alloc-fail handled, re-encode buffer sizing correct), the **corpus loaders + 64 MB
  cap** (clean over-cap rejection, every allocation under the 256 MB `ALLOC_MAX`), the
  **diffusion path** (in-bounds indexing, masked-CE div-by-zero guarded three ways, v7
  hostile-load rejects, NaN guard), and **checkpoint back-compat / memory** (packing is
  in-RAM only — format unchanged, v1–v7 load; the cap raise is a pure relaxation; the
  one-time u8→u16 widen is bounded). One finding refuted (the `--max-bytes`
  one-document overshoot — a documented streaming-stop semantic, binary hard-bounds
  ingestion regardless); two low advisories accepted (dedup sets bounded by
  `--max-bytes`; `--out` is expected dev-CLI behavior).

### Verified
- Cleanliness baseline + post-cut: `cyrius build` OK, lint 0 warnings, **977** grad-
  checks green on x86_64 AND aarch64/qemu; fuzz green; `make smoke` green; `make
  release` exit 0. Benchmark baseline captured (default step ~3.58 ms) — binary
  byte-identical across the cut, so no perf regression.

### Noted (deferred follow-on)
- A held-out **`--eval-corpus FILE`** flag (eval on a disjoint corpus through the
  loaded tokenizer) remains the clean way to score data's *generalization* value
  (X019's own-corpus-metric caveat) — a small additive feature for a future point
  release, out of this hardening pass's scope.

## [1.5.4] - 2026-06-13

**Curation at scale** (data-ingestion & curation 1.5.x arc, step 3; X019). The first
run to use 1.5.3's raised corpus ceiling: curate a **24 MB multi-shard** C4-en corpus
(6× the old 4 MB cap) and run the scaled data+capacity experiment vs the 4 MB
baseline. **Data + experiment only — the binary is unchanged** (CFG_VERSION bump for
`--version`; **977** checks unchanged).

### Added
- **X019** — scaled data+capacity experiment. Two curated C4-en corpora (4 MB/1-shard
  and **24 MB/12-shard**, both `c4_sample.py --curate`, BPE 256) × two model sizes
  (default ≈53 K, preset ≈232 K), matched compute. Eval bits/byte:

  | model | 4 MB curated | 24 MB curated |
  |-------|--------------|---------------|
  | default | 3.232 | 3.405 |
  | preset  | 2.666 | 2.741 |

  **Finding**: capacity is the dominant lever (default→preset −17.5% on 4 MB, −19.5%
  on 24 MB), and the diversity/volume penalty **halves with capacity** — the tiny
  model pays +5.4% bits/byte on the bigger diverse corpus (and its samples get more
  garbled), the preset only +2.8% (fluent, richer vocabulary). First attn11 evidence
  that diversity/volume starts paying off as capacity grows — validating the roadmap's
  sequencing of streaming + larger corpora with model scale (M16+). The 4 MB default
  cell **reproduces X017's 3.232 bit-for-bit**, re-confirming 1.5.3's transparency.
  Full analysis + the honest own-corpus-metric caveat in `docs/development/experiments.md`.

### Changed
- **`docs/examples/c4-english.md`** — added a "Curation at scale (1.5.4)" section with
  the recipe (`--shards 12 --max-bytes 24000000`) and the X019 grid.

### Noted (follow-on, not in this release)
- A held-out **`--eval-corpus FILE`** flag (eval on a disjoint corpus through the
  loaded tokenizer, no vocab-order check) is the clean way to score data's
  *generalization* benefit — bits/byte-on-own-corpus understates it. A small additive
  1.5.x follow-on; deferred to keep 1.5.4 binary-unchanged.

## [1.5.3] - 2026-06-13

**Token-packing unlock** (data-ingestion & curation 1.5.x arc, step 2; X018). The
corpus token stream `g_data` was one **i64 per token (8 B)**, so a 4 MB corpus cost
~32 MB and the in-RAM ceiling was ~32 MB against the 256 MB single-allocation cap.
1.5.3 stores it **packed** — `u8` for byte-level (vocab ≤ 256), `u16` for BPE
(vocab ≤ 768) — removing the 8×/4× bloat and lifting the cap to **64 MB** (the u16
`g_data` is then 128 MB, half the per-alloc cap). The model/training math is
untouched: the same token ids feed the same forward, so the **default run is
byte-identical** (verified — default/u8, BPE-64/u16, and `--preset` all match
1.5.2 byte-for-byte).

### Added
- **Packed `g_data` token store** — a `g_data_w` width global (1 = u8, 2 = u16) +
  width-generic accessors `_tw_ld`/`_tw_st`(buf, i, w) and `gd_ld`/`gd_st`(token
  index) over a byte buffer (`load8`/`store8`, `load16`/`store16` — zero-extending
  loads, low-byte stores; the same builtins `lib/cffi`/`slice`/`net`/`dynlib` use).
  Only `g_data` (the one buffer that scales with corpus size) is packed; the
  T-sized `A_tokens`/`A_targets`/`A_mask`/`G_win`/`g_enc_buf` stay i64. `main` sets
  the width before the corpus load (byte-level → u8, `--bpe` or `--load` → u16);
  `bpe_learn` self-widens a u8 corpus to u16 if a caller skipped that (BPE mints ids
  up to base+K−1 ≤ 766, which need u16). `tok_encode` gained an explicit output-width
  arg so it serves both the i64 prompt scratch (`g_enc_buf`, width 8) and the packed
  corpus rebuild (`g_data`, `g_data_w`).
- **`test_token_packing`** — pins the u8/u16 accessor round-trip (incl. boundary ids
  255 / 256 / 767), **width-invariance** (the same byte-level corpus yields IDENTICAL
  token ids at u8 and u16), and the `bpe_learn` self-widen. RNG-neutral, registered
  last so the diffusion group's pinned RNG stream is undisturbed. **966 → 977** checks.
- **X018** — the packing measurement: 8 B/token → 1 B (byte) / 2 B (BPE); corpus
  ceiling ~32 MB → ~128 MB (u16) / ~256 MB (u8) against the per-alloc cap; the
  `MAX_CORPUS_BYTES` raise to 64 MB; byte-identical default run confirmed.

### Changed
- **`MAX_CORPUS_BYTES` 4 MB → 64 MB** (`--corpus`/`--stdin` cap). Chosen so even the
  u16 corpus store (128 MB) sits at half the 256 MB single-allocation cap, with room
  for `g_text` (64 MB) + the model budget. An over-cap corpus still rejects cleanly
  (code −2, no crash) — verified at 65 MB.
- Docs/help updated to the 64 MB cap (`src/main.cyr` usage, `docs/STABILITY.md`,
  `docs/guides/getting-started.md`, `docs/examples/c4-english.md`).

### Verified
- **Byte-identical** default/u8, BPE-64/u16, and `--preset` runs vs 1.5.2.
- **977** grad-checks green on **x86_64 AND aarch64/qemu**; lint clean; fuzz green
  (100 random corpora + BPE round-trip); `make smoke` (hostile-CLI) green.
- A 6 MB corpus (byte + BPE) loads and trains (was over the old 4 MB cap); a 65 MB
  corpus rejects cleanly.

## [1.5.2] - 2026-06-13

**Quality-curating C4 sampler** — first step of the data-ingestion & curation 1.5.x
arc (per `docs/development/roadmap.md`): a tiny model is data-rich and capacity-poor,
so "data quality > volume". Tooling + data + an A/B experiment; no core binary change.

### Added
- **`scripts/c4_sample.py --curate`** — de-duplication (exact full-text + a 120-char
  normalized prefix), **multi-shard sampling** (`--shards N` spreads sampling across
  the crawl for diversity, not one consecutive block), and prose/register quality
  filters (letter & digit ratios, terminal punctuation, average word length,
  repetition, long-token spam). All stdlib + deterministic (stable hashing, no RNG);
  the multi-shard download is resilient (per-shard retries, skip-on-failure, a
  remaining-bytes budget that rebalances across a dropped shard). Defaults (no
  `--curate`, one shard) reproduce the 1.5.1 raw slice **byte-for-byte** (verified),
  so it stays a clean A/B baseline.
- **X017** — the raw-vs-curated A/B at iso-compute (default config + BPE 256). The
  quality filter alone (same shard) cut eval **bits/byte 3.43 → 3.23 (−5.9%)**;
  multi-shard diversity *raised* it (+2.7%) — a tiny model can't exploit diversity it
  can't fit, so **curate for quality now, diversity is a scale lever** (validates
  sequencing streaming + larger corpora with M16+). Recipe in
  `docs/examples/c4-english.md`.

## [1.5.1] - 2026-06-13

**C4 English experiment** — tooling + example for training attn11 on a real
large-corpus slice (the C4 `c4/en` dataset). No core binary change; a data-prep
helper, an example pipeline, and an experiment log. Demonstrates the diffusion-era
trunk (and the AR default) learning English statistics off web text.

### Added
- **`scripts/c4_sample.py`** — stream a small English sample out of C4 into a plain
  UTF-8 corpus file for `--corpus`, with **zero third-party deps** (stdlib `gzip` +
  `json`). C4's TFDS `c4/en` config is 305 GB and needs an Apache-Beam prep; instead
  the script streams ONE of C4's public gzipped JSON-lines shards (`allenai/c4`, the
  raw mirror of the same corpus) and stops at `--max-bytes`, so it downloads only ~a
  few MB, not the 319 MB shard. Reads a URL (urllib) or a piped shard (`--url -`).
- **`docs/examples/c4-english.md`** — the stream → train → sample recipe, why the
  full TFDS pipeline is skipped, honest expectations (a tiny model produces
  English-*flavored* text, not fluent prose), and measured results.
- **X016** — the C4-English run: a 53 K-param default+BPE model over a 4 MB C4 slice
  reaches 3.43 bits/byte and samples recognizable broken English; the `--preset`
  (232 K params, ctx 64) on the same slice reaches 2.70 bits/byte with visibly more
  word-shaped output — same data, better model ⇒ better English.

### Changed
- **`.gitignore`** — ignore `/data/` (sampled corpora + shards are large and
  regenerable; never committed).

## [1.5.0] - 2026-06-13

**Char-diffusion objective** (M15, E5; ADR 0013) — the first *training-objective*
departure from the autoregressive trunk. `--objective diffusion` trains a masked
absorbing-state diffusion model (D3PM/MDLM lineage): drop the causal mask, corrupt
a window by replacing positions with a learned `[MASK]` embedding, and predict the
originals with a **bidirectional** model; decode is MaskGIT-style confidence-ordered
parallel unmasking, not left-to-right. Opt-in and additive — the no-flag AR run is
**byte-identical**, every v1–v6 checkpoint still loads, and the milestone's two
hand-derived backwards (the masked-CE and the bidirectional attention) each land
behind a finite-difference grad-check.

### Added
- **`--objective ar|diffusion`** (default `ar`) + `--decode-steps N` /
  `--decode-schedule {cosine,linear}`. Scope for v1.5.0: MHA (GQA allowed) +
  learned-abs + dense + uniform (lin/ssm are causal recurrences; mla/MoE/RoPE-
  diffusion are additive fast-follows).
- **Masked cross-entropy** (`softmax_xent_masked_fwd`/`_bwd`, `src/ops.cyr`): mean
  CE over the masked positions only, gradient zeroed at given positions. Grad-check
  `test_masked_ce` (all-ones ≡ plain CE; all-zeros ⇒ 0).
- **Bidirectional attention** (`g_bidir`, `src/attn.cyr`): the causal `j<=i` loop
  bound becomes the full square under diffusion; the softmax-backward sum uses the
  same range. Grad-check `test_attn_bidir` (T≥3, with a future-attending pin that
  catches a fwd+bwd-both-causal regression the FD alone would miss).
- **Learned `[MASK]` embedding** (`src/model.cyr`): a `C`-vector appended after
  `lnf_b`, present only under diffusion (`g_NP += C` gated), so AR layout / RNG draw
  order / Adam state / checkpoints are byte-identical; the weight-tied head is
  untouched. Full-model grad-check `test_model_diffusion` (FDs both `mask_emb` and
  `tokemb` with mixed masked/given positions).
- **Diffusion sampler + decode** (`src/train.cyr`): `sample_window_diffusion` /
  `diffuse_mask` (per-example mask ratio t~U(0,1), ≥1-mask floor); `gen_diffusion`
  (uncached bidirectional, cosine/linear schedule, greedy + frozen tie-break,
  deterministic). Decode-determinism test.
- **`eval_diffusion` / `eval_diffusion_grid`**: denoising bits/byte at a mask-ratio
  grid {10,30,50,70,90}% + the ELBO bound (the mean — the MDLM 1/t weight cancels
  the t-scaling of the masked count). `--eval` reports this for a diffusion model.
- **Checkpoint v7**: the `objective` descriptor at slot [22] + the `+C` mask_emb in
  NP. v7 round-trip + `-47` rejection tests; 500 v7 fuzz rounds. A diffusion image
  writes v7; uniform AR still writes v5 and hybrid AR v6 (byte-identical).
- Diffusion bench case (fwd+bwd step + T-round parallel decode); **X015** (the
  honest matched-compute AR-vs-diffusion comparison).

### Changed
- `model_init_full` gains a trailing `objective` param (the delegators pass 0);
  `persist.cyr` accepts versions 1..7 (v≤6 synthesize `objective = 0`).

### Notes
- Diffusion mode is bidirectional, so attention is O(T²) (vs causal's half). At
  reference scale the greedy decode tends to collapse toward high-frequency tokens —
  a known dLLM-at-small-scale limitation, not a correctness issue (the grad-checks
  are tight); see X015 for the honest comparison and the ELBO-vs-exact-NLL caveat.

## [1.4.6] - 2026-06-13

**Benchmarking pass.** A dedicated perf release: one canonical bench run across the
whole mixer family on the current binary, the time-series and the perf doc brought
current (both had stalled — the CSV at 0.9.0, `docs/benchmarks.md` had no
MoE/linear/SSM/hybrid sections), and the rung-d padded-layout cost pinned. No
behavior change — measurement + docs, plus two param-count prints in the bench.

### Added
- **`docs/benchmarks.md`** — the missing sections: MoE (1.3.0), the sequence-mixer
  family (linear/SSM, 1.4.0/1.4.2), and the per-layer hybrid + padded layout
  (1.4.3/1.4.4), with a unified comparison table (step time, cached-decode latency,
  decode-cache bytes, scaling, params) from one canonical run, and the
  attention-fraction cache curve.
- **`bench-history.csv`** — a current `1.4.6` row (the time series was stale at
  0.9.0). The default-config training step is flat from 0.4.0 → 1.4.6 (~3.6 ms,
  ~4 450 tok/s b=16): the M12–M14 arc added five opt-in axes with zero regression
  to the no-flag path.
- **`X014`** (experiments ledger) — perf consolidation: the latency/cache/param
  table across all mixers + the headline finding that the **rung-d padding is
  memory-only** (the mha/ssm 1/3 hybrid step, 4.90 ms, equals the per-layer mix
  (1·MHA + 2·SSM)/3 = 4.94 ms — the zeroed pad is never read, so it costs params +
  Adam moments, not FLOPs).
- Bench harness: param-count prints for the two hybrid entries (quantifying the
  padding cost — mha/ssm +1 440 params / ~4% vs pure SSM; mha/lin free).

### Notes
- Confirmed: linear ≈ MHA in step/decode (constant-T cache); SSM ~1.58× the step
  (the O(T·C·N) scan) for a constant C·N cache; MoE ~1.95× the step at top-2 with
  5.5× the params (N=8). Numbers are the default config (V=25, C=32, T=16, NL=3),
  x86_64, stable to a few percent.

## [1.4.5] - 2026-06-13

**Hardening pass (P(-1)) — closes the 1.4.x arc.** A security/correctness audit of
the 1.4.x surface (the per-layer hybrid dispatch, the rung-d padded uniform-stride
layout, checkpoint v6, and the `--attn-every` CLI) via an adversarial multi-agent
review: five independent read-only dimensions, each finding then adversarially
verified against the committed code and live repro builds. **One** confirmed
finding (a CLI memory-safety bug), fixed; the other four dimensions came back
clean. No behavior change to training, the checkpoint format, or any valid run.

### Fixed
- **CLI stack-buffer overflow (security).** `--attn-every K` filled the fixed
  128-slot stack buffer `lkinds[1024]` with one entry per layer, but `--layers`
  (`cfg_nl`) was not bounded before that loop — the NL≤128 cap is enforced inside
  `model_init`, which runs *after*. So `--layers N --attn-every K` with N>128 (no
  `--load`) wrote 8·N bytes past the buffer — confirmed **SIGSEGV** at
  `--layers 100000 --attn-every 2`. Same class as the M7 `--layers` heap-OOB. Fixed
  with a `cfg_nl` bound (`1..128`, the buffer/`model_config_ok` cap) before the
  loop; the exploit now rejects cleanly. (`--layers` without `--attn-every` was
  always safe — `model_config_ok` rejects it before any allocation.)

### Added
- **`make smoke`** (in the `release` gate): a CLI regression that asserts hostile
  `--layers`/`--attn-every` combinations reject cleanly (exit < 128, never a
  signal) and a valid hybrid still builds — the grad-check and fuzz harnesses cover
  the math and the checkpoint loader but not the CLI arg path, which this closes.
- **Audit report** `docs/audit/2026-06-13-1.4.x-hardening-audit.md` (5 dimensions,
  dispositions, the finding + fix + regression, accepted residuals). The checkpoint
  format spec comment (persist.cyr) updated to document the **v6** per-layer region
  + v1–v5 back-compat (uniform models still write v5, byte-identical).

### Verified (audit, no change needed)
- Hostile v6 images are rejected cleanly (`-46`/`-10`/`-11`/`-18`); the per-layer
  region read is bounded before access; the padded-`np` math cannot overflow under
  the dim caps; v1–v5 load unchanged; the no-flag run stays byte-identical; the
  padded-region tail is zero by construction (`t_alloc` zeroes) and untrained; the
  mixed hybrid backward + cached decode are grad-/bit-checked (X012/X013 suites).

**Any-mixer hybrids (M14 rung d, E4) — completes M14.** Lifts 1.4.3's layout
restriction so a hybrid can interleave ANY of the four mixers `{mha, mla, lin, ssm}`
— including the survey's strongest pairing, full attention ⊕ the selective SSM
(attn11's best single mixer). The trick: each block's K/V weight region is **padded
to the max `_kvw` over the kinds present** (`_kvw_hyb`), so the per-block stride
stays uniform — only `_kv_weight_size()` plus the per-layer weight-init and cache
gates change, NOT every within-block offset (no per-layer-offset refactor). A
layer of a smaller kind tiles its own weights and leaves a zeroed pad. Opt-in and
additive — a no-flag run is byte-identical; checkpoint v6 is unchanged in shape
(it already carries the per-layer kinds, 1.4.3), the loader just sizes the padded
block the same way. `{mha, gqa, lin}` hybrids stay exact (shared `_kvw`, no pad).

### Added
- **`--attn-every K` now accepts any base** (`--attn-kind {mha, lin, mla, ssm}`):
  a full-attention block every K-th layer, the base mixer elsewhere. For an
  mla/ssm base the shared `--latent-dim` applies to those layers; the MHA blocks
  ignore it. (1.4.3 allowed only {mha, lin} bases.)
- **`_kvw_hyb`** (model.cyr): the padded per-block K/V-region size = max `_kvw`
  over the present per-layer kinds. `_kv_weight_size()` returns it for a hybrid
  (the single kind's `_kvw` for a uniform model — byte-identical). The weight-init
  loop branches per layer on `_lk(L)`; per-kind caches (`LA_c`, the ssm caches,
  `g_lin_state`) are held when ANY layer uses that kind (`_any_kind`).
- **`ckpt_expected_np_kvw`** (persist.cyr): the np computation for a given
  K/V-region size, so the v6 loader sizes a padded hybrid block exactly as
  model_init lays it out (`_kvw_hyb`); `kv_cache_bytes` sums the per-layer caches
  for all four kinds (mha K/V, lin/ssm constant state, mla latent).
- Tests: `test_model_hybrid_ssm` + `test_model_hybrid_mla` (the MIXED
  SSM/MLA ⊕ MHA full-model grad-checks through the padded layout, ~1e-4);
  `test_kv_hybrid` extended with mha/ssm + mha/mla bit-identity decode;
  `test_ckpt_hybrid` extended with a padded mha/ssm v6 round-trip; alloc-accounting
  + config-cap pins for mla/ssm hybrids. **857 → 907** checks. X013 (the
  attention ⊕ SSM ratio sweep) + a mha/ssm bench entry. ADR 0012.

### Changed
- `_hybrid_kinds_ok` drops the `_kvw`-equality (uniform-stride) requirement —
  padding removes it — and keeps the cross-cutting constraints: learned-abs
  positions, full heads (`nkv == nh`), and a valid shared latent iff any mla/ssm
  layer. `model_alloc_bytes_hyb` takes the per-layer `kinds` pointer (replacing the
  1.4.3 `lin_extra`/`pl` flags) and counts each present kind's caches. The summary
  print shows the real hybrid base (mha/mla/lin/ssm). The MHA/MLA/lin/SSM/MoE paths
  and the no-flag run are unchanged; `{mha,gqa,lin}` hybrids are byte-identical to 1.4.3.

### Performance
- Attention-fraction sweep, base SSM (default config, 1200 steps; the padding lifts
  the hybrid param count to MHA's 39 488 vs pure SSM's 38 048):

  | attention | config             | bits/byte | decode cache |
  |-----------|--------------------|-----------|--------------|
  | 0/3 (0%)  | pure ssm           | **0.218** | 12 288 B     |
  | 1/3 (33%) | ssm --attn-every 3 | 0.224     | 16 384 B     |
  | 2/3 (67%) | ssm --attn-every 2 | 0.219     | 20 480 B     |
  | 3/3 (100%)| pure mha           | 0.279     | 24 576 B     |

  The decode cache is a continuous knob from pure-SSM's constant `C·N` (12 288 B) to
  pure-MHA's `∝T` K/V (24 576 B) — each attention layer trades constant state for
  T-growing K/V. bits/byte within noise across ratios (all under pure-MHA, near pure
  SSM) — a "trains + grad-checks", not a scaling claim. Hybrid fwd+bwd step ~5.0 ms.

## [1.4.3] - 2026-06-13

**A per-layer mixer hybrid (M14 rung c, E4) — interleave attention with a cheap
mixer.** `--attn-every K` places a full-attention (MHA) block at every K-th layer
and a gated-linear block elsewhere — the survey's "a few attention layers among
many cheap recurrent ones" structural lever. The global `attn_kind` becomes a
per-layer `g_layer_kind`, read only by the three mixer-dispatch helpers. The
hybrid is restricted to LAYOUT-COMPATIBLE kinds {mha, gqa, lin} (gated-linear
reuses MHA's projections, ADR 0009), so the per-block stride stays uniform — no
per-layer parameter-offset refactor, and the hybrid is **parameter-free**. What it
buys is a knob on the decode cache: the attention fraction sets how much of the
cache is T-growing K/V vs constant state. Opt-in and additive — a no-flag run is
byte-identical. First checkpoint **format bump (v6)**: a hybrid carries its
per-layer pattern; uniform models still write v5, byte-identical.

### Added
- **`--attn-every K`**: a full-attention block every K-th layer (`L % K == 0`),
  the `--attn-kind` base (mha/lin) elsewhere. The attention-fraction knob for the
  hybrid-ratio sweep. Collapses to a uniform model when the pattern is one kind.
- **Per-layer dispatch** (`g_layer_kind` + `_lk(L)`): the three `_attn_block_*`
  helpers read the per-layer kind; uniform models (`g_layer_kind == 0`) get the
  global back, so this is byte-identical until a hybrid is built. `model_init_full`
  carries the per-layer array; `model_init_moe` is the uniform delegator.
- **Hybrid validity** (`_hybrid_kinds_ok`): the uniform-stride invariant — every
  layer's `_kvw` must equal the base's (so np/stride are unchanged), base in
  {mha, lin}, learned-abs positions. Rejects MLA/SSM layers (different `_kvw`).
- **Checkpoint v6** carries the NL per-layer kinds (after the fixed header, before
  the vocab); the loader rebuilds the hybrid and rejects an image whose per-layer
  kind breaks the invariant (`-46`). v≤5 synthesize the uniform default.
- Hybrid tests: `test_model_hybrid` (mixed-mixer full-model grad-check, ~1e-5),
  `test_kv_hybrid` (cached-decode bit-identity, two interleavings), `test_ckpt_hybrid`
  (v6 round-trip + `-46` rejects), config-cap + alloc-accounting pins. **801 → 857**
  checks. Hybrid-ratio sweep (X012) + a hybrid bench entry. ADR 0011.

### Changed
- `kv_cache_bytes` SUMS the per-layer caches for a hybrid (T-growing K/V for the
  attention layers + constant lin state for the rest). `g_lin_state` is allocated
  whenever ANY layer is linear; `model_alloc_bytes_hyb` accounts the per-layer
  array + the lin-state extra. `_gen_bits` factored into a shared `_gen_verify`
  body (uniform + hybrid drivers). The MHA/MLA/lin/SSM/MoE paths and the no-flag
  run are unchanged.

### Performance
- Attention-fraction sweep (default config, 1200 steps, embedded corpus;
  parameter-identical across the sweep — the hybrid only redistributes the cache):

  | attention | config             | bits/byte | decode cache | vs MHA |
  |-----------|--------------------|-----------|--------------|--------|
  | 0/3 (0%)  | pure lin           | 0.239     | 6 144 B      | 0.25×  |
  | 1/3 (33%) | lin --attn-every 3 | 0.244     | 12 288 B     | 0.50×  |
  | 2/3 (67%) | lin --attn-every 2 | **0.234** | 18 432 B     | 0.75×  |
  | 3/3 (100%)| pure mha           | 0.279     | 24 576 B     | 1.00×  |

  The decode cache scales with the attention fraction (the lever); at this
  reference scale the bits/byte spread is within noise (all hybrids beat pure-MHA,
  edge pure-lin) — a "trains + grad-checks", not a scaling claim. Hybrid fwd+bwd
  step ~3.73 ms (≈ the linear step; two of three blocks are linear).

## [1.4.2] - 2026-06-12

**Selective state-space model (M14 rung b, E4) — the third sequence mixer.**
`--attn-kind ssm` adds a minimal Mamba-lite diagonal SSM: a per-channel `N`-state
recurrence whose Δ, B, C are all functions of the input (the *selective* scan),
with the `exp(Δ·A)` discretization and a learned diagonal `A`. The earned
milestone is the **hand-derived BPTT through the data-dependent scan** —
grad-checked bit-tight. Like gated linear attention, the decode cache is the
constant `C·N` state, not a T-growing K/V. Opt-in and additive — a no-flag run is
byte-identical; it lands in its own `attn_ssm.cyr` (the 1.4.1 pattern).

### Added
- **`--attn-kind ssm`** (`attn_ssm.cyr`): the selective SSM. Per channel c, state
  `h_t[c,n] = exp(Δ_t[c]·A[c,n])·h_{t-1}[c,n] + Δ_t[c]·B_t[n]·a_t[c]`,
  `y_t[c] = Σ_n C_t[n]·h_t[c,n] + D[c]·a_t[c]`, with `Δ = softplus(a·W_dt)`,
  `B = a·W_B`, `C = a·W_C` (selective). Reuses Wq (as W_dt) and Wo (output
  proj) — so it rides `attn_kind = 3` and **checkpoint v5 carries it (no format
  bump)**, with the state size `N` reusing the `latent_dim` field. A inits to a
  negative ramp (S4D-style decay spread), D to 1 (identity skip).
- **Hand-derived BPTT** (`ssm_fwd`/`ssm_bwd`): the reverse scan accumulates `dh`
  through the data-dependent `exp(Δ·A)` transition; Δ/B/C/A/D/x all receive
  gradient (the selectivity). Grad-checked (`test_ssm_core`, ~1e-7) on every
  parameter + the input. Constant-state cached decode (`ssm_fwd_row`) bit-identical
  to the batch scan (`test_kv_ssm`).
- Mixer comparison (X011: SSM vs linear vs MLA vs MHA) + an SSM bench entry. ADR
  0010 (the selective-SSM design + the Wq/Wo + latent_dim reuse).

### Changed
- `_kvw` gets the `attn_kind == 3` region (`3·C·N + C` = A/W_B/W_C/D); `_o_ssm*`
  offsets; the 3 dispatch helpers + config/init/alloc/persist gain the SSM case.
  `kv_cache_bytes` reports the constant `NL·C·N` state for the SSM. The MHA/MLA/
  linear/MoE paths and the no-flag run are unchanged.

### Performance
- Mixer comparison (default config, 1200 steps, embedded corpus):

  | mixer | bits/byte | params | decode cache |
  |-------|-----------|--------|--------------|
  | MHA   | 0.279     | 39 488 | 24 576 B (∝ T) |
  | MLA   | 0.273     | 37 952 | 6 144 B (∝ T)  |
  | linear| 0.239     | 39 488 | 6 144 B (const)|
  | **SSM** | **0.218** | 38 048 | 12 288 B (const) |

  At this reference scale the selective SSM edges every other mixer on bits/byte,
  with a cache that is constant in T (8× under MHA at the preset T=64). Train step
  ~5.6 ms (the O(T·C·N) scan, ~1.56× the dense ~3.6 ms); cached gen ~258 µs/token.

### Tests
- **727 → 801** checks (x86_64 AND aarch64/qemu): `test_ssm_core` (per-op BPTT,
  ~1e-7), `test_model_ssm` (full-model 1e-3), `test_kv_ssm` (cached bit-identity),
  `test_ckpt_ssm` (v5 round-trip + `-40`/`-41`/`-42` rejections), SSM alloc-accounting
  + config-cap pins.

### Notes
- M14's remaining rung (c) — per-layer mixer interleaving (the hybrid-ratio sweep)
  — is next: turn the global `g_attn_kind` into a per-layer kind read inside the
  three dispatch helpers (the 1.4.1 refactor localized that). A vidya-scale bake-off
  across all four mixers is the natural follow-on X-entry.

## [1.4.1] - 2026-06-12

**Refactoring sweep — no behavior change.** A maintenance release that reorganizes
the mixer/attention machinery so the upcoming M14 rungs (a selective SSM, then
per-layer mixer interleaving) are cheap to add. The no-flag run is byte-identical
(loss 0.31234/0.22803 at steps 250/500, unchanged), **727 checks** unchanged on
x86_64 AND aarch64/qemu, and every checkpoint round-trips identically. Each step
landed behind the green gate (grad-checks + byte-identical default + both arches).

### Changed
- **One mixer-dispatch point** (`model.cyr`): the `if/elif (attn_kind…)` attention
  sublayer selection — previously inlined in `model_forward`, `model_eval_window`,
  `model_backward`, and `model_fwd_row` — is now three helpers (`_attn_block_fwd`,
  `_attn_block_bwd`, `_attn_block_fwd_row`). A new mixer adds one case in each, not
  in four functions; the M14 per-layer hybrid flips `g_attn_kind` to a per-layer
  lookup in one place.
- **Shared per-block param arithmetic** (`model.cyr` + `persist.cyr`): the K/V and
  MLP weight-region sizes are now pure descriptor-parameterized helpers
  (`_kvw`/`_mlpw`), used by both the offset helpers (`_kv_weight_size`/
  `_mlp_weight_size` wrap them) and the checkpoint validator (`ckpt_expected_np`).
  One source of the block-layout sizing — the M13/M14 model↔persist keep-in-sync
  hazard is gone (a new mixer's param region is defined once).
- **`attn_linear.cyr`**: the gated-linear mixer (`lin_core_*`, `attn_lin_*`) is
  split out of `attn.cyr` into its own file, establishing the one-file-per-mixer
  pattern (the M14 selective SSM lands as its own file next). `attn.cyr`
  1266 → 976 lines; entries gained an `include "src/attn_linear.cyr"`.
- **Test dedup**: the six near-identical 58-line `_gen_bits_*` cached-vs-uncached
  bit-identity helpers collapse to one parameterized `_gen_bits` driver + six thin
  one-line wrappers (call sites unchanged). `attn11.tcyr` −~280 lines.

### Notes
- Pure reorganization: `src/*.cyr` is identical in effect to 1.4.0 (no new flags,
  no checkpoint change — saves stay v5). Next: M14 rung (b), a minimal selective
  SSM (`attn_kind = 3`, `attn_ssm.cyr`), then rung (c), per-layer mixer
  interleaving + the hybrid-ratio sweep.

## [1.4.0] - 2026-06-12

**Gated linear attention (M14 rung a, E4) — the arc's first non-softmax sequence
mixer.** `--attn-kind lin` replaces the softmax/PV core with a causal RetNet-style
**retention recurrence** `S_t = γ_h·S_{t-1} + k_t⊗v_t`,
`out_t = (1/√hd)·S_t^T q_t`, with a fixed per-head decay `γ_h = 1−2^{−(3+h)}`
(parameter-free, like RoPE). It reuses the MHA Q/K/V/O projections, so it rides
the existing `attn_kind` descriptor (value 2) and **checkpoint v5 — no format
bump**. The headline: the decode cache is the **constant** `nh·hd²` retention
state, not a T-growing K/V. Opt-in and additive — a no-flag run is byte-identical.

### Added
- **`--attn-kind lin`**: gated linear attention (full heads, learned-abs
  positions; RoPE is softmax-only). New core `lin_core_fwd`/`lin_core_bwd` +
  wrappers `attn_lin_fwd`/`attn_lin_bwd` (`attn.cyr`). The hand-derived backward
  needs **no state caching** — `dq` via a forward S-recompute, `dk`/`dv` via a
  reverse `dS` accumulator. Pure multiply/add (no softmax/exp), so the grad-check
  is bit-tight (`test_lin_core` ~1e-9) and there's no x86-only trig/exp.
- **Constant-state cached decode** (`lin_core_fwd_row`/`attn_lin_fwd_row`,
  per-layer `g_lin_state`): one recurrence step per token against the persistent
  `nh·hd²` state (reset at window start / context-shift re-prime). Bit-identical
  to the uncached batch scan (`test_kv_lin`). `kv_cache_bytes` reports the state
  size — independent of T.
- A mixer comparison (X010, MHA vs MLA vs linear) + a linear bench entry
  (train-step, cached-gen, constant cache bytes). **ADR 0009** (the sequence-mixer
  axis: `attn_kind` as a mixer family, RetNet retention, fixed decay, MHA-layout reuse).

### Changed
- `model_config_ok` / `model_alloc_bytes` / the persist loader accept
  `attn_kind == 2` (full heads, no latent, learned-abs); `attn_arena_size` carries
  a small `2·hd²` lin scratch (S/dS), `model_alloc_bytes` the per-layer state cache.
  The MHA/MLA/MoE paths and the no-flag run are unchanged.

### Performance
- Mixer comparison (default config, 1200 steps, embedded corpus):

  | mixer | bits/byte | params | decode cache (B) |
  |-------|-----------|--------|------------------|
  | MHA   | 0.279     | 39 488 | 24 576 (∝ T)     |
  | MLA   | 0.273     | 37 952 | 6 144 (∝ T)      |
  | **linear** | **0.239** | 39 488 | **6 144 (constant in T)** |

  Linear matches MHA's parameter count (parameter-free over the projections) and,
  at this reference scale, edges it on bits/byte while holding a **constant** cache
  — at the preset (T=64) that's 16 384 B vs MHA's 262 144 B (16×). Train step
  ~3.8 ms (~6% over the dense ~3.6 ms); cached gen ~160 µs/token (the O(hd²) state
  update beats the O(T·hd) cache scan).

### Tests
- **673 → 727** checks (x86_64 AND aarch64/qemu): `test_lin_core` (per-op, 1e-9),
  `test_model_lin` (full-model 1e-3, incl. the now-real K-bias gradient),
  `test_kv_lin` (cached-vs-uncached bit-identity across shifts), `test_ckpt_lin`
  (attn_kind=2 round-trip + `-40`/`-41`/`-42` rejections), plus linear
  alloc-accounting and config-cap pins.

### Notes
- The retention backward is the only new hand-derived math; everything else
  composes grad-checked pieces. M14's rungs (b) a minimal selective SSM and
  (c) per-layer mixer interleaving (the hybrid-ratio sweep) are the follow-on
  increments; a vidya-scale perplexity bake-off across the mixer/attention axes is
  the natural next X-entry.

## [1.3.0] - 2026-06-12

**Mixture of Experts (M13, E8) — the first FFN-density axis on the 1.x arc.** The
dense GELU MLP in each block becomes **N experts + a top-K router**
(`--experts N --expert-topk K`), decoupling parameter count from per-token FLOPs.
The earned milestone is the **router backward**: a discrete top-K pick (frozen,
lower-index tie-break — bit-reproducible cross-arch) feeds a renormalized softmax
combine (Mixtral-style; the normalizer cancels), and a Switch-style load-balance
auxiliary loss keeps the experts from collapsing — **both hand-derived and
finite-difference grad-checked**. Opt-in and additive: `--experts 1` is the dense
baseline, so a no-flag run is **byte-identical**. Checkpoint **v5** records the
descriptor; v1–v4 still load.

### Added
- **`--experts N` / `--expert-topk K`** (N in 1..256; 1 = dense; K active
  experts/token, default 2). Per block: a bias-free router gate `C → N`, N experts
  each the dense `(C→F, GELU, F→C)` quad, packed array-of-structs (expert 0 aliases
  the dense MLP offsets, so the dense layout is unchanged). Output is the
  gate-weighted sum over the top-K experts.
- **Router combine op** (`moe_fwd`/`moe_bwd`, `ops.cyr`): router logits → top-K
  (frozen tie-break) → softmax over the selected logits → expert MLPs, gate-weighted.
  The backward sends gradient ONLY to the selected logits (straight-through for the
  discrete pick), composes each selected expert's MLP backward scaled by its gate,
  and accumulates tokens that share an expert. Per-op grad-checked (`test_moe_op`,
  1e-4) incl. top-1 (renorm gate ≡ 1 → zero combine gradient) and K=N edges.
- **Load-balance aux loss** (`moe_aux_fwd`/`moe_aux_dr`/`moe_aux_bwd`, Switch,
  arXiv:2101.03961): `L_aux = α·N·Σ fᵢ·Pᵢ` (α = 0.01), with the dispatch fraction
  `fᵢ` held constant (straight-through) and the mean router prob `Pᵢ`
  differentiated. Grad-checked against finite differences (`test_moe_aux`, 1e-5);
  added to the CE training loss, off the eval/bits-per-byte path.
- **Checkpoint v5** (`persist.cyr`): header slots `[20] num_experts`,
  `[21] expert_topk`; `ckpt_expected_np_moe` mirrors the expert+gate layout; new
  hostile-rejection codes `-44` (num_experts out of `1..256`) / `-45`
  (topk out of `1..N`), both bound **before** allocation. v1–v4 load (synthesize
  the dense MLP). Round-trip + rejection tests (`test_ckpt_moe`, `test_ckpt_v4_compat`).
- **Expert-utilization metric**: per-expert dispatch histogram + `moe_entropy`
  (normalized routing entropy, 1 = balanced, 0 = collapse), accumulated over the
  `--eval` pass and printed alongside total / per-token-active params.
- **Density-sweep experiment** (X009, `scripts/moe-sweep.sh`) + a MoE bench entry
  (train step + cached-gen + param count, `attn11.bcyr`).

### Changed
- **Toolchain pin `6.2.1 → 6.2.2`** (`cyrius.cyml`), with `cyrius update`
  resyncing `./lib/` (byte-identical snapshot — a clean patch realign; the pin and
  snapshot move together). Re-verified green on the realigned compiler.
- `model_init_moe` / `model_config_ok_moe` / `model_alloc_bytes_moe` /
  `ckpt_expected_np_moe` extend the `_arch` forms with `(num_experts, topk)`; the
  `_arch` forms delegate with the dense default, so every existing caller is
  unchanged. `eval_corpus` now reports **pure cross-entropy** bits/byte (subtracts
  the aux term; byte-neutral for the dense path).

### Performance
- MoE train step (default config, 8 experts, top-2): **~6.9 ms** vs the dense
  **~3.6 ms** (top-2 = two active expert MLPs + the router); 215 648 params vs the
  dense 39 488. Cached generation ~273 µs/token.
- Density sweep (default config, 1200 steps, embedded corpus, top-2): total params
  scale ~linearly with N (39 K → 1.62 M at N=64) while **per-token-active params stay
  ~65–71 K**; **routing entropy 0.993–0.999** (the aux loss keeps load balanced even
  at N=64); bits/byte best at N=8–16, rising past N=16 (over-parameterization at
  reference scale — the expected honest caveat). See X009.

### Tests
- **572 → 673** checks (x86_64 AND aarch64/qemu): `test_moe_op`, `test_moe_aux`,
  `test_model_moe` (full-model, 1e-3), `test_param_layout_moe`, `test_kv_moe`
  (cached-vs-uncached bit-identity, top-1/top-2/K=N + odd-T + 2-token window),
  `test_ckpt_moe` (v5 round-trip + `-44`/`-45`/`-10` rejections), `test_ckpt_v4_compat`,
  plus MoE alloc-accounting and config-cap pins. Fuzz extended (v5 base image; the
  `num_experts` field in the wild-field + boundary-combination modes).

### Notes
- The router/aux backward is the only new hand-derived math; everything else
  composes grad-checked pieces (`linear`, `gelu`, `softmax`). A vidya-corpus
  perplexity bake-off (MoE density vs dense vs the M12 variants) is the natural
  follow-on X-entry. New decision: **ADR 0008** (router combine + aux + frozen
  tie-break + the `experts==1`-is-dense invariant).

## [1.2.4] - 2026-06-12

**Toolchain realignment + docs.** A maintenance release — no feature change, the
no-flag run and every checkpoint byte-identical. The cyrius pin is realigned to
the installed compiler, and the docs are tidied for handoff now that M12 is
complete.

### Changed
- **Toolchain pin `6.1.37 → 6.2.1`** (`cyrius.cyml [package].cyrius`), with
  `cyrius update` resyncing the `./lib/` snapshot so the pin and snapshot move
  together (the standing rule). Re-verified green on the realigned toolchain:
  **572** checks on x86_64 AND aarch64/qemu, the `--agnos` static-ELF build, fuzz,
  and lint — `make release` exit 0, no shadow/drift warnings.

### Docs
- Roadmap trimmed to **forward-facing only** (M12 complete and removed from the
  milestone list; **M13 — Mixture of Experts** now leads). `state.md` carries a
  **handoff** section (build/test/release how-to, the invariants that matter, where
  things live) and a tightened loose-ends list. Flag/version/pin/count references
  swept current across `STABILITY.md`, `getting-started.md`, the ADR/architecture
  indices, and `sources.md`.

### Notes
- No source-behavior change: `src/*.cyr` is identical in effect to 1.2.3 (only the
  pin, `cyrius.lock`, `VERSION`/`CFG_VERSION`, and docs move). Next on the arc is
  **M13 — Mixture of Experts** (`--experts`, checkpoint v5).

## [1.2.3] - 2026-06-12

**Decoupled RoPE (M12 increment 5) — closes M12.** `--pos-kind rope-decoupled`
adds the faithful DeepSeek-V2 decoupled RoPE for MLA (arXiv:2405.04434): position
rides a **separate `d_rope` channel** that bypasses the latent compression, so the
latent stays absorbable and the cache stays small. The attention score splits
into a CONTENT term (over the compressed per-head K) plus a POSITION term (over
the rope channel), summed and scaled by `1/sqrt(hd + d_rope)`. Opt-in, additive —
a no-flag run is byte-identical; v4 value-fills the reserved `pos_kind = 2` +
`rope_dim` fields (no format bump). With this, M12's `--pos-kind` switch
(`learned` / `rope` / `rope-decoupled`) is complete.

### Added
- **`--pos-kind rope-decoupled` + `--rope-dim N`** (MLA-only; `d_rope` even,
  `2..d_model`, default ~`hd/2`). Two new bias-free projections per block:
  `W_QR` (per-head rope query, `C → nh·d_rope`) and `W_KR` (the **shared** rope
  key, `C → d_rope`, computed from `x` directly). Both rope parts are RoPE-rotated
  by absolute position; the shared `K^R` is one `d_rope` row per token.
- **Decoupled attention core** (`attn_dec_core_fwd`/`attn_dec_core_bwd`,
  `attn_mla_dec_fwd`/`attn_mla_dec_bwd`, `attn.cyr`): the two-term score with the
  shared rope key (`dKr` accumulates across heads). Reuses the `rope_apply_*`
  rotation primitives from 1.2.2; the novel hand-derived piece is the decoupled
  softmax/PV backward, grad-checked **bit-tight** in isolation
  (`test_attention_mla_dec`, dWqr/dWkr ~1e-6/1e-8) + full-model wiring
  (`test_model_mla_dec`).
- **Latent + rope KV-cache decode** (`attn_mla_dec_fwd_row`): the persistent cache
  holds the latent `c` (`d_c`) **and** the shared rope key `K^R` (`d_rope`) per
  token; each is rotated per absolute position on read. Cached-vs-uncached
  **bit-identity** gate across context-shifts (`test_kv_dec`, greedy + temperature,
  incl. a non-even content head dim and a degenerate 2-token window).
- **Layout / alloc / checkpoint** — `W_QR`/`W_KR` tile the block after the
  up-projections (FD-blind layout pin `test_param_layout_mla_dec`); alloc
  accounting covers the new caches (`test_alloc_accounting`); v4 `pos_kind = 2`
  round-trips with `rope_dim` (`test_ckpt_dec`) + hostile rejections (decoupled on
  non-MLA → `-41`, odd / out-of-range `d_rope` → `-43`). **470 → 572** checks green
  on x86_64 AND aarch64/qemu; fuzz + lint green.

### Changed
- The checkpoint loader accepts `pos_kind = 2` + a non-zero even `rope_dim` on an
  MLA image (bounded `2..C` before allocation). `ckpt_expected_np` and
  `model_alloc_bytes` gain the decoupled (`W_QR`/`W_KR` + the rope caches) terms.
- `kv_cache_bytes()` reports `NL·T·(d_c + d_rope)·8` for decoupled MLA — at
  `d_c = 16`, `d_rope = 4` that is **7680 bytes** (latent 6144 + rope 1536), still
  ~3.2× under MHA's 24576, now carrying relative position faithfully.

### Notes
- The decoupled reference materializes K/V from the latent each step (like 1.2.1);
  the absorption compute optimization remains future work.
- M12 is complete (MLA core + latent cache + coupled + decoupled RoPE). Next on
  the arc is **M13 — Mixture of Experts** (`--experts`, checkpoint v5).

## [1.2.2] - 2026-06-12

**Coupled RoPE (M12 increment 4) — the positional-encoding switch opens.**
`--pos-kind {learned, rope}` adds rotary positional embeddings (Su et al. 2021,
RoFormer, arXiv:2104.09864) on dense MHA/GQA: a position-dependent rotation of Q
and K so the score `(R_m q)·(R_n k)` depends only on the relative offset `m-n`.
Parameter-free, opt-in, additive — a no-flag run is byte-identical, no
checkpoint-format change (the reserved v4 `pos_kind` field is value-filled; ADR
0007). The faithful decoupled-RoPE form for MLA is the next rung (1.2.3).

### Added
- **`--pos-kind {learned, rope}`.** `learned` (default) keeps the learned
  absolute embeddings; `rope` rotates Q/K inside attention instead (the two are
  mutually exclusive — under RoPE the learned posemb is off the path and receives
  exactly zero gradient, pinned in `test_model_rope`). Coupled RoPE is MHA/GQA
  only with an EVEN head dim; MLA + RoPE (decoupled) is reserved for 1.2.3.
- **RoPE rotation** (`rope_apply_fwd`/`rope_apply_bwd` in `attn.cyr`): the
  original interleaved convention (pairs `(2k, 2k+1)` rotate by `m·θ_k`,
  `θ_k = 10000^(-2k/hd)`). The backward is the transpose rotation — the only new
  gradient (the rotation is parameter-free), grad-checked bit-exact in isolation
  (`test_rope_op`).
- **Portable trig.** `f64_sin`/`f64_cos` are x86-only builtins (no aarch64
  polyfill), so RoPE computes `cos θ_k`/`sin θ_k` (with `θ_k ∈ (0,1]`) by a
  Maclaurin series — no range reduction — then raises `(cos θ + i sin θ)` to the
  position power by binary exponentiation (pure `f64` + `f64_pow`/`f64_sqrt`,
  both arch-polyfilled). Computed directly from the absolute position, so the
  batch and cached single-row paths stay bit-identical
  (`docs/architecture/005-rope-portable-trig.md`).
- **Grad checks + bit-identity** — **376 → 470** checks green on x86_64 AND
  aarch64/qemu: the isolated rotation backward (bit-exact) + the relative-
  position invariance pin (`test_rope_op`); attention-with-RoPE at `hd ∈ {6,8,10}`
  × MHA/GQA/MQA, including the now-REAL K-bias gradient (a rotated bias is no
  longer shift-invariant — `test_attention_rope`); the full-model wiring check
  (`test_model_rope`); the cached-vs-uncached **bit-identity gate** across
  context-shifts (`test_kv_rope`); and the v4 `pos_kind=1` checkpoint round-trip +
  hostile-descriptor rejections (odd head dim, `mla+rope`, reserved `pos_kind=2`).

### Changed
- The checkpoint loader accepts `pos_kind = 1` (RoPE) on a dense, even-head-dim
  image; `pos_kind = 2` (rope-decoupled) stays reserved (`-41`), `rope_dim ≠ 0`
  stays reserved (`-43`). RoPE is parameter-free, so `np` and the parameter
  layout are unchanged — a v4 RoPE image differs from an mha image by one
  descriptor field only.

### Notes
- RoPE cached decode is ~10% slower per token and the training step ~2% slower
  at the default config (the per-pair rotation: a Maclaurin cos/sin + a
  binary-exponentiation to the position); benched in `docs/benchmarks.md`.
- Decoupled RoPE (`--pos-kind rope-decoupled`) for MLA — the faithful
  cache-efficient DeepSeek form, the last M12 rung — lands in 1.2.3.

## [1.2.1] - 2026-06-12

**MLA latent KV-cache decode (M12.2) — the deferred M12 gate.** 1.2.0 trained
and sampled MLA via the uncached reference forward; this adds the inference
compression win: a cached single-row decode path that stores ONE low-rank
latent `c` (`d_c` per token per layer) instead of full per-head K/V, and
up-projects to K/V on read. Additive and bit-identical — no checkpoint-format
change, the no-flag run untouched (ADR 0007).

### Added
- **Cached MLA decode** (`attn_mla_fwd_row`). The persistent cache holds the
  `d_c`-wide latent (the per-layer `LA_c` buffer); each step appends the new
  latent and re-up-projects the cached block `0..pos` to full K/V into transient
  scratch (the attention arena), then runs the shared single-row core. The K/V
  scratch is decode working set, re-derived from the latents — not persisted —
  so the stored cache stays `d_c`-wide. `--attn-kind mla` now generates through
  the KV cache like MHA/GQA.
- **Shared single-row core.** `attn_core_fwd_row` extracted from `attn_fwd_row`;
  MHA/GQA and MLA run the **identical** cached per-head softmax/PV kernel (one
  source of truth, `docs/architecture/003`), mirroring the batch-path extraction
  in 1.2.0. MHA/GQA cached decode stays bit-identical.
- **Bit-identity gate** (`test_kv_mla`). Cached-vs-uncached MLA verified two
  ways — prefill logits at every prefix `1..T`, and decode ids + final logits
  across context-shifts (greedy AND temperature) — at `hd ∈ {6, 8, 10}`,
  `d_c = C/2` / `d_c` not dividing `C`, odd `T`, and a degenerate 2-token
  window. **351 → 376** checks green on x86_64 AND aarch64/qemu.
- **KV-cache-bytes table** (bench): MLA latent vs the MHA/MQA full-K/V baselines
  at the default config. `d_c = 16` gives **6144 bytes — a 4× reduction** vs
  MHA's 24576, matching MQA's footprint **at full head expressiveness** (no
  head-sharing). Cached decode also benches ~4.6× the uncached MLA reference.

### Changed
- `kv_cache_bytes()` reports the latent footprint (`NL·T·d_c·8`) for MLA, the
  full-K/V footprint otherwise.

### Notes
- The reference re-up-projects the cached latents each step, so per-step compute
  is not yet at MHA-cached parity — the **absorption** optimization (fold `W_UK`
  into `W_Q` to attend latents directly) is the compute win and is future work;
  it changes accumulation order, so it would ride its own bit-identity story.
- Coupled/decoupled RoPE (`--pos-kind`, ADR 0007 increments 4–5) remain reserved
  in the v4 descriptor and land in a later M12 increment.

## [1.2.0] - 2026-06-12

**Multi-Head Latent Attention (M12) — the first new architecture on the 1.x arc.**
`--attn-kind mla` adds DeepSeek-V2-style MLA (arXiv:2405.04434): K and V are
factored through a low-rank latent (down-project `C → d_c`, up-project `d_c → C`)
instead of projected from `x` directly. Opt-in and additive — a no-flag run is a
byte-identical MHA transformer, and the frozen 1.0 surface is intact (ADR 0007).

### Added
- **`--attn-kind {mha, mla}` + `--latent-dim N`.** MLA uses full heads
  (`nkv = nh`; the compression is the latent, not head-sharing); `d_c` defaults to
  `d_model/2` (~4× the cached-KV footprint reduction it will enable). The latent
  down/up projections are plain linear layers, so the backward is matmul-backward
  with **no novel hand-derived math**.
- **Shared attention core.** `attn_core_fwd`/`attn_core_bwd` extracted from
  `attn_fwd`/`attn_bwd`; MHA/GQA and MLA run the **identical** softmax/PV kernel
  (one source of truth, `docs/architecture/003`). MHA/GQA stays bit-identical
  (grad-checks unchanged).
- **Grad checks.** Per-op MLA backward (`test_attention_mla`, tight 1e-4) +
  full-model MLA (`test_model_mla`) + the MLA parameter-layout pin
  (`test_param_layout_mla`, the FD-blind-aliasing guard) + alloc-accounting and
  config-cap pins. The latent down/up/output gradients land ~1e-8.
- **Checkpoint v4 fills the descriptor.** The reserved `attn_kind`/`latent_dim`
  fields (added as groundwork ahead of the math) now carry MLA; round-trips
  bit-for-bit (`test_ckpt_mla_roundtrip`), rebuilds with full heads, and the
  `-40`/`-42` gates enforce descriptor consistency (`d_c ∈ [1, C]`, full heads).
  **v1/v2/v3 still load.** Hostile-input fuzz extended to the v4 descriptor.

### Changed
- Checkpoint header is **v4**: saves record the architecture descriptor
  (`attn_kind`, `pos_kind`, `latent_dim`, `rope_dim`). A default-descriptor v4 is
  a byte-identical resume of a v3. New reject codes `-40..-43`.
- `248 → 351` grad-check/property tests green on x86_64; fuzz + lint green.

### Notes / deferred
- **MLA generation uses the uncached reference forward** (`model_eval_window`);
  the **latent KV-cache decode path** — the inference compression win, its
  bit-identity gate, and the KV-bytes table — is the **M12.2** follow-on. MLA
  *trains*, checkpoints, and samples here; the cache is additive on top.
- `--pos-kind` (coupled/decoupled RoPE) remains reserved in v4 (ADR 0007); the
  flag and math land in a later M12 increment.

### Docs
- **ADR 0007** (MLA + positional-encoding switch) and the **1.x architecture-arc
  roadmap** (M12 → M17, E7–E9 incl. MoE's 8→256 sweep and late-chain RL).
- DeepSeek-V2 (2405.04434) added to `docs/sources.md`.

## [1.1.0] - 2026-06-12

**The extraction — attn11 becomes the reference consumer.** The reusable numeric
core is lifted out of attn11 into two sovereign sibling libraries, which attn11
now consumes and dogfoods. No user-facing change: byte-identical training and
sampling, same CLI, same checkpoint format, all **248** grad-check/property
tests green. The frozen 1.0 surface is intact — this release is internal/
additive, not breaking.

### Changed
- **Tensor storage / BLAS-1 / dense matmul (+ its gradient) → [rosnet](https://github.com/MacCracken/rosnet) 0.1.0.**
  `t_alloc`/`t_zero`/`t_fill`/`t_copy`, `tget`/`tset`/`ff`,
  `t_axpy`/`t_add_into`/`t_scale`/`t_sum`, `f64_is_finite`, `t_randn`, and
  `linear_fwd`/`linear_bwd` now resolve from `[deps.rosnet]`. Matmul and its
  backward are pure linear algebra (`dx = dy·Wᵀ`, `dW = xᵀ·dy`), reusable beyond
  ML — so they belong in the tensor lib, not the model.
- **Deterministic statistical PRNG → [tyche](https://github.com/MacCracken/tyche) 0.1.0.**
  `rng_seed`/`rng_u64`/`rng_uniform`/`rng_normal` and the `_rng_state` stream now
  resolve from `[deps.tyche]`. attn11 still reads/writes `_rng_state` directly for
  crash-atomic checkpoint save/restore and bit-for-bit resume.
- `src/tensor.cyr` is reduced to attn11-local float printing (`f_print`); the
  model-specific differentiable layers (LayerNorm / GELU / dropout / softmax
  cross-entropy) stay in `src/ops.cyr` — only the general matmul moved out.
- Building from source now resolves the two sibling libs via `cyrius deps`
  (fetched + pinned in `cyrius.lock`).

### Unchanged
- **Still no BLAS / libc / autodiff.** rosnet and tyche are pure-Cyrius `f64`
  (IEEE-754 bit patterns in `i64`), sovereign-ecosystem libraries — not foreign
  numeric dependencies. The "assembly-up, everything-is-i64" property holds.
- User-facing surface, v3 checkpoint format, and runtime behavior are
  byte-identical to 1.0.0. attn11's own finite-difference grad-checks now double
  as rosnet's validation (linear / attention / full-model gradients); resume
  determinism + checkpoint roundtrip validate tyche's RNG state.
- Toolchain pinned at cyrius 6.1.37.

## [1.0.0] - 2026-06-11

**The clean cut — first non-prerelease.** attn11 is a complete, from-scratch,
dependency-free GPT-style transformer in Cyrius: forward pass, hand-derived
backprop (every op finite-difference grad-checked), Adam, and the full training
loop on raw `f64` arrays — no BLAS, no libc, no autodiff. The user-facing
surface is frozen ([`docs/STABILITY.md`](docs/STABILITY.md)); past 1.0 it is
additive-only. This tag adds **no features** over 0.9.0 — it is the final audit
plus release-hygiene fixes.

### What attn11 is at 1.0
- Byte-level (default) + opt-in BPE tokenizer; token + learned positional
  embeddings; `n_layers` stacked pre-norm blocks
  (LayerNorm → causal MHA/GQA → residual → LayerNorm → GELU MLP → residual);
  weight-tied LM head; softmax cross-entropy.
- **Hand-written backprop through every op** — 248 grad-check/property tests,
  green on x86_64 **and** aarch64 (qemu). Adam + global-norm clipping +
  warmup→cosine LR; NaN/inf training guard.
- 4-wide SIMD matmul / attention / LM head; **KV-cached generation**
  (bit-identical to the uncached reference) + GQA/MQA; the `--preset` scale
  config.
- Validated **v3 checkpoints** (v1/v2 load forever), crash-atomic save,
  deterministic bit-for-bit resume; `O_NOFOLLOW`/size-capped corpus + checkpoint
  loaders (structurally immune to the model-file-deserialization-RCE genre).
- Runs on Linux x86_64, **aarch64** (cross + qemu), and the **AGNOS kernel**
  (ring-3). Toolchain pinned at cyrius 6.1.37.

### Audit
- Final v1.0 audit — 5 adversarial dimensions (hostile-input, math correctness,
  memory safety, frozen surface, release integrity) — returned **go on all
  five, zero blockers** ([`docs/audit/2026-06-11-v1.0-final-audit.md`](docs/audit/2026-06-11-v1.0-final-audit.md)).
  Six prior audits (M2/M3/M5/M6/M7/M8 + the M10 freeze sweep) precede it.

### Fixed (release hygiene)
- `--save` failure now **exits non-zero** (previously printed an error but
  returned 0 — a script couldn't detect a failed checkpoint write).
- `scripts/version-bump.sh` now rewrites `src/main.cyr` `CFG_VERSION` (the
  `--version` flag) so a bump can't leave it stale — CI hard-gates the match.
- SECURITY.md threat model: present-tense the `--corpus`/`--stdin` wording
  (corpus loading shipped in 0.3.0, not "a future release").

## [0.9.0] - 2026-06-11

**Freeze, docs & cleanup (roadmap M10).** The run-up to the v1.0.0 clean cut:
the user-facing surface is now declared **frozen** (additive-only past v1.0),
the CLI is hardened, the docs audited, and the vidya example pipeline landed.
A no-flag run, the checkpoint format, and training behavior are unchanged.

### Added
- **`docs/STABILITY.md`** — the frozen-surface contract: the 12 CLI flags
  (additive-only), the compile-time `CFG_*` knobs and their frozen default
  values, checkpoint format v3 (v1/v2 load forever), the magnitude caps, and
  an explicit "not part of the contract" list.
- **`--help`/`-h` and `--version`** flags; the parser now **rejects unknown
  arguments** and **errors on a value-flag given without a value** (both exit
  non-zero with usage) instead of silently ignoring them — robustness expected
  of a frozen CLI. (`--version` is gated against `VERSION` in CI.)
- **The vidya example pipeline** (`docs/examples/vidya-pipeline.md`): preset +
  the 488 KB vidya corpus → train (loss 2.12 → **1.089** at 4000 steps,
  bits/byte **1.760**) → checkpoint (~5 MB, reloads bit-for-bit) → sample,
  with a BPE variant. The "curated small corpus" workflow against a tagged
  build (X001/X003).

### Changed
- **Toolchain pin `6.1.34` → `6.1.37`** (`cyrius update` resynced the `lib/`
  snapshot; pin and snapshot move together). 248 checks green on both arches +
  the agnos build; no drift warning.
- Docs audit (5-dimension multi-agent sweep): ADR 0005 generation figure
  corrected to the bench of record (6.2×); architecture note 001 now lists the
  M9-vectorized LM head among the SIMD paths; `sources.md` gains the missing
  dropout citation (Srivastava et al. 2014); `benchmarks.md` records the M9
  perf-lever outcomes (X004) instead of listing rejected levers as open work;
  ADR 0006↔0002 amendment link made bidirectional; getting-started/README gain
  the magnitude caps + a STABILITY pointer.

### Removed
- Dead code: `secure_write_file` (superseded by the crash-atomic writer),
  `f_println_lbl`, and the unused `CFG_NKV` accessor.

### Notes
- Cross-repo loose end (not in this tree): attn11 still needs a row in
  `agnos/scripts/stage-tools.sh` — `stage_one attn11 src/main.cyr attn11` —
  to stage `/bin/attn11` on the AGNOS rootfs. That is the agnos maintainer's
  edit.

## [0.8.1] - 2026-06-11

**Performance — SIMD tied LM head (roadmap M9, lever 1).** The first of the
one-lever-per-release perf cuts. `head_fwd_row` (the weight-tied output
projection) was a scalar dot product; the matmul has been 4-wide SIMD since
0.4.0. Vectorized it with the same `f64v_fmadd` accumulator + scalar tail.

### Changed
- **`head_fwd_row` is now 4-wide SIMD**: `head_fwd` (V=768, C=64, T=64)
  **~9.7 ms → ~3.59 ms (2.7×)**. The head is `O(V·C)` per row and runs in every
  training forward and every generated token, so the win scales with the
  vocabulary — negligible at the V=25 default (default-config training is
  unchanged within noise), ~17% of the forward at a BPE-scale V=768. The kernel
  is shared by the training and cached-generation paths, so the cached-vs-
  uncached bit-identity gate is unaffected. See
  [`docs/benchmarks.md`](docs/benchmarks.md) + the CSV.

### Added
- `head SIMD == scalar dot (C=6 tail)` test (248 checks): every other config
  uses `C ∈ {8,16}`, so the new `C % 4 ≠ 0` scalar tail was otherwise
  unexercised. Compares the vectorized head to an independent scalar dot at
  C=6; **mutation-verified** (a dropped tail fails it). Green on x86_64 and
  aarch64; grad checks + KV bit-identity + resume-determinism unchanged.

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
