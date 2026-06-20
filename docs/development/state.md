# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.8.3** — *M18 1.8.3: the LM head on the GPU — a transposed-weight matmul, TOLERANCE*
(E-infra; ADR 0016, X030). Adds **`head_fwd`** (`logits = f · tokembᵀ`) to **`--gpu-tc`**,
reusing 1.8.0's `GPU_TK`-tiled matmul with a **transposed-weight** kernel (`_gpu_build_tile_t`:
embedding contracted on its last dim → weight index `n·K+k`, advance +1). The reduction is
**sequential**, so vs the CPU head's 4-lane-partial + tree dot (`head_fwd_row`) it matches only
to ~1e-15 → TOLERANCE (the QKV/O/MLP matmuls are bit-exact because `linear_fwd` accumulates
sequentially; the head's SIMD-tree order is the difference). Rides `g_gpu_tc`, never plain
`--gpu`. **Validation (X030):** `tests/gpu_head.cyr` (`make gpu-test`) — allclose
(`atol=rtol=1e-10`, worst ~4.5e-13) across default (T16·V25·C32) / preset (T64·V256·C64) /
max-BPE (T64·V768·C64), engagement proven; statistical — a `--gpu-tc` 500-step run's loss +
bits/byte match CPU to print precision (`0.17790`; `0.29942`), no NaN. **Gate:** plain `--gpu`
byte-identical to no-flag (head not engaged, verified); **1056** grad-checks green x86_64 **and**
aarch64/qemu; AGNOS static-ELF clean (GPU path incl. head guarded out); lint clean; fuzz +
`make smoke` green; matmul + ln + gelu tests re-pass. `gpu_head_fwd` at the `head_fwd_n` seam;
new `tests/gpu_head.cyr`. **Remaining for the full forward (1.8.4):** softmax (fine-grained,
fused-attention — `_gpu_emit_exp` + max/sum reduction), then backward + Adam + the perf X-entry.
pin `mabda = 3.4.1`; cyrius pin stays **6.2.29** (installed cycc 6.2.31, benign drift).
`src/*.cyr` unchanged except the additive GPU wiring + CFG_VERSION.

(**1.8.2** — *M18 1.8.2: GELU on the GPU via an in-shader f64 `exp` — the first TOLERANCE
op* (E-infra; ADR 0016, X029). Crosses the transcendental wall (X028): f64 `exp` is an x86
hardware builtin with no bit-exact SPIR-V equivalent, so GELU is the first `--gpu` op that is
**not** bit-exact. It rides a **separate gate** — **`--gpu-tc`** (implies `--gpu`); **plain
`--gpu` stays matmul + layernorm only and remains byte-identical** to the no-flag run (the
strong invariant is preserved, opt out of it explicitly). **In-shader exp:** transliterated
from `math.cyr`'s aarch64 `_f64_exp_polyfill` — Cody-Waite reduction (magic-`1.5·2⁵²` round),
11-term Taylor Horner, `ldexp` (GLSL 53) — from proven f64 primitives only; **~2.3e-13** rel
vs CPU `f64_exp`. GELU composes it twice (`tanh=(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)`); `gpu_gelu_fwd` is
**~3e-14 abs** vs `gelu_fwd`, one elementwise kernel (id_bound ~110, no tiling). **Validation
(X029):** `tests/gpu_gelu.cyr` (`make gpu-test`) — **allclose** (`atol=rtol=1e-10`; pure
relative error is wrong near GELU's zero-crossings) on default/preset/decode widths, engagement
proven; plus the statistical gate — a `--gpu-tc` 500-step run's loss + eval bits/byte match CPU
to print precision (0.30858→0.17790; bits/byte 0.29942), never NaN. **Gate:** plain `--gpu`
byte-identical to no-flag (GELU not engaged, verified); **1056** grad-checks green x86_64 **and**
aarch64/qemu; AGNOS static-ELF builds clean (GPU path incl. GELU guarded out); lint clean; fuzz
+ `make smoke` green; matmul + ln tests re-pass. New `g_gpu_tc` + `--gpu-tc`; `gpu_gelu_fwd` at
the `gelu_fwd` seam; `tests/gpu_gelu.cyr`. **Still CPU:** softmax + `head_fwd`/QK (1.8.3), then
backward + Adam. pin `mabda = 3.4.1`; cyrius pin stays **6.2.29** (installed cycc 6.2.31, benign
drift). Invariants: [`../architecture/010-gpu-transcendentals.md`](../architecture/010-gpu-transcendentals.md).
`src/*.cyr` unchanged except the additive GPU wiring + CFG_VERSION.)

(**1.8.1** — *M18 1.8.1: layernorm forward on the GPU — bit-exact, the second `--gpu` op*
(E-infra; ADR 0016, X028). Extends `--gpu` from matmul (1.8.0) to **`ln_fwd`** (~2×/layer).
Same native-AMD f64 SPIR-V path (mabda 3.4.1); **bit-exact** vs the CPU `ln_fwd` oracle —
a `--gpu` run's checkpoint is **byte-identical** to the no-flag run with **both** matmul and
layernorm on-device (verified 30 steps: 17,514 matmuls + 6,811 layernorms dispatched,
identical checkpoint). **The transcendental wall (the gating finding):** `ln_fwd`'s
reductions are **sequential** and use only sqrt/div → bit-exact; but **softmax** and **GELU**
(=`tanh`=exp-based) need `f64_exp`, an **x86 hardware builtin** with no SPIR-V bit-exact
equivalent (and mabda's native path has no f64 transcendentals), and **`head_fwd`**/QK dots
use SIMD **tree** reductions (different order) — so they cannot be bit-exact and are a
separately-gated next increment (in-shader polynomial exp at a *tolerance*, which would break
byte-identity). 1.8.1 ships only the op that **keeps the invariant**. **Implementation:** the
256-id compile cap forbids a per-row unroll, so ln is **3-pass host-tiled** — (1) tiled Σx →
`S`, host `mean=S/C`; (2) tiled Σ(x−mu)² → `V`, host `rstd=1/√(V/C+eps)`; (3) elementwise
normalize — with `mean`/`rstd` written for the CPU backward. Three generated SPIR-V kernels
share a new `_gpu_pre` preamble; the GTT buffer set grew 3→7 via a clean alloc/teardown pair
(matmul re-validated, no regression). `gpu_ln_fwd` hooked at `ln_fwd` (`ops.cyr`), g_gpu-gated
+ `#ifndef CYRIUS_TARGET_AGNOS`; per-shape CPU fallback (no device / `C`∤16 / over limits).
Validated by `tests/gpu_ln.cyr` (`make gpu-test`, X028): bit-exact (y+mean+rstd) across
default 16×32 / preset 64×64 / decode 1×32 / wide 8×128, engagement proven, `C=24` fallback
checked; the 1.8.0 matmul test re-passes. **1056** grad-checks green x86_64 **and**
aarch64/qemu; AGNOS static-ELF builds clean (GPU guarded out + `SYS_IOCTL` stub); lint clean;
fuzz + `make smoke` green; `--gpu` run byte-identical to no-flag. Binary unchanged in
character (mabda already vendored at 1.8.0); pin `mabda = 3.4.1`. New `tests/gpu_ln.cyr`;
invariants in [`../architecture/009-gpu-layernorm-and-the-transcendental-wall.md`](../architecture/009-gpu-layernorm-and-the-transcendental-wall.md).
The cyrius pin stays **6.2.29** (installed cycc 6.2.30 warns of drift, builds+runs clean —
separate realign). `src/*.cyr` unchanged except the additive GPU wiring + CFG_VERSION.)

(**1.8.0** — *M18: the GPU compute backend — `--gpu` forward matmul on the native-AMD
f64 SPIR-V path, bit-exact vs the CPU oracle* (E-infra; ADR 0016, X027). The first
execution-target milestone. The hand-derived forward matmul now dispatches to the GPU
behind **`--gpu`**, riding **mabda 3.4.1**'s in-tree SPIR-V→GFX9 f64 emitter
(launcher-free, pure-Cyrius — bit-exact on the dev AMD Cezanne gfx90c). The CPU
scalar/SIMD path stays the f64 **oracle** and the byte-identical no-flag default; a
`--gpu` run **reproduces the no-flag run byte-for-byte** (the GPU matmul is bit-exact,
not f32-tolerant). **Sequencing inverted** (ADR 0016): the native-AMD f64 path shipped
*first* (portable wgpu doesn't run compute here + needs a C launcher + loses f64). The
kernel is **generated SPIR-V** (`src/gpu.cyr`) — `gfx9_compile` takes straight-line only
(no `OpLoopMerge`/`OpPhi`) and caps a module at 256 ids, so the dot product is **unrolled**
and the contraction **host-tiled** (`GPU_TK=16`-term RMW kernel, host pre-fills `y` with
bias, `ceil(K/16)` serialized dispatches; 1-D grid `M·N`, `idx=GlobalInvocationId.x`).
Hooked at `qlinear_fwd` (QKV / O / MLP — ~80% of FLOPs); per-shape self-fallback to CPU
(no device / `K`∤16 / over limits). Backward + the head + mixers stay CPU (1.8.2 / later).
Validated by `tests/gpu_matmul.cyr` (`make gpu-test`, X027): `gpu_matmul_fwd` vs
`linear_fwd` **bit-exact** across attn11's real shapes, engagement proven (non-zero return
= failure), `K=24` CPU-fallback checked — a **separate** device-dependent harness (skips
cleanly with no GPU, so the grad-check total stays environment-independent), **not** in the
CI release gate. End-to-end: a `--gpu` 40-step run's checkpoint is byte-identical to CPU's.
**1056** grad-checks unchanged green x86_64 **and** aarch64/qemu (mabda cross-compiles; no
device under qemu → clean CPU fallback); **AGNOS static-ELF builds clean** (main + suites; a
one-line `#ifdef CYRIUS_TARGET_AGNOS var SYS_IOCTL = 16` lets the auto-prepended Linux-only
mabda dep type-check as dead code there — cyrius has no target-conditional deps); lint clean;
fuzz + `make smoke` green; no-flag run byte-identical to pre-M18 (proven vs the HEAD binary). New files `src/gpu.cyr` (the backend)
+ `tests/gpu_matmul.cyr`; invariants in
[`../architecture/008-gpu-matmul-spirv.md`](../architecture/008-gpu-matmul-spirv.md).
**Binary SIZE (a cost, never a gate):** `qlinear_fwd` now references `src/gpu.cyr`, so every
binary that includes `ops.cyr` (main, test, fuzz, bench) carries mabda's ~1 MB dist —
binary **373,504 → ~1,376,680 B** — until cyrius ships `dep-module-call` (slims back with
zero attn11 change). mabda is **consumed, never modified**; pin **`mabda = 3.4.1`** (the
collision-fixed release). The cyrius toolchain pin stays **6.2.29**; the installed cycc
(6.2.30) warns of drift but builds + runs clean on both arches (the realign is separate
maintenance). `src/*.cyr` unchanged except the additive GPU wiring + CFG_VERSION.)

(**1.7.4** — *Toolchain realign (cyrius 6.2.27 → 6.2.29) + M18 GPU-arc unblock verified*
(maintenance + a milestone-status finding). The installed cycc rolled ahead of the pin (drift
at 6.2.27 vs 6.2.29); bumped `cyrius.cyml` and resynced the gitignored `lib/` + `cyrius.lock`.
The 6.2.27 → 6.2.29 stdlib snapshot differs in **only 5 files** — four off attn11's compile
path (`bayan`/`fdlopen`/`hashmap_fast`/`tls_native_conn`), the fifth an upstream **bugfix**
(`syscalls_aarch64_linux.cyr`: `SYS_FACCESSAT` 48 → 269, dodging an ESYSXLAT collision that made
native-aarch64 `sys_access()` return -EBADF; attn11 calls none). So the **x86_64 binary is
byte-identical** across the bump (proven — `1f8683b0…` ≡ `1f8683b0…`, 373,504-byte static ELF),
and the aarch64 binary differs only in that one uninvoked constant (default run byte-identical).
**1056** checks green x86_64 + aarch64/qemu; lint + fuzz + `make smoke` exit 0. `src/*.cyr`
unchanged except CFG_VERSION. **Milestone finding (X026):** an isolated-worktree integration
probe proved **M18 is unblocked**. mabda **3.4.1** fixes the one real gate — the `F64_HALF`/
`F64_TWO` symbol collision with `math` that statically zeroed those constants → NaN in ganita
tanh/GELU — by renaming its copies to `MABDA_F64_*`; `include "lib/mabda.cyr"` now builds clean
and the no-`--gpu` CPU path is byte-identical (finite loss, no NaN), and `gpu_probe`
(`src/gpu.cyr`) brings the native-AMD Cezanne f64 SPIR-V device online (X025). The second item,
cyrius `dep-module-call`, is **reclassified as a transient binary-size cost, not a gate** —
without it, including mabda amalgamates its ~1 MB dist (binary 373,504 → 1,338,344 B measured)
even when `--gpu` is unused; it is slated for the 6.2.x line and slims back with **zero attn11
change**. So M18 is **GO**; mabda stays staged in the default build only to keep it lean until
the landing approach is chosen — binary unchanged this cut.)

(**1.7.3** — *Toolchain realign + dependency bump* (maintenance). cyrius pin **6.2.6 →
6.2.27** (installed cycc had rolled well ahead; local builds warned of drift) plus the first
realign where the consumed AGNOS crates move too: **tyche 0.1.0 → 0.1.1** and **rosnet 0.1.0
→ 0.1.1**, each itself a pure toolchain-realign (6.2.11 pin + `dist/` regen, no API change),
so attn11's call surface is untouched. `cyrius update` resynced the gitignored `lib/`
snapshot + `cyrius.lock`; default training run **byte-identical** to 1.7.2 (baseline binary
built pre-bump, diffed post-bump). **1056** checks unchanged green; no pin-drift warning.
`src/*.cyr` unchanged except CFG_VERSION.)

(**1.7.2** — *Competitor benchmarks (B-series B0) + toolchain realign* (the wrap-up cut
before the GPU pause on mabda 3.x). **B0 harness** `scripts/compete-bench.sh` (+ `make
compete-bench`, `competitor-bench.csv`): attn11 vs llm.c/nanoGPT/llama2.c/micrograd on
training + decode, enforcing the fairness rules (matched config + **param-count assert**,
taskset/warmup, attn11 single-thread first, recorded upstream commit, gitignored `bench/`,
explicit skip-rows — no fabricated numbers). **B3 zero-deps story is real + complete**:
attn11 = **one 372,896-byte static ELF, no shared-lib deps** vs every competitor's runtime
stack. Validated by cloning + building llm.c (`f1e2ace`) + llama2.c (`350e04f`) + attn11's
real row (~4 393 tok/s, 39 488 params); **matched-config B1/B2 runs are harness-ready but
unpopulated** (need competitor stacks + GPT-2 data-prep on a bench machine; nanoGPT skipped
— no PyTorch here). B4 rides M18. **Toolchain**: cyrius pin **6.2.5 → 6.2.6** (`cyrius
update` resync); default/RL/ternary runs **byte-identical** to 1.7.1. **1056** checks
unchanged green x86_64 + aarch64/qemu; lint + fuzz + `make smoke` + `make release` exit 0.
`src/*.cyr` unchanged except CFG_VERSION. Write-up: [`../benchmarks.md`](../benchmarks.md).)

(**1.7.1** — *Toolchain realignment* (maintenance): cyrius pin **6.2.2 → 6.2.5**. The
installed cycc had moved ahead of the pin (local builds warned of toolchain drift); bumped
`cyrius.cyml` and ran `cyrius update` to resync the `lib/` snapshot + `cyrius.lock`. `lib/`
is gitignored (CI regenerates it from the pin), so the tracked diff is `cyrius.cyml` +
`cyrius.lock` only. The 6.2.5 snapshot differs from 6.2.2 **only in files attn11 doesn't
use on the Linux path** — networking/TLS (`net`, `tls_native*`), threading (`thread*`),
`chrono`, `syscalls_x86_64_agnos`; the core libs (`rosnet`/`tyche`/`alloc`/`fmt`/`io`/
`math`/`simd`/…) are **unchanged** — so the binary and all runs
(default/preset/BPE/diffusion/ternary/RL) are **byte-identical** to 1.7.0 (verified). Pin +
snapshot + lock moved together (the new-compiler/old-lib mismatch is the AGNOS
`argc()==0` trap, and `syscalls_x86_64_agnos` is among the changed files), so the agnos
target was rebuilt + the full gate re-run on the realigned snapshot. **1056** checks
unchanged, green x86_64 + aarch64/qemu; lint + fuzz + `make smoke` + `--agnos` build +
`make release` exit 0. `src/*.cyr` unchanged except CFG_VERSION.)

(**1.7.0** — *M17: Reinforcement learning (REINFORCE)* (E9; ADR 0015, X024; the last
milestone in the chain). `--objective rl` trains the policy toward a **reward** via
on-policy REINFORCE (Williams 1992): sample `batch` rollouts at temperature 1, score each
with a deterministic reward, weight its log-prob gradient by the advantage `(R − b)`
(`b` = EMA of past mean rewards). Because `∇log π(a) = −∇CE(a)` for a softmax policy, the
policy gradient **IS the existing softmax-CE backward over the sampled rollout scaled by
`(R − b)`** — no new forward, no new backward op, no autodiff, no checkpoint bump (the
model stays plain AR; an RL image is a normal v5). The scale is injected at `D_logits` in
`model_backward`, gated on `g_rl` (so AR/diffusion are untouched and the no-flag run is
**byte-identical**). Reward = count of a target char (`--rl-target C`, default space) per
rollout. New `g_rl`/`g_reinforce_scale` (model.cyr) + `rl_train`/`rl_rollout`/`rl_reward`/
`rl_prompt`/`rl_eval` (train.cyr) + `--objective rl`/`--rl-target` (main.cyr). Grad-checked
three ways (`test_rl_op`: RL grad == advantage × AR grad at maxrel 0, FD vs `advantage × CE`
at 1e-5, sign-flip + zero-advantage) + rollout structural (`test_rl_rollout`): **1045 →
1056** checks green x86_64 + aarch64/qemu. **Gate met (X024):** the policy moves decisively
toward the reward — target-char freq **9–19% (SFT) → ~99.7% (RL)** — with the honest SFT→RL
alignment tax (corpus bits/byte 0.24 → ~13; naive count reward = reward hacking; PPO/GRPO +
richer rewards are the documented follow-on). lint + fuzz + `make smoke` (valid `--objective
rl` + errors) green; `make release` exit 0. ADR 0015; runner `scripts/m17-rl.sh`.)

(**1.6.5** — *X021: the data-volume held-out win* (1.6.x group; experiment, **binary
unchanged**). The run X019 wanted + X020 unblocked: does more clean data generalize better
to a THIRD disjoint held-out set? It needs the **overfit** regime (X020 found <1.3%
own→held at sub-epoch — nothing to convert), so X021 forces it — a 256 KB train slice (many
epochs) vs a disjoint 4 MB slice at matched compute (4000 steps, BPE 256, seed 1337), both
scored on a third disjoint 512 KB slice of a curated 12 MB C4 pool. Rides the shipped
`--eval-corpus` + `c4_sample.py` (**no new binary surface**; `src/*.cyr` byte-identical to
1.6.4 bar CFG_VERSION; the runner is `scripts/x021-heldout.sh`). **Result (held-out
bits/byte):** at **preset** capacity, 4 MB beats 256 KB by **−14.2%** (2.383 vs 2.776) —
the data-volume generalization win; at **default** (tiny) capacity a **tie** (−0.06%), so
the win is a **capacity lever** (X017/X019 thesis, on held-out text). preset·256 KB
overfits hard (**+40.4%** own→held gap; own 1.977 looks *better* than the 4 MB model's 2.366
but it is memorization — on held-out the 4 MB model wins), so own-corpus bits/byte **inverts
the truth above the overfit threshold** (and is trustworthy below it — validating X020).
Closes the data story X016→X021. **1045** checks unchanged; `make release` exit 0. Full
write-up: [X021](experiments.md).)

(**1.6.4** — *BPE GB-scale shards* (1.6.x group; the second 1.6.2 fast-follow). The
in-RAM `--encode-shard` caps the corpus at 64 MB; **`--corpus FILE --bpe K --encode-shard
OUT --stream-encode`** now pre-encodes an arbitrarily large file to a BPE (u16) shard in
**bounded RAM** (~the 64 MB learn budget, independent of corpus size). Two passes:
(0) learn the byte vocab + K merges from a bounded prefix; (1) re-read the file in 1 MB
chunks, `tok_encode` each, emit the **stable** id prefix and carry the trailing
`2*BPE_MAX_TOKLEN` raw bytes; `ntokens` is patched into the header via `lseek` before the
atomic rename. **Chunk-boundary exactness**: a final BPE token spans ≤ `BPE_MAX_TOKLEN`
and greedy-LTR BPE is prefix-stable except within one token of the edge, so a token ending
≥ `BPE_MAX_TOKLEN` from the edge never changes as more bytes arrive — carrying the trailing
2 tokens (re-encoded from a true boundary) makes the chunked encode byte-identical to a
whole-corpus encode. Verified byte-identical to in-RAM `--bpe --encode-shard` across real
1 MB boundaries (2.6 MB BPE-128) and a 7-byte forced chunk (`test_stream_bpe_encode`,
+6). Byte-level `--stream-encode` works too (no carry). New `shard_stream_encode` in
`src/stream.cyr`; default run byte-identical to 1.6.3. **1039 → 1045** checks green
x86_64 + aarch64/qemu; lint + fuzz + `make smoke` (stream-encode round-trip + errors)
green; `make release` exit 0.)

(**1.6.3** — *Resume-from-stream* (1.6.x group; the first 1.6.2 fast-follow). A
checkpoint saved mid-stream reloads against the SAME token-shard to continue training —
**`--load` + `--stream-corpus`** (the prior rejection is removed). The shard supplies
the corpus, the checkpoint the weights/moments/step/RNG; resume is **bit-for-bit
deterministic** (`train(K)` streamed == `train(K1)` → checkpoint → `train(K)` streamed,
byte + BPE — streaming touches no RNG and the schedule horizon is held). Because a
streamed corpus holds its tokens on disk (`g_data == 0`, no re-encode), the loader
requires the shard to carry the **EXACT** tokenizer the checkpoint trained with — kind,
Vb, K, every base-vocab byte, every merge pair — not just the base vocab the in-memory
path checks (it adapts via re-encode). This closes a real hole: a BPE shard whose base
vocab matches a byte checkpoint would otherwise stream ids ≥ Vb past the byte model's
embedding table; both directions now reject with `-15`. In-memory resume is unchanged;
default run byte-identical to 1.6.2. `ckpt_load_buf` splits the vocab check into an
in-memory branch and a streamed (`g_stream != 0`) full-tokenizer-identity branch.
`test_resume_stream` (+7), dead-last: **1032 → 1039** checks green x86_64 + aarch64/qemu;
lint + fuzz + `make smoke` (resume-from-stream round-trip + vocab-mismatch) green;
`make release` exit 0.)

(**1.6.2** — *Streaming token-shard ingestion* (1.6.x group; the RAM-independent
large-corpus path). Two new flags decouple corpus size from RAM. **`--encode-shard
PATH`** pre-encodes the loaded corpus (byte or `--bpe`) to a self-describing
**token-shard** — header + tokenizer (byte vocab + BPE merges, as a checkpoint
carries) + packed token ids (`width` bytes/token, the on-disk `g_data`) — then exits.
**`--stream-corpus PATH`** trains/evals *from* a shard **without loading the tokens
into RAM**: `gd_ld` pulls windows by offset through one 64 K-token chunk cache
(`stream_tok`; file `lseek`+`read`, or a memory blob in the test), so RAM is
O(model + chunk) regardless of corpus size — empirically **flat ~13 MB** as the corpus
grows 1.9 → 8.1 MB (in-memory grows with it; at GB scale in-memory is impossible vs the
256 MB alloc cap). The design property: `gd_ld` is the *only* corpus-token reader, so
streaming is a read-through cache inside it and **`sample_window`/`eval_corpus`/the
model are untouched** — byte-identity is structural. Verified bit-for-bit: a streamed
run reproduces the in-memory run's params at byte + BPE width, across chunk boundaries
(449 K-token shard, 7× one chunk), on x86_64 **and** aarch64/qemu (the real
`open`/`lseek`/`read` path). `scripts/c4_sample.py --emit-shard` is the GB-scale
producer (streams C4, byte-level shard in bounded RAM, **byte-identical to `attn11
--encode-shard`** on the same text). Hostile shards rejected before the tokenizer is
installed (distinct `-60..-73` codes; the merge table validated as a well-founded DAG
with bounded expansion, mirroring the checkpoint loader) and validated against the real
file size, so a window seek never passes EOF; fuzz adds 500 shard rounds (487 rejected).
The **default run stays byte-identical** to 1.6.1 (default/preset/BPE/ternary/diffusion/
ssm-hybrid all verified). Streaming is Linux-only (`lseek`); agnos rejects `--stream-corpus`
(-73). Resume-from-stream + BPE GB shards are documented fast-follows. **1014 → 1032**
checks green x86_64 + aarch64/qemu; lint + fuzz + `make smoke` (new encode/stream + bad-shard
cases) green; `make release` exit 0. New file `src/stream.cyr`; invariants in
[`../architecture/007-streaming-token-shards.md`](../architecture/007-streaming-token-shards.md).)

(**1.6.1** — *i64-add ternary matmul + bench, M16 increment 2* (E6; ADR 0014, X023;
**closes the M16 gate**). The BitNet realization of `--ternary`: `x·W_eff = γ·(x·t)`,
`t ∈ {−1, 0, +1}`, collapses the contracted-dim multiply to **add / subtract / skip** with
one γ-scale per output (M·N multiplies vs the dense M·K·N). Two attn11-local reference
kernels `ternary_matmul_fwd`/`ternary_matmul_dx` (`ops.cyr`; `ternary_signs` fills the i64
signs + returns γ = absmean) implement it, grad-checked by `test_ternary_matmul` — the
forward and `dx` **pinned against the SIMD-f64 `W_eff` path at maxrel 0** (exact to
rounding), `dx` FD'd at 1e-5, `signs·γ == W_eff` value-exact (`dW` is the unchanged STE
pass-through pinned in `test_ternary_quant`): **1010 → 1014** checks, green x86_64 **and**
aarch64/qemu. Benched **head-to-head** vs the 1.6.0 SIMD-f64 path (X023,
[`experiments.md`](experiments.md)) at T×C×F = 16×32×128: the collapse is **~3.0× slower**
(matmul, 60.5 → 181.5 µs) / **~2.4× slower** (end-to-end with quant, 94.2 → 225.0 µs) —
`f64v_fmadd` retires 4 fused multiply-adds per instruction while the collapse is scalar
add/sub/skip, so the wide-SIMD f64 multiply is already cheaper per element than a scalar
add (the ~31% zero-skip on Gaussian `W` doesn't recover it). An **honest negative**: the
integer-add win needs **activation quantization** (int8 acts → a literal integer matmul,
the scoped-out heavier follow-on) and/or non-FMA hardware; the orthogonal **memory** win
(1.58 bits/weight) is real and unmeasured by this kernel. So the **default ternary forward
keeps the SIMD-f64 path**; the collapse ships as the grad-checked + benched reference
kernel, wired into no run — **ternary *and* default runs stay byte-identical to 1.6.0**
(default, preset, BPE verified). lint + fuzz + `make smoke` green; `make release` exit 0.)

(1.6.0 — *Ternary (BitNet-style) training, M16 increment 1* (E6; ADR 0014, X022). A
new **`--ternary`** flag trains with weights quantized to **{−1, 0, +1}**: each quantized
weight matrix becomes `W_eff = γ·clamp(round(W/γ),−1,+1)` (γ = absmean) in the forward,
while the **master weights stay f64** (fake-quant). The backward is a **straight-through
estimator** that needs no new gradient math — `dx` flows through the fixed `W_eff` (a
normal linear backward) and `dW = xᵀ·dy` is the pass-through (rosnet's `linear_bwd`
computes `dW` without reading `W`). Two attn11-local wrappers `qlinear_fwd`/`qlinear_bwd`
(in `ops.cyr`, since `lib/rosnet.cyr` is vendored) quantize into a pre-allocated scratch
`g_qscratch`; with `g_ternary == 0` they are exact passthroughs, so the **default run is
byte-identical** (default, preset, BPE all verified bit-for-bit). Scope (v1.6.0): **MHA +
dense MLP + uniform + AR + learned-abs** (mla/ssm/lin/MoE/rope/diffusion ternary are
documented fast-follows), gated at the CLI with a `model_init_full` backstop. **Checkpoint
v8** adds `g_ternary` at slot [23] (objective at [22]; a v8 image is AR); non-ternary AR
still writes v5, hybrid v6, diffusion v7 — byte-identical; `-48`/`-49` reject a hostile
v8 before allocation; the weight blob is unchanged (master f64). Grad-checked **where
defined** — `test_ternary_quant` FDs the `dx` path (1e-5) + the full-precision bias and
**pins the STE `dW` bit-exact** (a naive FD of `dW` through the piecewise-constant
quantizer is meaningless); `test_model_ternary` FDs the smooth params end-to-end;
`test_ckpt_ternary` round-trips v8: **986 → 1010** checks, green x86_64 **and**
aarch64/qemu; lint + fuzz + `make smoke` (new ternary cases) green. X022 (the accuracy
run, [`experiments.md`](experiments.md)): at reference scale ternary is **competitive
with f64** (default config, 2000 steps: f64 **0.254** bits/byte — bit-for-bit X015 — vs
ternary **0.228**; the ~1.58-bit constraint regularizes a memorizable tiny corpus) — "it
learns + is grad-checked", not a general win. The **i64-add ternary matmul + bench vs the
SIMD-f64 path** (the remaining M16 gate) was the increment-2 follow-on, shipped in 1.6.1
(X023); increment 1 reuses the f64 `linear_fwd` for correctness first. `make release` exit 0.)

(1.5.6 — *Held-out (cross-corpus) eval* (X020; the deferred X019 follow-on). A new
**`--eval-corpus PATH`** flag scores the model on a **disjoint** corpus: re-encode the
file through the **loaded tokenizer** (same byte vocab / BPE merges, no vocab-order
check) and run the active objective's eval — AR `eval_corpus` (CE/token + bits/byte) or
the diffusion denoising grid + ELBO bound. The generalization metric X019 flagged as
missing: bits/byte-on-own-corpus understates data's value in absolute terms; a held-out
pass measures it directly. The flag swaps the encoded held-out stream in for the eval
and restores `g_data`/`g_datalen`; **RNG-neutral** (eval sets `g_training = 0`);
combines with `--eval` (own then held-out both print); works under `--gen-only --load`
(score a saved checkpoint on unseen text). Unknown bytes → id 0; BPE replays merges (the
loader's idempotent re-encode). A failed held-out eval (missing/empty/too-short file)
sets a **non-zero exit** (mirrors `--save`); a hostile path **never crashes** (new
`make smoke` case). New `test_eval_held` pins the invariant — held-out of the OWN bytes
reproduces `eval_corpus()` **bit-for-bit** (byte + BPE) and rejects a too-short buffer:
**977 → 986** checks, green x86_64 **and** aarch64/qemu. The no-flag/default run is
**byte-identical** to 1.5.5 (verified). First cross-corpus generalization numbers (X020,
[`experiments.md`](experiments.md)): the own → held-out gap is **tiny (< 1.3%)** at both
scales (default 3.306 → 3.330; preset 2.738 → 2.773) — sub-epoch training barely
memorizes, so own-corpus bits/byte is a near-unbiased generalization proxy, retroactively
**validating the X016–X019 own-corpus numbers**; the data-volume "more data → held-out
win" comparison (train-4MB vs train-24MB, eval a third disjoint set) is now unblocked as
the X021 follow-on (lands at M16+ where a model can overfit a small corpus).
`make release` exit 0.)

(1.5.5 — *Hardening / audit / security pass* (P(-1); **closes the data-ingestion
& curation 1.5.x arc**). A security/correctness audit of the 1.5.x surface — the
packed `g_data` store + raised 64 MB cap (1.5.3), the `c4_sample.py` curation script
ingesting untrusted shards (1.5.1/2/4), and the char-diffusion path (1.5.0) — via an
adversarial multi-agent review: five read-only dimensions, each medium+ finding
adversarially verified (refute-by-default) against the committed code. **One**
confirmed finding, fixed; four dimensions clean. Finding: a **gzip-bomb / unbounded
line buffer** in `c4_sample.py` (`for line in gz` materializes a whole line → a
no-newline runaway member OOMs); fixed with a bounded `iter_lines()` (1 MB chunks,
8 MB per-line cap, drop+resync, `oversized=N` reported) — regression-verified normal
extraction unchanged, a 12 MB bomb completes (no OOM), and re-curation byte-identical
to before (X016/X017/X019 reproducibility preserved). The fix is **Python-side; the
Cyrius binary is byte-identical** (CFG_VERSION bump only) — **977** checks unchanged.
Audit at [`../audit/2026-06-13-1.5.x-hardening-audit.md`](../audit/2026-06-13-1.5.x-hardening-audit.md)
(the ninth; GO, 0 blockers). `make release` exit 0.)
(1.5.4 — *Curation at scale* (X019; the data-ingestion & curation 1.5.x arc,
step 3). The first run on 1.5.3's raised ceiling: curate a **24 MB / 12-shard** C4-en
corpus (`c4_sample.py --curate --shards 12`, 6× the old 4 MB cap, impossible before
the packed store) and run the scaled data+capacity experiment vs the 4 MB curated
baseline — two model sizes (default ≈53 K, preset ≈232 K) × two corpora, BPE 256,
matched compute. Eval bits/byte: default 3.232 (4 MB) / 3.405 (24 MB); preset 2.666 /
2.741. **Finding (X019)**: capacity is the dominant lever (default→preset −17.5% on
4 MB, −19.5% on 24 MB), and the diversity/volume penalty **halves with capacity** —
the tiny model pays +5.4% bits/byte on the bigger diverse corpus (samples more
garbled), the preset only +2.8% (fluent, richer vocabulary): the first attn11 evidence
that diversity/volume starts paying off with scale, validating sequencing larger
corpora + streaming with M16+. The 4 MB default cell reproduces X017's 3.232
bit-for-bit (curation + 1.5.3 packing both deterministic/transparent). **Data +
experiment only — binary unchanged** (CFG_VERSION bump; **977** checks unchanged).
Honest caveat: bits/byte-on-own-corpus understates data's generalization value; a
held-out **`--eval-corpus`** flag is the clean follow-on (deferred, additive). Recipe
+ grid in [`../examples/c4-english.md`](../examples/c4-english.md).
1.5.3 — *Token-packing unlock* (X018; the data-ingestion & curation 1.5.x arc,
step 2). The corpus token stream `g_data` was one **i64 per token (8 B)**; 1.5.3
stores it **packed** — `u8` for byte-level (vocab ≤ 256), `u16` for BPE (vocab ≤ 768)
— removing the 8×/4× bloat and raising `MAX_CORPUS_BYTES` **4 MB → 64 MB** (the u16
store is then 128 MB, half the 256 MB single-alloc cap). A `g_data_w` width global +
width-generic accessors (`gd_ld`/`gd_st` over `g_data`, `_tw_ld`/`_tw_st`(buf,i,w)
built on `load8/16`+`store8/16`); **only `g_data`** is packed (the T-sized
`A_tokens`/`A_targets`/`A_mask`/`G_win`/`g_enc_buf` stay i64). `main` sets the width
before the corpus load (byte-level → u8, `--bpe`/`--load` → u16); `bpe_learn`
self-widens a u8 corpus to u16 (BPE ids reach ≤ 766); `tok_encode` gained an
output-width arg (i64 prompt scratch vs the packed corpus rebuild). The model /
training math is untouched — same ids, same forward — so the **default run is
byte-identical** (verified: default/u8, BPE-64/u16, `--preset` all match 1.5.2
byte-for-byte). New `test_token_packing` (round-trip at u8/u16 incl. boundary ids
255/256/767, width-invariance, the self-widen) takes **966 → 977** checks. Invariants
in [`../architecture/006-packed-token-store.md`](../architecture/006-packed-token-store.md).
A larger corpus (6 MB byte + BPE) loads/trains; a 65 MB corpus rejects cleanly (−2).
1.5.2 — *Quality-curating C4 sampler* (X017; the data-ingestion & curation 1.5.x
arc, step 1). `scripts/c4_sample.py --curate` adds de-duplication (exact + prefix),
**multi-shard sampling** (`--shards N`, spread across the crawl), and prose/register
quality filters — all stdlib + deterministic, with a resilient multi-shard download
(per-shard retries / skip / rebalancing budget); defaults reproduce the 1.5.1 raw
slice byte-for-byte. A/B at iso-compute (default + BPE 256, 600 steps): the
**quality filter alone cut eval bits/byte 3.43 → 3.23 (−5.9%)** on the same shard,
while multi-shard *diversity* raised it (+2.7%) — a 53 K-param model can't exploit
diversity it can't fit, so **curate for quality now; diversity/volume is a scale
lever** (validates sequencing streaming with M16+). NO core binary change (only
`CFG_VERSION`); **966** checks unchanged.
1.5.1 — *C4 English experiment* (X016). Tooling + example for training on a
real **large external corpus** — a 4 MB slice of **C4** (`c4/en`, 305 GB), streamed
by `scripts/c4_sample.py` (stdlib `gzip`+`json`, no tensorflow/TFDS/pip — it streams
one public C4 shard and stops, ~1 MB downloaded) into a **gitignored** `data/` file
for `--corpus`. NO core binary change (only `CFG_VERSION`); the example
([`docs/examples/c4-english.md`](../examples/c4-english.md)) trains a 53 K-param
default+BPE model to **3.43 bits/byte** and samples recognizable broken English
(real words + structure between subword wobble) — fluency is a model-capacity story
(M16+), not a pipeline one. **966** checks unchanged.
1.5.0 — *Char-diffusion objective* (M15, E5; ADR 0013, X015). The first
*training-objective* departure from the AR trunk: `--objective diffusion` trains a
masked absorbing-state diffusion model (D3PM/MDLM) — drop the causal mask, corrupt a
window by replacing positions with a **learned `[MASK]` embedding** (mask_emb, +C
appended after lnf_b, gated on diffusion), and predict the originals with a
**bidirectional** model (`g_bidir`: the `j<=i` causal loop bound becomes the full
square); the loss is masked-CE over the masked positions only. Decode is MaskGIT-
style confidence-ordered parallel unmasking (uncached bidirectional — the KV cache is
AR-only). Two hand-derived backwards, each finite-difference grad-checked:
`test_masked_ce` (the masked-CE, ~0) and `test_attn_bidir` (the bidirectional core,
T≥3 + a future-attending pin); the full-model `test_model_diffusion` FDs both
mask_emb AND tokemb at mixed masked/given positions (~1e-5..1e-4). Scope: MHA (GQA
ok) + learned-abs + dense + uniform (lin/ssm are causal; mla/MoE/RoPE-diffusion are
fast-follows). **Checkpoint v7** adds one `objective` field at slot [22] + the +C
mask_emb (`-47` rejects a hostile v7; v≤6 synthesize AR); a uniform AR model still
writes v5 and a hybrid AR v6, **byte-identical**. X015 (matched 2000-step compute,
default config): AR memorizes the corpus (exact **0.254** bits/byte) while diffusion
only edges below uniform (ELBO bound **3.79**) — the "super data learner" advantage
is a scale phenomenon, absent at 39 K params / 190 bytes; honest negative result,
the gate is the grad-checks + a logged comparison, not a win. A no-flag run is
byte-identical. Verified: **966** checks x86_64 AND aarch64/qemu, agnos, fuzz, lint,
smoke.
1.4.6 — *Benchmarking pass*. A dedicated perf release: one canonical
`./build/bench` run across the whole mixer family on the current binary, with the
time series (`bench-history.csv`, stale at 0.9.0) and the perf doc
(`docs/benchmarks.md`, no MoE/linear/SSM/hybrid sections) brought current, and the
rung-d padded-layout cost pinned. Headline finding (X014): the **padding is
memory-only** — the mha/ssm 1/3 hybrid step (4.90 ms) equals the per-layer mix
(1·MHA + 2·SSM)/3 = 4.94 ms, so the zeroed pad costs params + Adam moments, not
FLOPs. linear ≈ MHA in step/decode (constant-T cache); SSM ~1.58× the step (the
O(T·C·N) scan); MoE ~1.95× at top-2 with 5.5× params. The default-config training
step is **flat 0.4.0 → 1.4.6** (~3.6 ms, ~4 450 tok/s) — the M12–M14 arc added five
opt-in axes with zero regression to the no-flag path. No behavior change
(measurement + docs + two bench param-prints). Verified: **907** checks x86_64 AND
aarch64/qemu, agnos, fuzz, lint, smoke.
1.4.5 — *Hardening pass* (P(-1), **closes the 1.4.x arc**). A security/correctness
audit of the 1.4.x surface (the per-layer hybrid dispatch, the rung-d padded
uniform-stride layout, checkpoint v6, the `--attn-every` CLI) via an adversarial
multi-agent review — five read-only dimensions, each finding adversarially verified
against the committed code + live repro builds. **One** confirmed finding, fixed: a
**CLI stack-buffer overflow** — `--attn-every K` filled the 128-slot `lkinds[1024]`
buffer one entry per layer, but `--layers` was unbounded before that loop (the
NL≤128 cap is in `model_init`, which runs after), so `--layers 100000 --attn-every 2`
wrote past the buffer (**SIGSEGV**, the M7 `--layers`-OOB class). Fixed with a
`cfg_nl` 1..128 guard before the loop; now a clean reject. New **`make smoke`** (in
the release gate) regression-guards hostile `--layers`/`--attn-every` combos (the
CLI arg path the grad-check/fuzz harnesses don't cover). The other four review
dimensions came back clean; audit at `docs/audit/2026-06-13-1.4.x-hardening-audit.md`,
and the checkpoint format comment now documents v6 + v1–v5 back-compat. NO behavior
change to training, the format, or any valid run. Verified: **907** checks x86_64
AND aarch64/qemu, agnos, fuzz, lint, smoke.
1.4.4 — *Any-mixer hybrids* (M14 rung d, E4, **completes M14**; ADR 0012).
Lifts 1.4.3's layout restriction so a hybrid interleaves ANY of the four mixers
`{mha, mla, lin, ssm}` — including full attention ⊕ the selective SSM (the survey's
strongest pairing, attn11's best single mixer). The trick: each block's K/V region
is **padded to the max `_kvw` over the kinds present** (`_kvw_hyb`), so the
per-block stride stays uniform — only `_kv_weight_size()` + the per-layer init/cache
gates change, NOT every `_o_*` offset (no per-layer-offset refactor). A smaller
kind tiles its weights and leaves a zeroed pad. `_hybrid_kinds_ok` keeps the
cross-cutting constraints (learned-abs, full heads, a valid shared latent iff any
mla/ssm). Checkpoint **v6 unchanged** (already carries the per-layer kinds, 1.4.3);
the loader sizes the padded block the same way (`ckpt_expected_np_kvw`). The mixed
SSM/MLA ⊕ MHA backward grad-checks (`test_model_hybrid_ssm`/`_mla`, ~1e-4);
mha/ssm + mha/mla cached decode bit-identical; padded mha/ssm v6 round-trip green.
Attention-fraction sweep (X013, base ssm): bits/byte within noise (0.218 pure-ssm →
0.279 pure-mha), the decode cache a continuous knob from constant `C·N` to ∝T K/V.
`{mha,gqa,lin}` hybrids stay exact (no pad). A no-flag run is byte-identical.
Verified: **907** checks x86_64 AND aarch64/qemu, agnos, fuzz, lint.
1.4.3 — *Per-layer mixer hybrid* (M14 rung c, E4, **the interleaving lever**;
ADR 0011). `--attn-every K` places a full-attention (MHA) block at every K-th layer
and a gated-linear block elsewhere — the survey's "a few attention layers among
many cheap recurrent ones" structural shift. The global `attn_kind` becomes a
per-layer `g_layer_kind`, read ONLY by the three `_attn_block_*` dispatch helpers
(`_lk(L)`); uniform models get the global back, byte-identical. Restricted to
**layout-compatible kinds {mha, gqa, lin}** (gated-linear reuses MHA's projections,
ADR 0009), so `_kvw`/`_blk`/the offsets/`g_NP` are identical and the per-block
stride stays uniform — no per-layer offset refactor, and the hybrid is
**parameter-free**. What it buys is a knob on the decode cache: `kv_cache_bytes`
SUMS the per-layer caches, so the attention fraction sets how much is T-growing K/V
vs constant lin state (1/3 attention ⇒ half of pure-MHA's). First checkpoint
**format bump (v6)**: a hybrid persists its per-layer pattern (loader rejects an
invariant-breaking kind, `-46`); uniform models still write v5, byte-identical. The
mixed backward grad-checks (`test_model_hybrid` ~1e-5); cached decode bit-identical
per interleaving (`test_kv_hybrid`); v6 round-trip green (`test_ckpt_hybrid`).
Attention-fraction sweep (X012, default config, 1200 steps): bits/byte within noise
across ratios (all beat pure-MHA 0.279), cache scales with the fraction — a
"trains + grad-checks", not a scaling claim. Verified: **857** checks x86_64 AND
aarch64/qemu, agnos, fuzz, lint.
1.4.2 — *Selective SSM* (M14 rung b, E4, **the third sequence mixer**; ADR
0010). `--attn-kind ssm` adds a minimal Mamba-lite diagonal SSM: a per-channel
N-state recurrence `h_t = exp(Δ·A)·h_{t-1} + Δ·B·a`, `y = Σ C·h + D·a`, with
Δ/B/C all functions of the input (the *selective* scan). The milestone is the
**hand-derived BPTT through the data-dependent scan** — `test_ssm_core` grad-checks
every parameter + the input at **~1e-7**. Reuses Wq (W_dt) + Wo (output proj) and
the `latent_dim` field (= state size N), so it rides `attn_kind = 3` — **checkpoint
v5, no format bump**. The decode cache is the constant `C·N` state (a third
constant-cache mixer). Mixer comparison (X011, default config, 1200 steps): SSM
**bits/byte 0.218** — best of the four (vs linear 0.239, MLA 0.273, MHA 0.279) — at
38 048 params, cache constant in T (8× under MHA at the preset). Full-model +
cached bit-identity green; lands in its own `attn_ssm.cyr`. A no-flag run is
byte-identical. Verified: **801** checks x86_64 AND aarch64/qemu, agnos, fuzz, lint.
1.4.1 — *Refactoring sweep* (maintenance; no behavior change — the no-flag
run byte-identical, **727** checks unchanged on both arches, every checkpoint
round-trips). Reorganizes the mixer machinery so the M14 rungs are cheap:
(1) the `attn_kind` dispatch is now ONE point each — `_attn_block_fwd`/`_bwd`/
`_fwd_row` in `model.cyr` (was inlined in four functions); (2) the per-block param
arithmetic is shared pure helpers `_kvw`/`_mlpw` used by both the offset helpers
and the checkpoint validator (the model↔persist keep-in-sync hazard is gone);
(3) the gated-linear mixer moved to its own `attn_linear.cyr` (one-file-per-mixer
pattern; `attn.cyr` 1266→976); (4) the six `_gen_bits_*` test helpers collapsed
to one driver. `src/*.cyr` identical in effect to 1.4.0.
1.4.0 — *Gated linear attention* (M14 rung a, E4, **opens the second
sequence-mixer family**; ADR 0009). `--attn-kind lin` swaps the softmax/PV core
for a causal RetNet-style **retention recurrence** `S_t = γ_h·S_{t-1} + k_t⊗v_t`,
`out_t = (1/√hd)·S_t^T q_t`, fixed per-head decay `γ_h = 1−2^{−(3+h)}`
(parameter-free). It reuses the MHA Q/K/V/O projections, so it rides the existing
`attn_kind` slot (value 2) — checkpoint **v5, no format bump**. The headline: the
decode cache is the **constant** `nh·hd²` retention state, not a T-growing K/V. The
hand-derived backward needs no state caching (`dq` via a forward S-recompute,
`dk`/`dv` via a reverse `dS`); pure multiply/add, so `test_lin_core` grad-checks at
**~1e-9**, full-model + cached bit-identity green. Mixer comparison (X010, default
config, 1200 steps): linear **bits/byte 0.239** (vs MHA 0.279, MLA 0.273) at MHA's
exact param count (39 488), with a 6 144 B cache that is **constant in T** (16×
under MHA at the preset). A no-flag run is byte-identical. Verified green: **727**
checks x86_64 AND aarch64/qemu, the `--agnos` static-ELF build, fuzz, lint.
1.3.0 — *Mixture of Experts* (M13, E8, **opens the FFN-density axis**; ADR
0008). The dense GELU MLP in each block becomes **N experts + a top-K router**:
`--experts N --expert-topk K` (N in 1..256, default topk 2; `--experts 1` = the
byte-identical dense baseline). The milestone is the **router backward** —
a discrete top-K pick (frozen, lower-index tie-break, bit-reproducible cross-arch)
→ a Mixtral-style renormalized top-K softmax combine (gradient only to the
selected logits, straight-through) + a Switch-style load-balance aux loss
(`α·N·Σ fᵢ·Pᵢ`, dispatch counts held constant) — both hand-derived and
finite-difference grad-checked (`test_moe_op` 1e-4, `test_moe_aux` 1e-5, full
model 1e-3; cached-vs-uncached **bit-identity** `test_kv_moe`). Checkpoint **v5**
records `num_experts`/`topk` (v1–v4 load, synthesizing the dense MLP). The
density sweep (X009, `scripts/moe-sweep.sh`): total params scale ~linearly with N
(39 K → 1.62 M at N=64) while per-token-active stays ~65–71 K, **routing entropy
0.993–0.999** (load stays balanced), bits/byte best at N=8–16. The cyrius pin also
moved **6.2.1 → 6.2.2** (clean patch realign, byte-identical `./lib/` snapshot).
Verified green: **673** checks x86_64 AND aarch64/qemu, the `--agnos` static-ELF
build, fuzz, lint — `make release` exit 0.
1.2.4 — *Toolchain realignment + docs* (maintenance): pin **6.1.37 → 6.2.1**,
`./lib/` resynced, **572** checks green on both arches + agnos + fuzz; roadmap
trimmed forward-facing, handoff section added. `src/*.cyr` identical to 1.2.3.
1.2.3 — *Decoupled RoPE* (M12 increment 5, **closes M12**; ADR 0007):
`--pos-kind rope-decoupled --rope-dim d_rope` — the faithful DeepSeek-V2 form for
MLA (arXiv:2405.04434). Position rides a **separate `d_rope` channel** that
bypasses the latent; the score = CONTENT (compressed per-head K) + POSITION (rope
channel), scaled `1/sqrt(hd+d_rope)`. Two bias-free projections — `W_QR`
(per-head) + the **shared** `W_KR` — both RoPE-rotated. The decoupled softmax/PV
backward (shared `K^R` → `dKr` accumulates across heads) grad-checked bit-tight in
isolation + full-model + cached-vs-uncached bit-identity (`test_kv_dec`). Decode
cache = latent + shared `K^R` (`NL·T·(d_c+d_rope)·8` = **7680 B** at d_c=16/
d_rope=4, ~3.2× under MHA). v4 value-fills `pos_kind=2`+`rope_dim`; no format bump.
M12's `--pos-kind` switch complete (learned / rope / rope-decoupled).
1.2.2 — *Coupled RoPE* (M12 increment 4): `--pos-kind rope` rotates Q/K by
absolute position on dense MHA/GQA (`rope_apply_fwd`/`rope_apply_bwd`,
interleaved pairs, RoFormer arXiv:2104.09864) so the score depends only on `m-n`.
Parameter-free (grad-checked bit-exact + relative-position pin + cached bit-
identity); posemb off-path → zero gradient. Portable trig (Maclaurin + complex
binary-exponentiation; `f64_sin`/`f64_cos` are x86-only —
[`../architecture/005-rope-portable-trig.md`](../architecture/005-rope-portable-trig.md)).
1.2.1 — *MLA latent KV-cache decode* (M12.2): the cached single-row MLA decode
path (`attn_mla_fwd_row`) stores ONE low-rank latent `c` (`d_c` per token,
per-layer `LA_c`) instead of full per-head K/V, up-projecting on read.
`attn_core_fwd_row` extracted so MHA/GQA and MLA share the cached single-row
kernel. `--attn-kind mla` generates through the KV cache, bit-identical to the
uncached reference; `kv_cache_bytes` reports the latent footprint — **6144 B at
d_c=16, 4× under MHA**, MQA's footprint at full heads.
1.2.0 — *Multi-Head Latent Attention* (M12, the first new architecture on the
1.x arc): `--attn-kind mla` factors K/V through a low-rank latent (down `C→d_c`,
up `d_c→C`, `--latent-dim`; full heads), the DeepSeek-V2 parameterization
(arXiv:2405.04434). A shared `attn_core_fwd`/`attn_core_bwd` was extracted so
MHA/GQA and MLA run the **identical** softmax/PV kernel; the MLA backward composes
from `linear_bwd` + the core (no novel hand-derived math). **Checkpoint v4**
records the architecture descriptor (`attn_kind`/`latent_dim`), round-trips
bit-for-bit, and v1/v2/v3 still load. 1.2.0 generated MLA via the uncached
reference path; M12.2 (above) added the latent KV-cache decode.)
(1.1.0 — *the extraction*: the reusable numeric core lifted to **rosnet** 0.1.0
(tensor/BLAS-1/matmul + gradient) and **tyche** 0.1.0 (deterministic PRNG),
resolved via `cyrius deps`; additive/internal, byte-identical, attn11 the
reference consumer. Still no BLAS/libc/autodiff (both libs pure-Cyrius
`f64`-in-`i64`).
1.0.0 — the clean cut, **first non-prerelease**: the **final audit** (5
adversarial dimensions: hostile-input, math, memory, frozen-surface, release —
**go on all five, 0 blockers**;
[`../audit/2026-06-11-v1.0-final-audit.md`](../audit/2026-06-11-v1.0-final-audit.md))
plus release-hygiene fixes (`--save` exits non-zero on failure; `version-bump.sh`
updates `CFG_VERSION`; SECURITY wording); no features over 0.9.0, the surface
declared frozen ([`STABILITY.md`](../STABILITY.md)), additive-only past 1.0.
0.9.0 — freeze, docs & cleanup (M10): the user-facing surface declared
**frozen** ([`STABILITY.md`](../STABILITY.md)); CLI hardened (`--help`/
`--version`, rejects unknown args + missing flag values); 5-dimension docs
audit; dead code removed (`secure_write_file`/`f_println_lbl`/`CFG_NKV`); the
**vidya example pipeline**
landed ([`examples/vidya-pipeline.md`](../examples/vidya-pipeline.md): preset +
488 KB corpus → loss 1.089, bits/byte 1.760 → 5 MB checkpoint → sample); and
the toolchain pin moved **6.1.34 → 6.1.37** (`lib/` resynced). A no-flag run,
the checkpoint format, and training behavior are unchanged. 0.8.1 —
performance, M9 lever 1: SIMD tied LM head, **2.7×** at V=768; the three other
M9 levers were measured and rejected (X004). 0.8.0 — security sweep (M8): a
survey→map hardening release;
checkpoint **format immunity** to the model-file-deser RCE genre confirmed; a
dropped `_file_size` arg crashing every **AGNOS `--load`** and checkpoint
**save broken on the aarch64 lane** (qemu `fsync` → `fdatasync`) both fixed;
`_atoi` saturation, merge-scratch pin, **CI supply-chain** hardening. See
[`../audit/2026-06-11-m8-security-sweep-audit.md`](../audit/2026-06-11-m8-security-sweep-audit.md).
0.7.1 — scale preset + BPE (M7, E3): `--preset` (ctx 64 / d_model 64; gen
**23×**), opt-in **BPE** (`--bpe K`, ADR 0006), checkpoint **v3**, `--eval`
bits-per-byte, pin 6.1.33 → 6.1.34; X003 byte-vs-BPE −11 to −13% bits/byte.
0.7.0 — inference efficiency (M6, E1+E2): KV-cached generation
(6.2×) + GQA, checkpoint v2, pin 6.1.31 → 6.1.33. 0.6.0 — AGNOS kernel port
(M5), bit-for-bit checkpoint vs Linux. 0.5.1 — standards conformance. 0.5.0 —
aarch64 validation, NaN/inf guard, soak, crash-atomic save. 0.4.0: 4-wide
SIMD matmul, ~2.27× faster. 0.3.0: corpus loading, checkpoints +
deterministic resume. 0.2.0: stacked layers, grad clipping, LR schedule.)

## Toolchain

- **Cyrius pin**: `6.2.29` (in `cyrius.cyml [package].cyrius`) — bumped from 6.2.27
  in 1.7.4 (6.2.6 → 6.2.27 in 1.7.3, 6.2.5 → 6.2.6 in 1.7.2, 6.2.2 → 6.2.5 in 1.7.1) to
  realign with the fast-rolling installed cycc (`cyrius update` resyncs the gitignored `lib/`
  snapshot + `cyrius.lock`; tracked diff = `cyrius.cyml` + `cyrius.lock`). 1.7.3 also
  moved the consumed AGNOS crates for the first time — **tyche/rosnet 0.1.0 → 0.1.1**
  (each a pure 6.2.11-pin realign + `dist/` regen, no API change). Most realigns' lib
  changes are in files attn11 doesn't use on its Linux path (networking/TLS, threading,
  `chrono`, `syscalls_x86_64_agnos`); the core libs (`rosnet`/`tyche`/`alloc`/`fmt`/`io`/
  `math`/`simd`/…) are functionally unchanged, so all runs (default/RL/ternary/etc.) stay
  byte-identical across the bump (verified 1.7.1 + 1.7.2 + 1.7.3 + 1.7.4). **1.7.4 is the
  first realign to touch an on-path file:** the 6.2.27 → 6.2.29 snapshot differs in only
  5 files, and one — `syscalls_aarch64_linux.cyr` — is compiled on the aarch64 target
  (`SYS_FACCESSAT` 48 → 269, an upstream ESYSXLAT-collision bugfix that had made native
  aarch64 `sys_access()` return -EBADF). attn11 invokes no `sys_access`, so x86_64 stays
  **binary-identical** (`1f8683b0…`) and the aarch64 binary differs only in that one
  uninvoked constant (default run byte-identical). 1056 checks green both arches, no drift
  warning. (1.3.0 moved the pin 6.2.1 → 6.2.2; 1.2.4 moved
  6.1.37 → 6.2.1.) The pin and snapshot must always move together: cycc
  6.1.32 fixed attn11's agnos argv-capture issue (r15-parked init rsp; the old
  `_agnos_init_rsp` global is gone) during M6, and a new-compiler/old-lib
  mismatch reproduces `argc()==0` under the kernel — the run gate caught it, so
  **every pin bump is followed by `cyrius update`** and a both-arches retest.
  (`docs/architecture/002` retired at ≥6.1.32.)

## Performance

4-wide SIMD (`f64v_fmadd`) matmul. x86_64:

- Training (default config): fwd+bwd step ~3.7 ms, **~4 300 tokens/sec**
  (b=16) — unchanged from 0.6.0 within noise. Preset (ctx 64 / d_model 64):
  fwd+bwd ~63 ms, ~1 000 tok/s — ~17× the default step for 5.2× the params
  and 4× the context.
- Generation, default config (0.7.0): uncached 1 050 579 ns/token →
  **KV-cached 170 392 ns/token (6.2×, 951 → 5 868 tok/s)**, greedy.
- Generation, **preset** (0.7.1): uncached 15 564 530 ns/token → **KV-cached
  672 747 ns/token (23×, 64 → 1 486 tok/s)** — the context-shift re-prime
  amortizes over T/2 = 32 tokens at ctx 64.
- KV cache bytes (default config): 24 576 at `nkv=4` → 12 288 (`nkv=2`) →
  6 144 (`nkv=1`).
- **Gated linear attention** (1.4.0, default config): fwd+bwd step **~3.8 ms**
  (~6% over the dense ~3.6 ms); cached gen **~160 µs/token** (the O(hd²) state
  update beats the O(T·hd) cache scan). Decode cache **6 144 B, constant in T**
  (16× under MHA at the preset); bits/byte 0.239 vs MHA 0.279 at the same params
  (X010).
- **Selective SSM** (1.4.2, default config, N=16): fwd+bwd step **~5.6 ms** (the
  O(T·C·N) scan, ~1.56× the dense ~3.6 ms); cached gen **~258 µs/token**. Decode
  cache **12 288 B, constant in T** (8× under MHA at the preset). Best bits/byte of
  the four mixers at reference scale — **0.218** (X011).
- **Per-layer hybrid** (1.4.3, default config, mha/lin): the attention-fraction
  sweep is **parameter-identical** (39 488 each) — the hybrid only redistributes the
  decode cache: 6 144 B (0/3 attn) → 12 288 (1/3) → 18 432 (2/3) → 24 576 (3/3,
  pure MHA). bits/byte within noise across ratios (0.234–0.244, all under MHA's
  0.279). Hybrid fwd+bwd step **~3.7 ms** (≈ the linear step; two of three blocks
  linear). X012.
- **Any-mixer hybrid** (1.4.4, default config, mha ⊕ ssm, base ssm): bits/byte
  0.218 (pure ssm) → 0.224 (1/3 mha) → 0.219 (2/3) → 0.279 (pure mha) — within noise,
  near pure ssm. The decode cache is a continuous knob: 12 288 B (pure ssm, constant
  `C·N`) → 16 384 (1/3) → 20 480 (2/3) → 24 576 (pure mha, ∝T). Padding lifts the
  hybrid params to MHA's 39 488 (vs pure ssm's 38 048, +1 440). Hybrid fwd+bwd step
  **~5.0 ms** (between pure ssm ~5.6 ms and the dense step). X013.
- **MoE** (default config, 8 experts, top-2): fwd+bwd step **~6.9 ms** vs the
  dense ~3.6 ms (top-2 = two active expert MLPs + the `C→N` router), 215 648 params
  vs dense 39 488; cached gen ~273 µs/token. Per-token compute scales with `topk`,
  parameter count with `N` (X009 density sweep).
- BPE merge training (`--bpe K`): one-shot ~110 ms for 256 KB at K=128.
- **Char-diffusion** (1.5.0, default config): fwd+bwd step **~3.98 ms** (~1.12× the
  causal MHA ~3.57 ms — bidirectional attention is the full T×T square vs causal's
  half-triangle); parallel decode **~0.80 ms/round** over the window (uncached
  bidirectional — diffusion denoises the whole window each round, no KV cache). +32
  params (the mask_emb) over dense's 39 488. Quality at this scale: X015 (AR's exact
  0.254 bits/byte vs diffusion's 3.79 ELBO bound).
- **i64-add ternary matmul** (1.6.1, X023, the M16 increment-2 gate): the BitNet
  collapse `x·W_eff = γ·(x·t)` (add/sub/skip + one γ-scale) benched head-to-head vs
  the SIMD-f64 `W_eff` path at T×C×F = 16×32×128 — **~3.0× slower** matmul-only (60.5 →
  181.5 µs) / **~2.4× slower** end-to-end with quant (94.2 → 225.0 µs). An honest
  negative: `f64v_fmadd` is 4-wide while the collapse is scalar, so the integer-add win
  needs activation quantization and/or non-FMA hardware (the memory win, 1.58 bits/weight,
  is orthogonal). The default ternary forward keeps the SIMD-f64 path; the collapse is the
  grad-checked + benched reference kernel only.
- **Canonical mixer/hybrid snapshot (1.4.6, X014):** one bench run cross-compares
  every mixer (step / cached-decode / cache bytes / params) — linear ≈ MHA, SSM
  ~1.58× the step, MoE ~1.95× at top-2; the **padded hybrid is memory-only** (its
  step is the per-layer mix, no pad compute); the default step is flat 0.4.0 →
  1.4.6. The full tables (incl. the attention-fraction cache curve) live in
  `docs/benchmarks.md`; the per-release bullets above are kept as the timeline.

See [`benchmarks.md`](../benchmarks.md) + [`../../bench-history.csv`](../../bench-history.csv).

## What works

End-to-end, on Linux x86_64, **aarch64** (cross-build + qemu; all checks pass
on both), and **the AGNOS kernel** (ring-3, booted in QEMU; bit-for-bit
checkpoint vs Linux at fixed CPU — `scripts/agnos-smoke.sh`):

- **Byte-level adaptive tokenizer** (default) + opt-in **simple BPE**
  (`--bpe K`, ≤512 merges; 0.7.1, ADR 0006): merges layer on the byte base
  vocab, frozen deterministic tie-break, pure i64 (bit-reproducible
  cross-arch), decode via a precomputed flat span table (no recursion)
- Token + learned positional embeddings
- **`n_layers` stacked** pre-norm Transformer blocks, each:
  `LayerNorm → causal multi-head self-attention → residual → LayerNorm → MLP (GELU) → residual`
- **Scale `--preset`** (0.7.1): ctx 64 / d_model 64 / 8 heads / 4 layers,
  with `--heads`/`--kv-heads`/`--layers` overrides for fresh models
  (magnitude-capped + alloc-pre-flighted in `model_init`, mirroring the
  checkpoint loader — file and CLI config gates share one invariant)
- **Grouped-query attention** (0.7.0): `n_kv_heads ≤ n_heads` shares K/V
  heads across query-head groups (`nkv = nh` = classic MHA, the default;
  `nkv = 1` = MQA); K/V projections are `C × Ckv`
- Final `LayerNorm` + weight-tied LM head → softmax cross-entropy
- Hand-written backprop through every op and the full residual stack
  (verified; incl. grouped dK/dV accumulation)
- **Adam** + **global-norm gradient clipping** + **LR warmup→cosine** schedule
- GPT-2 residual-projection init scaling (`1/sqrt(2·n_layers)`)
- Config-gated **attention biases** and **residual dropout** (dropout
  auto-disabled in eval/generation)
- Mini-batch grad accumulation; training logs loss / lr / grad-norm
- **NaN/inf training guard** (stops cleanly instead of poisoning weights)
- **KV-cached autoregressive generation** (0.7.0): per-layer K/V caches, one
  cached row per token, context-shift (drop oldest T/2 + re-prime) when the
  window fills; **bit-identical** to the uncached reference path; greedy +
  temperature sampling
- **Corpus from file/stdin** (`--corpus`/`--stdin`): `O_NOFOLLOW`, `fstat`
  size-cap (**64 MB**, 1.5.3, was 4 MB), byte-level adaptive vocab (raw bytes
  retained for BPE re-encode)
- **Packed corpus token store** (1.5.3, `g_data_w`): `g_data` holds one **u8**/token
  byte-level (vocab ≤ 256) or **u16**/token BPE (vocab ≤ 768) instead of an i64,
  removing the 8×/4× bloat; only the corpus buffer is packed (the T-sized batch
  buffers stay i64), the token ids and forward are unchanged (default run
  byte-identical). `bpe_learn` self-widens u8→u16; `tok_encode` carries an
  output-width arg. [`../architecture/006-packed-token-store.md`](../architecture/006-packed-token-store.md)
- **`--eval`** (0.7.1): one deterministic, RNG-neutral pass over the corpus →
  CE/token + **bits-per-byte** (tokenizer-comparable); runs after `--save`,
  so checkpoints are bit-identical with or without it
- **Checkpoints** (`--save`/`--load`): validated **v4** header — tokenizer
  triple + merge table validated as a well-founded DAG with bounded expansion +
  the architecture descriptor (`attn_kind`/`pos_kind`/`latent_dim`/`rope_dim`),
  all checked before allocation; **v1/v2/v3 still load** — + bit-for-bit
  **deterministic resume** (BPE re-encodes the retained corpus); **crash-atomic
  save** (temp + fsync + rename)
- **Multi-head latent attention** (1.2.0, `--attn-kind mla`, ADR 0007): K/V
  factored through a low-rank latent (down `C→d_c`, up `d_c→C`; `--latent-dim`,
  default `d_model/2`; full heads). Shares the extracted `attn_core_*` kernel with
  MHA/GQA; grad-checked per-op + full-model; checkpoint v4 carries the descriptor.
  Trains + checkpoints + samples.
- **MLA latent KV-cache decode** (1.2.1, M12.2): cached single-row MLA generation
  (`attn_mla_fwd_row`) stores the `d_c` latent per token (per-layer `LA_c`) and
  up-projects to K/V on read; `attn_core_fwd_row` shared with the MHA/GQA cached
  path. Bit-identical to the uncached reference. `kv_cache_bytes` reports the
  latent footprint — 6144 B at `d_c = 16`, **4× under MHA**, MQA's footprint at
  full heads. The absorption compute optimization is future work.
- **Coupled RoPE** (1.2.2, `--pos-kind rope`, ADR 0007): rotary positional
  embeddings on dense MHA/GQA (`rope_apply_fwd`/`rope_apply_bwd`) — Q/K rotated by
  absolute position so the score depends on `m-n` only (RoFormer). Parameter-free;
  mutually exclusive with learned-abs (posemb off-path → zero gradient). Portable
  trig (no x86-only `f64_sin`/`f64_cos`; `docs/architecture/005`). Even head dim,
  MHA/GQA only. Grad-checked + cached/uncached bit-identical; `np`/layout unchanged.
- **Decoupled RoPE** (1.2.3, `--pos-kind rope-decoupled --rope-dim d_rope`, ADR
  0007, **closes M12**): the faithful DeepSeek-V2 form for MLA — position on a
  separate `d_rope` channel (`W_QR` per-head query + `W_KR` shared key, bias-free,
  RoPE-rotated), score = content + position, scale `1/sqrt(hd+d_rope)`. The decode
  cache holds the latent + the shared `K^R` (`kv_cache_bytes = NL·T·(d_c+d_rope)·8`,
  7680 B at d_c=16/d_rope=4). New decoupled core (`attn_dec_core_*`,
  `attn_mla_dec_*`); grad-checked per-op + full-model + cached/uncached
  bit-identical; v4 carries `pos_kind=2`/`rope_dim`.
- **Mixture of Experts** (1.3.0, `--experts N --expert-topk K`, ADR 0008): the
  dense MLP becomes N experts (each `C→F→C`) + a bias-free router gate `C→N`;
  `--experts 1` is the byte-identical dense baseline. Forward: router logits →
  top-K (frozen lower-index tie-break) → renormalized top-K softmax → gate-weighted
  expert sum. Backward (`moe_fwd`/`moe_bwd`, `moe_aux_*` in `ops.cyr`): gradient to
  the selected logits only (straight-through pick) + the Switch load-balance aux
  loss (`α·N·Σ fᵢ·Pᵢ`, α=0.01, added to CE, off the eval path). Per-op +
  full-model grad-checked; cached decode bit-identical (MoE MLP is
  position-independent). Checkpoint v5 carries `num_experts`/`topk`. `--eval`
  reports total / per-token-active params + routing entropy. Trains + checkpoints
  + samples; the density sweep is X009 (`scripts/moe-sweep.sh`).
- **Gated linear attention** (1.4.0, `--attn-kind lin`, ADR 0009): a non-softmax
  sequence mixer — causal RetNet retention `S_t = γ_h·S_{t-1} + k_t⊗v_t`,
  `out_t = (1/√hd)·S_t^T q_t`, fixed per-head decay (parameter-free), over the MHA
  projections. New core `lin_core_fwd`/`bwd` + `attn_lin_fwd`/`bwd` (`attn.cyr`);
  hand-derived backward with no state caching (grad-checked ~1e-9). Cached decode
  (`lin_core_fwd_row`, per-layer `g_lin_state`) is the **constant** `nh·hd²` state
  — bit-identical to the batch scan. Rides `attn_kind=2` (checkpoint v5, no bump);
  full heads, learned-abs positions. Trains + checkpoints + samples (X010).
- **Selective SSM** (1.4.2, `--attn-kind ssm`, ADR 0010): a minimal Mamba-lite
  diagonal SSM — per-channel N-state `h_t = exp(Δ·A)·h_{t-1} + Δ·B·a`,
  `y = Σ C·h + D·a`, with Δ=softplus(a·W_dt)/B=a·W_B/C=a·W_C all input-dependent
  (selective). Core `ssm_fwd`/`bwd` + `ssm_fwd_row` (`attn_ssm.cyr`); hand-derived
  BPTT through the data-dependent scan (grad-checked ~1e-7). Reuses Wq (W_dt) + Wo
  (output proj) + `latent_dim` (= state N), so it rides `attn_kind=3` (checkpoint
  v5, no bump). Constant `C·N` decode cache (`g_ssm_state`); A inits to a negative
  ramp, D to 1. Best bits/byte of the four mixers at reference scale (X011).
- **Per-layer mixer hybrid** (1.4.3 rung c + 1.4.4 rung d, `--attn-every K`, ADR
  0011/0012): a full-attention (MHA) block every K-th layer, the `--attn-kind` base
  elsewhere. The global `attn_kind` becomes a per-layer `g_layer_kind` read only by
  the `_attn_block_*` dispatch helpers (`_lk(L)`). Rung c allowed {mha, gqa, lin}
  (shared `_kvw`, parameter-free); **rung d admits ANY mix of {mha, mla, lin, ssm}**
  by PADDING each block's K/V region to the max `_kvw` over the present kinds
  (`_kvw_hyb`) — uniform stride, no per-layer offset refactor, at the cost of a
  zeroed pad (and a few % params) on smaller-kind layers. `_hybrid_kinds_ok`: any
  kinds, learned-abs, full heads, a valid shared latent iff any mla/ssm.
  `kv_cache_bytes` sums the per-layer caches (mha K/V, lin/ssm constant state, mla
  latent). **Checkpoint v6** carries the per-layer pattern (`-46` on an invalid
  kind); the loader sizes the padded block via `ckpt_expected_np_kvw`; v≤5
  synthesize uniform. Mixed SSM/MLA ⊕ MHA full-model grad-check + cached-decode
  bit-identity green. Trains + checkpoints + samples (X012, X013).
- **Char-diffusion objective** (1.5.0, `--objective diffusion`, ADR 0013): a masked
  absorbing-state diffusion model on the MHA trunk — a learned `[MASK]` embedding
  (`mask_emb`, +C after lnf_b), **bidirectional** attention (`g_bidir`), masked-CE
  over the masked positions (`softmax_xent_masked_*`), confidence-ordered parallel
  decode (`gen_diffusion`). Deterministic mask sampling (`diffuse_mask`, per-example
  t~U(0,1), ≥1-mask floor); `eval_diffusion` reports denoising bits/byte at a mask
  grid + the ELBO bound. MHA + learned-abs + dense + uniform only. Checkpoint v7;
  grad-checked per-op + full-model (X015). A no-flag run is byte-identical.
- **Ternary (BitNet-style) training** (1.6.0, `--ternary`, ADR 0014, E6): weights
  quantized to **{−1, 0, +1}** in the forward — `W_eff = γ·clamp(round(W/γ),−1,+1)`,
  γ = absmean — with a **straight-through estimator** backward (`dx` through the fixed
  `W_eff`; `dW = xᵀ·dy` pass-through). Master weights stay f64. Two `ops.cyr` wrappers
  `qlinear_fwd`/`qlinear_bwd` (the vendored `lib/rosnet.cyr` is unmodified) quantize
  into `g_qscratch`; `g_ternary == 0` is an exact passthrough (default byte-identical).
  Quantizes the MHA Q/K/V/O + dense MLP only; scope MHA + dense + uniform + AR +
  learned-abs (gated at the CLI + a `model_init_full` backstop). Checkpoint v8 (slot
  [23]; `-48`/`-49` reject a hostile image). Grad-checked where defined + STE `dW` pinned
  bit-exact (`test_ternary_quant`/`test_model_ternary`/`test_ckpt_ternary`, X022). A
  no-flag run is byte-identical.
- **i64-add ternary matmul** (1.6.1, ADR 0014, E6, **closes the M16 gate**): the BitNet
  collapse `x·W_eff = γ·(x·t)`, `t∈{−1,0,+1}` — add/sub/skip + one γ-scale per output.
  Reference kernels `ternary_signs`/`ternary_matmul_fwd`/`ternary_matmul_dx` (`ops.cyr`),
  grad-checked (`test_ternary_matmul`: forward + `dx` pinned against the SIMD-f64 `W_eff`
  path at maxrel 0, `dx` FD'd) and benched head-to-head (X023). Honest result: **~2.4–3×
  slower** than the SIMD-f64 path on x86_64 (4-wide `f64v_fmadd` beats the scalar collapse;
  the integer-add win needs activation quant / non-FMA hardware). The default ternary
  forward keeps the SIMD-f64 path — the collapse is wired into no run, so ternary *and*
  default runs stay byte-identical to 1.6.0.
- **Streaming token-shard ingestion** (1.6.2, `--encode-shard` / `--stream-corpus`,
  1.6.x group): the RAM-independent large-corpus path. `--encode-shard PATH` pre-encodes
  the loaded corpus to a self-describing **token-shard** (header + tokenizer + packed ids,
  `src/stream.cyr`); `--stream-corpus PATH` trains/evals from one with RAM independent of
  corpus size — `gd_ld` reads windows by offset through one 64 K-token chunk cache
  (`stream_tok`), so `sample_window`/`eval_corpus`/the model are unchanged and a streamed
  run is **byte-identical** to in-memory (byte + BPE, across chunk boundaries, x86_64 +
  aarch64/qemu). `scripts/c4_sample.py --emit-shard` is the GB-scale byte-level producer
  (byte-identical to `--encode-shard`). Hostile shards rejected before install (`-60..-73`;
  merge-DAG + file-size validated). Linux-only; mutually exclusive with `--corpus`/`--stdin`/
  `--bpe`. `test_stream_ingest`, +18 checks; no-flag run byte-identical.
  [`../architecture/007-streaming-token-shards.md`](../architecture/007-streaming-token-shards.md)
- **Resume-from-stream** (1.6.3, `--load` + `--stream-corpus`): a checkpoint saved
  mid-stream reloads against the same shard to continue — **bit-for-bit deterministic**
  (byte + BPE). `ckpt_load_buf`'s vocab check has a streamed branch requiring the shard's
  FULL tokenizer (kind + Vb + K + base vocab + merges) to match the checkpoint's, so a
  mismatched shard rejects (`-15`) instead of streaming ids past the embedding table.
  `test_resume_stream`, +7 checks; in-memory resume + the default run unchanged.
- **BPE GB-scale shards** (1.6.4, `--stream-encode`): `--corpus FILE --bpe K
  --encode-shard OUT --stream-encode` pre-encodes an arbitrarily large file to a BPE
  (u16) shard in **bounded RAM** (`shard_stream_encode`): learn the tokenizer from a
  bounded prefix, then chunked `tok_encode` with a `2*BPE_MAX_TOKLEN` raw-byte carry
  (greedy-LTR BPE is prefix-stable except within one token of the edge → the chunked
  encode is **byte-identical** to a whole-corpus encode). `ntokens` patched via `lseek`
  before the atomic rename. Byte-level works too (no carry). `test_stream_bpe_encode`,
  +6 checks; default run unchanged.
- **Reinforcement learning** (1.7.0, `--objective rl`, M17/E9, ADR 0015): on-policy
  **REINFORCE** over the plain AR model — sample rollouts, reward = count of a target char
  (`--rl-target C`, default space), weight the log-prob gradient by the advantage `(R − b)`
  (`b` = EMA baseline). Since `∇log π(a) = −∇CE(a)`, the policy gradient is the existing
  softmax-CE backward scaled by `(R − b)` — injected at `D_logits` in `model_backward`
  (gated on `g_rl`, so AR/diffusion + the no-flag run stay byte-identical). No new
  forward/backward/checkpoint (an RL image is a normal v5 AR checkpoint). `rl_train`/
  `rl_rollout`/`rl_reward`/`rl_prompt`/`rl_eval` (train.cyr). Grad-checked 3 ways
  (`test_rl_op`) + rollout structural (`test_rl_rollout`), +11 checks. X024: policy
  target-char freq 9–19% → ~99.7% (gate met) with the honest SFT→RL alignment tax.
- CLI: `--corpus --stdin --load --save --steps --gen-only --preset --heads
  --kv-heads --layers --attn-kind --latent-dim --attn-every --pos-kind --rope-dim
  --experts --expert-topk --bpe --eval --eval-corpus --encode-shard --stream-encode
  --stream-corpus --objective --decode-steps --decode-schedule --ternary --rl-target`
  (`--attn-kind` takes `mha`/`mla`/`lin`/`ssm`; `--latent-dim` is the MLA latent /
  SSM state size; `--attn-every K` builds the per-layer hybrid over the `--attn-kind`
  base; `--objective` takes `ar`/`diffusion`/`rl` — `diffusion` adds `--decode-steps` /
  `--decode-schedule {cosine,linear}`, `rl` adds `--rl-target C` (REINFORCE reward char);
  `--ternary` is M16 BitNet-style ternary weights on mha + dense + AR; `--encode-shard`
  writes a token-shard + exits, `--stream-encode` does so for a GB-scale file in bounded
  RAM, `--stream-corpus` trains from one in bounded RAM; `--load` + `--stream-corpus`
  resumes from a stream)

Default run (`./build/attn11`, 3 layers): loss `~3.2 → ~0.13` over 2000 steps;
sampled output reproduces real corpus phrases.

## Default hyperparameters (`src/main.cyr`)

| name        | value | note                              |
|-------------|-------|-----------------------------------|
| vocab `V`   | 25    | unique chars in the corpus        |
| `d_model` C | 32    |                                   |
| context `T` | 16    |                                   |
| heads `nh`  | 4     | head dim = C/nh = 8               |
| kv heads    | 4     | = nh (full MHA; `nkv < nh` = GQA) |
| layers `NL` | 3     | stacked pre-norm blocks           |
| MLP `F`     | 128   | = 4·C                             |
| attn bias   | on    | Q/K/V/O biases (config-gated)     |
| dropout     | 0.0   | residual dropout (config-gated)   |
| params      | 39488 | total trainable f64 (3 layers, biases) |
| optimizer   | Adam  | β 0.9/0.999, global-norm clip 1.0 |
| lr schedule | warmup 100 → cosine | base 3e-3 → min 3e-4  |
| steps/batch | 2000 / 16 |                               |

`--preset` overrides to C 64 / T 64 / nh 8 / NL 4 (205 760 params at the
embedded corpus); `--heads`/`--kv-heads`/`--layers` override individual dims
(magnitude-capped: nh|C, nkv|nh, NL ≤ 128, C ≤ 4096, T ≤ 8192). `--bpe K`
raises V to `base + K` (≤ 768). `--attn-kind mla --latent-dim d_c` (1 ≤ d_c ≤ C,
default C/2) swaps the K/V projections for the low-rank latent (37 952 params at
d_c=16 vs MHA's 39 488). `--pos-kind rope` (1.2.2; MHA/GQA, even head dim) swaps
learned-abs positions for coupled RoPE — parameter-free, so `params` is unchanged.
`--experts N --expert-topk K` (1.3.0; N ≤ 256, default topk 2) replaces the dense
MLP with N experts + a `C→N` gate: total params scale with N (215 648 at N=8 vs
39 488 dense) while per-token-active compute scales with topk; `--experts 1` is
the dense default (params unchanged). `--attn-kind lin` (1.4.0; full heads,
learned-abs) swaps the softmax core for the gated retention recurrence —
parameter-free (same 39 488 params as MHA), and the decode cache becomes the
constant `nh·hd²` state instead of a T-growing K/V.

## Source (`src/`, ~1500 LOC)

- `tensor.cyr` — attn11-local float printing (`f_print`) + `_putc`/`puts` (40
  lines); the f64-array helpers + dense matmul moved to **rosnet**, the PRNG to
  **tyche** (1.1.0 extraction)
- `ops.cyr` — layernorm, GELU (tanh approx), softmax cross-entropy (forward +
  backward); `linear_fwd`/`linear_bwd` now resolve from **rosnet**; the
  **MoE** router/combine (1.3.0) `moe_fwd`/`moe_bwd` (top-K pick + renormalized
  softmax + per-expert MLP, gradient to selected logits only) and the Switch
  load-balance aux (`moe_aux_fwd`/`moe_aux_dr`/`moe_aux_bwd`); plus the **M16
  ternary** quantizer (1.6.0) `ternary_quant` (absmean-scaled `{−1,0,+1}`) and the
  quantizing wrappers `qlinear_fwd`/`qlinear_bwd` over rosnet's linear (passthrough
  when `g_ternary == 0`; STE backward — `dx` through `W_eff`, `dW` pass-through); plus
  the **M16 increment-2** i64-add reference kernels (1.6.1) `ternary_signs` (i64 signs +
  γ) / `ternary_matmul_fwd` / `ternary_matmul_dx` (the BitNet collapse `γ·(x·t)`,
  add/sub/skip — grad-checked + benched vs the SIMD-f64 path, X023; wired into no run)
- `attn.cyr` — the shared attention core `attn_core_fwd`/`attn_core_bwd` (causal
  scaled-dot-product softmax/PV), wrapped by `attn_fwd`/`attn_bwd` (MHA/GQA
  projections) and `attn_mla_fwd`/`attn_mla_bwd` (MLA low-rank latent down/up
  projections; 1.2.0); one pre-allocated arena; the shared cached single-row core
  `attn_core_fwd_row`, wrapped by `attn_fwd_row` (MHA/GQA KV cache) and
  `attn_mla_fwd_row` (1.2.1: latent KV-cache decode — store the `d_c` latent,
  up-project to K/V on read) — each bit-identical per row to its batch path; plus
  coupled **RoPE** (1.2.2) `rope_apply_fwd`/`rope_apply_bwd` (gated by `pos_kind`
  in `attn_fwd`/`attn_bwd`/`attn_fwd_row`) with portable trig
  (`_rope_unit_cossin`/`_rope_pow` — Maclaurin + complex binary-exponentiation,
  no x86-only `f64_sin`/`f64_cos`; `docs/architecture/005`); plus the **decoupled
  RoPE** core (1.2.3) `attn_dec_core_fwd`/`bwd` + `attn_mla_dec_fwd`/`bwd` +
  `attn_mla_dec_fwd_row` (two-term content+position score, shared `K^R`, the
  latent+rope decode cache); `attn_arena_size` carries a `2·hd²` S/dS scratch for
  the gated-linear core. The pure `_kvw` per-block K/V-region size (1.4.1) is
  shared with persist's checkpoint validator (one layout source).
- `attn_linear.cyr` — the **gated-linear** mixer (1.4.0; split out of `attn.cyr`
  in 1.4.1, the one-file-per-mixer pattern): `lin_core_fwd`/`bwd` (retention
  recurrence `S_t = γ_h S_{t-1} + k_t⊗v_t`, fixed per-head decay) + `attn_lin_fwd`/
  `bwd` wrappers + `lin_core_fwd_row`/`attn_lin_fwd_row` (the constant-state cached
  decode). Included after `attn.cyr` in each entry.
- `attn_ssm.cyr` — the **selective SSM** mixer (1.4.2, ADR 0010; one-file-per-mixer):
  `ssm_fwd`/`ssm_bwd` (the data-dependent scan + the hand-derived BPTT through
  `exp(Δ·A)`) + `ssm_fwd_row` (constant `C·N`-state cached decode) + `_ssm_softplus`.
  Reuses Wq (W_dt) + Wo (output proj); A/W_B/W_C/D are the `attn_kind=3` K/V region.
- `fileio.cyr` — secure file I/O (`O_NOFOLLOW`, `fstat` size, looped read/write),
  stdin reader
- `model.cyr` — per-layer packed parameters (block stride + `_o_*`/`PL`/`GL`
  helpers; one `_kv_weight_size()` branches the K/V region MHA↔MLA, ADR 0007),
  per-layer activation caches (+ the MLA latent `LA_c`, doubling as the 1.2.1
  latent decode cache), embeddings, tied head, full N-layer forward/backward
  (`attn_kind`-branched), grad clipping, Adam; KV caches + `model_fwd_row`
  (cached row, `attn_kind`-branched: full-K/V for MHA/GQA, latent for MLA) +
  `model_eval_window` (uncached eval reference); `kv_cache_bytes` reports the
  latent footprint for MLA; `model_init_arch`/`model_config_ok_arch`/
  `model_alloc_bytes_arch` carry the descriptor (the `_arch` forms; the old
  names delegate as MHA). `model_config_ok_arch`/`model_init_arch` also carry
  `pos_kind` (1.2.2: gate rope→even-hd + MHA/GQA); `embed_fwd_n`/`embed_bwd`/
  `model_fwd_row` skip the learned posemb add/grad under RoPE. **MoE** (1.3.0):
  `_mlp_weight_size()` branches the MLP region dense↔experts (ADR 0008, the second
  config-dependent region after `_kv_weight_size`), `_o_expert(e)`/`_o_Wgate()`
  index the experts + gate; the `_moe` forms (`model_init_moe`/`model_config_ok_moe`/
  `model_alloc_bytes_moe`) carry `num_experts`/`topk` (the `_arch` forms delegate as
  dense); the block MLP fwd/bwd + `model_fwd_row` + `model_eval_window` branch on
  `g_num_experts > 1`; `moe_entropy`/`moe_disp_*` report routing-entropy utilization.
  **Gated linear** (1.4.0): the per-layer retention state `g_lin_state` is the
  constant decode cache, and `kv_cache_bytes` reports it. **Mixer dispatch (1.4.1):**
  the mixer-kind branch lives in ONE place each — `_attn_block_fwd`/`_attn_block_bwd`/
  `_attn_block_fwd_row` — so a new mixer (SSM) touches those three, not four functions.
  **Per-layer hybrid (1.4.3 rung c + 1.4.4 rung d, ADR 0011/0012):** those three
  helpers read `_lk(L)` (the per-layer `g_layer_kind`, else the global) — uniform
  models byte-identical; `model_init_full` carries the per-layer array
  (`model_init_moe` is the uniform delegator). Rung d's `_kvw_hyb` pads each block's
  K/V region to the max `_kvw` over the present kinds (so the stride stays uniform
  for any mix), the init loop + caches gate per-layer on `_lk(L)`/`_any_kind(k)`,
  `_hybrid_kinds_ok` validates the relaxed constraints, and `model_alloc_bytes_hyb`
  takes the per-layer `kinds` pointer to count each present kind's caches
- `train.cyr` — byte + **BPE** tokenizer (`bpe_learn`/`tok_encode`/
  `bpe_build_spans`/`tok_emit`), corpus (embedded/file/stdin, raw bytes
  retained), batch sampling, LR schedule, resumable training loop, KV-cached
  generation (`gen_prime`/`gen_decode` + context-shift), `eval_corpus`
  (CE/token + bits-per-byte — pure CE, excludes the MoE aux term; accumulates the
  routing-entropy dispatch histogram)
- `persist.cyr` — validated checkpoint serialize/load (tokenizer triple +
  merge-table DAG/expansion validation + the v4 architecture descriptor, codes
  `-40..-43`, + the v5 **MoE** descriptor `num_experts`/`topk`, codes `-44`/`-45`;
  `ckpt_expected_np_moe` mirrors the MLA/decoupled/MoE layout; `pos_kind=1`
  (coupled RoPE) accepted on even-hd MHA images, `pos_kind=2`+`rope_dim` (decoupled)
  accepted on MLA images with even `2≤d_rope≤C`, `num_experts∈1..256`/`topk∈1..N`
  — all bounded before alloc; v1/v2/v3/v4 accepted, synthesizing dense MLP). **v6**
  (1.4.3): a hybrid appends NL per-layer kinds after the fixed header (`_ckpt_pl`),
  `_hybrid_kinds_ok`-validated on load (code `-46`); `CKPT_VER()` writes v6 only for
  a hybrid (uniform → v5, byte-identical); v≤5 synthesize the uniform pattern
- `main.cyr` — CLI arg parsing (incl. `--preset`/`--heads`/`--kv-heads`/
  `--layers`/`--attn-kind`/`--latent-dim`/`--attn-every`/`--pos-kind`/`--rope-dim`/
  `--experts`/`--expert-topk`/`--bpe`/`--eval`, null-guarded `_atoi`; `--attn-every K`
  builds the per-layer hybrid and dispatches to `model_init_full`) + orchestration
- `test.cyr` — the `[build].test` entry (`cyrius.cyml`); delegates the unit suite to
  `tests/attn11.tcyr`

## Tests

- `tests/attn11.tcyr` — **1014 checks**: finite-difference gradient checks
  (every op incl. dropout; attention at head dims 6/8/10 and GQA/MQA at
  `nkv ∈ {1, 2, nh}` incl. `dWk`/`dWv`/`dbv`; the `|dbk| ≈ 0`
  softmax-shift-invariance pin; 2-layer full model at MHA and GQA), the **MLA
  suite** (1.2.0: per-op `attn_mla_fwd`/`bwd` grad-check at 3 latent configs
  incl. the `|dbuk| ≈ 0` shift-invariance pin, full-model MLA grad-check, the
  **MLA parameter-layout/alloc/config pins**, and the v4 MLA checkpoint
  round-trip + `-42` descriptor-consistency rejections), the SIMD
  bit-contract, the **SIMD-LM-head tail pin** (`C % 4 ≠ 0` at C=6 vs a scalar
  dot — mutation-verified; no other config exercises it), the **parameter-layout tiling pin** (FD is blind to offset
  aliasing, MHA + MLA), the **alloc-accounting pin** (`model_alloc_bytes` ==
  `model_init`, incl. V=300 + MLA), the **config-magnitude-cap pin**
  (`model_config_ok` rejects out-of-range V/C/T/NL + the MLA `d_c`/`nkv` gates —
  the `--layers` heap-OOB regression), resume-determinism (dropout off/on + MQA + BPE), the
  **file-path round-trip** (`ckpt_save_file`/`ckpt_load_file` — the in-memory
  tests never touched the file loader; pins `_file_size`/`fdatasync`, the M8
  aarch64 save fix), checkpoint rejection smokes (+ `-18` pre-alloc bound,
  `-19` rng=0, the v3 `-32…-39` matrix incl. the merge-table forgery cascade)
  + v2/GQA/**v3** round-trip + **v1/v2-compat load**, the **BPE suite** (known-merge,
  round-trip, cross-arch determinism, generation bit-identity), the
  **eval/bits-per-byte** determinism + RNG-neutrality pin, the **KV
  bit-identity suite** (prefill at every prefix + decode across
  context-shifts, greedy + temperature, at hd ∈ {4, 6, 8, 10} ×
  nkv ∈ {1, 2, nh} incl. odd-T shifts, **preset shape**, and V=300), the
  **MLA latent KV-cache bit-identity suite** (1.2.1, `test_kv_mla`:
  cached-vs-uncached prefill + decode across shifts, greedy + temperature, at
  hd ∈ {6, 8, 10} × `d_c = C/2`/`d_c ∤ C`, odd T, 2-token window), the **coupled
  RoPE suite** (1.2.2: `test_rope_op` — bit-exact rotation backward +
  relative-position invariance; `test_attention_rope` — attention grad-check at
  hd ∈ {6,8,10} × MHA/GQA/MQA incl. the now-real K-bias gradient;
  `test_model_rope` — full-model wiring + posemb-zero-gradient pin; `test_kv_rope`
  — cached-vs-uncached bit-identity across shifts; `test_ckpt_rope` — v4
  `pos_kind=1` round-trip + odd-hd/mla+rope rejections;
  config-cap rope gates), the **decoupled RoPE suite** (1.2.3: `test_attention_mla_dec`
  — per-op decoupled grad-check (dWqr/dWkr + the shared-Kr backward, tight 1e-4) at
  3 d_c/d_rope configs; `test_model_mla_dec` — full-model wiring; `test_kv_dec` —
  cached-vs-uncached bit-identity across shifts incl. a non-even content head dim;
  `test_param_layout_mla_dec` — the W_QR/W_KR aliasing pin; `test_ckpt_dec` — v4
  `pos_kind=2`/`rope_dim` round-trip + decoupled hostile rejections; alloc-accounting
  + config-cap decoupled gates), the **MoE suite** (1.3.0: `test_moe_op` — per-op
  combine grad-check (dx/dWg/dWe, 1e-4) at 4 configs incl. top-1 and K=N;
  `test_moe_aux` — the load-balance aux backward vs FD (1e-5, non-uniform dispatch);
  `test_model_moe` — full-model wiring (dWe/dWgate/dWq/dtokemb, 1e-3); `test_kv_moe`
  — cached-vs-uncached bit-identity (top-1/top-2/K=N, odd T, 2-token window);
  `test_param_layout_moe` — the experts+gate tiling/aliasing pin; `test_ckpt_moe` —
  v5 round-trip + `-44`/`-45`/`-10` rejections; `test_ckpt_v4_compat`; MoE
  alloc-accounting + config-cap gates), the **SSM suite** (1.4.2: `test_ssm_core`
  per-op BPTT ~1e-7, `test_model_ssm`, `test_kv_ssm`, `test_ckpt_ssm`), the
  **per-layer hybrid suite** (1.4.3 + 1.4.4: `test_model_hybrid` — the MIXED mha/lin
  full-model grad-check ~1e-5; `test_model_hybrid_ssm`/`_mla` — the SSM/MLA ⊕ MHA
  mixed backward through the padded layout ~1e-4; `test_kv_hybrid` — cached-decode
  bit-identity across mha/lin, mha/ssm, mha/mla interleavings; `test_ckpt_hybrid` —
  the v6 per-layer round-trip (incl. a padded mha/ssm image) + `-46` rejects; hybrid
  config-cap + alloc-accounting pins for all kinds), the **char-diffusion suite**
  (1.5.0: `test_masked_ce` — the masked-CE backward vs FD (+ all-ones ≡ plain CE,
  all-zeros ⇒ 0 pins); `test_attn_bidir` — the bidirectional attention backward at
  T≥3 + a future-dependency pin; `test_model_diffusion` — the full-model FD over
  mask_emb AND tokemb at mixed masked/given positions; `test_diffusion_decode_-
  determinism` — the greedy parallel decode resolves every position + is
  reproducible; `test_ckpt_diffusion` — the v7 round-trip + `-47` rejects + the
  AR-still-v5 pin), the **token-packing suite** (1.5.3: `test_token_packing` —
  `gd_ld`/`gd_st` round-trip at u8/u16 incl. boundary ids 255/256/767, the
  width-invariance pin (a byte-level corpus reads back IDENTICAL ids at u8 and u16),
  and the `bpe_learn` u8→u16 self-widen; RNG-neutral, registered last), a
  **soak/leak** test and a **NaN guard** test. All pass on x86_64 AND aarch64
  (`cyrius test`; aarch64 via qemu).
- `tests/attn11.bcyr` — benchmark harness (training timings + tokens/sec,
  generation cached/uncached, KV bytes per nkv, MQA timings, **preset
  train+gen**, **MLA latent-cache gen + the cache-bytes table** (latent vs
  MHA/MQA full-K/V), **RoPE train-step + cached-gen overhead**, **MoE train-step +
  cached-gen + param count**, **`bpe_learn` cost**, **diffusion fwd+bwd step +
  T-round parallel decode** (1.5.0)).
- `tests/attn11.fcyr` — fuzz harness: 500 mutated-checkpoint rounds (v2/v3
  header fields incl. nkv/step, + a **boundary-combination** mode: every size
  field at/over its cap at once) + **500 BPE-image rounds** (merge-slot
  clobber, (V,Vb,K) triple inconsistency, expansion-bomb rewrite, + the
  **max-vocab triple** `V=768/Vb=256/K=512`) + **500 v7 diffusion-image rounds**
  (1.5.0: objective-field clobbers + diffusion-vs-mixer/rope/MoE combos) + 100
  random corpora + a **BPE round-trip** property; loaders reject malformed input
  without crashing.
- **`make smoke`** (1.4.5, in the `release` gate) — a CLI-arg regression: hostile
  `--layers`/`--attn-every` combinations must reject cleanly (exit < 128, never a
  signal) and a valid hybrid must still build. Closes the CLI arg path the
  grad-check/fuzz harnesses don't exercise (pins the 1.4.5 stack-overflow fix).
- The M2 (persistence), M3 (SIMD), M5 (AGNOS port), M6 (KV-cache/GQA),
  **M7 (BPE/preset/v3)**, and **M8 (security sweep)** code each passed an
  adversarial multi-agent review; all confirmed findings fixed and
  regression-tested. M6's review (50 agents) drove the checkpoint pre-alloc
  bound; M7's (9 agents) caught the **`--layers` heap-OOB**; M8's survey→map
  (12 agents) confirmed checkpoint **format immunity** to the model-file-deser
  RCE genre and surfaced the **AGNOS `--load`** crash + the **aarch64 save**
  break (qemu `fsync`). See CHANGELOG +
  [`../audit/2026-06-11-m8-security-sweep-audit.md`](../audit/2026-06-11-m8-security-sweep-audit.md)
  + [`../audit/2026-06-11-m7-bpe-audit.md`](../audit/2026-06-11-m7-bpe-audit.md)
  + [`../audit/2026-06-11-kv-gqa-audit.md`](../audit/2026-06-11-kv-gqa-audit.md).
- The M5 run gate (`scripts/agnos-smoke.sh`) is a developer-side check (needs
  the agnos/gnoboot/agnoshi sibling repos); it now also exercises `--load`
  under AGNOS (M8). CI gates the `--agnos` build + static-ELF shape only.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math,
  ganita, args
- **[rosnet](https://github.com/MacCracken/rosnet) 0.1.0** — tensor storage +
  BLAS-1 + dense matmul/gradient (`linear_fwd`/`linear_bwd`, `t_*`); 1.1.0
  extraction, pinned in `cyrius.lock`
- **[tyche](https://github.com/MacCracken/tyche) 0.1.0** — deterministic
  statistical PRNG (`rng_seed`/`rng_u64`/`rng_uniform`/`rng_normal`); 1.1.0
  extraction, pinned in `cyrius.lock`

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). **Shipped: v1.0.0 (clean cut) → v1.1.0
(extraction) → v1.2.0–1.2.4 (M12: MLA core, latent KV-cache decode, coupled +
decoupled RoPE; then 1.2.4 toolchain realign + docs) → v1.3.0 (M13: Mixture of
Experts) → v1.4.0 (M14 rung a: gated linear attention) → 1.4.1 (refactor sweep) →
v1.4.2 (M14 rung b: selective SSM) → v1.4.3 (M14 rung c: per-layer hybrid) → v1.4.4
(M14 rung d: any-mixer hybrids — completes M14) → v1.5.0 (M15: char-diffusion
objective).** The surface is frozen ([`STABILITY.md`](../STABILITY.md)) and
additive-only past 1.0; the numeric core lives in **rosnet** + **tyche**. The 1.x
arc now has the attention/position axes
`--attn-kind {mha, mla, lin, ssm}` × `--pos-kind {learned, rope, rope-decoupled}`,
the FFN-density axis `--experts N --expert-topk K`, **two non-softmax mixers**
(gated linear attention + the selective SSM), the **per-layer hybrid**
`--attn-every K` (any mix of the four), and the **training-objective axis**
`--objective {ar, diffusion}`. **M12, M13, M14, and M15 are all complete.**

**Next — M16 (the data-ingestion & curation 1.5.x arc is CLOSED).** M12–M15 shipped: the
architecture arc (MLA/RoPE, MoE, the mixer family + hybrid) closed at 1.4.6, then
M15 (1.5.0) the first *training-objective* departure (char-diffusion; X015 — AR wins
at this tiny scale, the super-data-learner advantage is a scale phenomenon), then
v1.5.1 the C4 data-ingestion tooling (X016). The **1.5.x data arc** (per [`roadmap.md`](roadmap.md))
**is complete**: **1.5.2** quality-curating sampler (X017 — the prose-quality filter cut
bits/byte 5.9%; multi-shard diversity is a scale lever); **1.5.3** token-packing unlock
(X018 — u8/u16 `g_data` vs i64; removed the 8×/4× bloat, raised
`MAX_CORPUS_BYTES` 4 MB → 64 MB; default run byte-identical); **1.5.4** curation at
scale (X019 — a 24 MB/12-shard curated corpus, 6× the old cap; the
diversity/volume penalty halves with capacity, +5.4%→+2.8%, so more clean data starts
paying off at scale; binary unchanged); **1.5.5** hardening/audit/security pass (P(-1),
the ninth audit — GO, the one finding a `c4_sample.py` gzip-bomb OOM, fixed; binary
unchanged) **closing the arc**. Then **M16** ternary
(BitNet-style) training (E6), into whose 1.6.x group **streaming token-shard
ingestion** folds (RAM-independent large corpora — it pays off once a scaled model
can absorb the data); **M17** RL (E9) last. Open diffusion fast-follows:
MLA/MoE/RoPE-diffusion, a stochastic/temperature decode, a scaled AR-vs-diffusion
run. A vidya-scale mixer/hybrid bake-off remains the standing X-entry.

### Handoff — how to pick this up

- **Build/test/release**: `make check` (lint + x86 grad-checks) for the fast loop;
  `make release` (lint + x86 + aarch64/qemu + DCE build + fuzz) is the full gate
  before tagging — both must exit 0. Quick refs at the top of
  [`CLAUDE.md`](../../CLAUDE.md) (Quick Start). `cyrius deps` resolves rosnet/tyche
  (pinned in `cyrius.lock`); a no-flag `./build/attn11` trains + samples.
- **The discipline that matters**: every hand-derived backward lands behind a
  finite-difference grad-check (`cyrius test`); kernel changes land in BOTH the
  batch and the cached single-row path or neither (the bit-identity contract,
  [`../architecture/003`](../architecture/003-cached-inference-bit-contract.md));
  the pin and `lib/` snapshot move together (`cyrius update` after any bump, then
  retest both arches); additive-only past 1.0 (new flags + a new checkpoint
  version with permanent back-compat). New backward op → grad-check it first,
  then plumb (the M12.5 increments are the worked example: per-op grad-check in
  isolation, then model wiring, then the cached path + bit-identity).
- **Where things live**: live state here; forward plan in
  [`roadmap.md`](roadmap.md); shipped narrative in [`../../CHANGELOG.md`](../../CHANGELOG.md);
  experiment evidence in [`experiments.md`](experiments.md); decisions in
  [`../adr/`](../adr/); non-obvious invariants in [`../architecture/`](../architecture/);
  the frozen surface in [`../STABILITY.md`](../STABILITY.md).

**Loose ends / known items**: (1) an `attn11` row upstream in
`agnos/scripts/stage-tools.sh` (`stage_one attn11 src/main.cyr attn11`) — a
cross-repo edit on the agnos maintainer's side, not actionable here. (2) The
**MLA absorption** compute optimization (fold `W_UK` into `W_Q` to attend latents
directly, avoiding the per-step K/V re-materialization in 1.2.1/1.2.3) is deferred
— it reorders accumulation, so it would ride its own bit-identity story; the MoE
combine has an analogous folding option but the same caveat. (3) A vidya-scale
perplexity bake-off across the mixers/positions/MoE-density/hybrid-ratio axes
remains an open X-series follow-on (the reference-scale comparisons ran as
X009–X014; the scaled run is the follow-on). (4)
The MoE aux coefficient α is fixed at 0.01 (no `--aux-alpha` flag); an α sweep is
a small additive follow-on if it earns one. The pin is now **6.2.2** (realigned
1.3.0, byte-identical `lib/` snapshot); the cycc argv-capture issue is resolved
upstream (6.1.32; `docs/architecture/002` retired).
