# attn11 — Roadmap

> **Forward sequencing only** — what ships next, in what order, against what
> gates. Shipped history lives in [`CHANGELOG.md`](../../CHANGELOG.md) (the
> release narrative) and [`experiments.md`](experiments.md) (the X-series ledger);
> live state (current version, test/assertion counts, perf numbers) lives in
> [`state.md`](state.md). This file is the plan ahead.
>
> Every milestone keeps the **invariants**: hand-derived backward,
> finite-difference grad-checked (`cyrius test` green on x86_64 **and**
> aarch64/qemu), `src` lints clean, the no-flag run stays byte-identical, and any
> new checkpoint version keeps permanent back-compat (older images always load).

## Where we are

Current: **v1.7.1**. The v1.0 surface is frozen and additive-only
([`STABILITY.md`](../STABILITY.md)); the reusable numeric core lives in
**[rosnet](https://github.com/MacCracken/rosnet)** + **[tyche](https://github.com/MacCracken/tyche)**
(v1.1.0). The 1.x architecture arc through M14 has shipped (the attention/KV, FFN-
density, and sequence-mixer axes), **M15** (the char-diffusion *training objective*,
v1.5.0) is the first objective departure, the **data-ingestion & curation 1.5.x arc
is COMPLETE** (v1.5.1 the C4 tooling X016 → v1.5.2 the quality-curating sampler X017 →
v1.5.3 the token-packing unlock X018 → v1.5.4 curation at scale X019 → v1.5.5 the
hardening/audit pass → v1.5.6 the held-out `--eval-corpus` follow-on X020), and **M16
(ternary / BitNet-style training, E6) is COMPLETE** — `--ternary` weights in {−1,0,+1}
(absmean scale) with a straight-through estimator, grad-checked, default run byte-identical,
checkpoint v8 (v1.6.0, X022), then the **i64-add ternary matmul + bench** closing the M16
gate (v1.6.1, X023 — the collapse is exact + grad-checked but ~2.4–3× slower than wide-SIMD
f64 at this scale, an honest negative; the default forward keeps the SIMD-f64 path). The
1.6.x group's remaining work (streaming ingestion, the X021 data-volume held-out win) is
below. For *what* shipped,
see [`CHANGELOG.md`](../../CHANGELOG.md)
(release narrative), [`experiments.md`](experiments.md) (the X-series), and
[`state.md`](state.md) (the
live snapshot — current flags, counts, perf). This file is the plan ahead only.

## Versioning

`VERSION` is the single source of truth (`cyrius.cyml` derives it via
`${file:VERSION}`); bumps go through [`scripts/version-bump.sh`](../../scripts/version-bump.sh),
which also rewrites `CFG_VERSION` and stubs the CHANGELOG. The major is `1`, so
releases are stable, additive-only tags (no v2 fork is planned — the architecture
arc rides as 1.x minors). The local release gate is `make release` (lint + x86
grad-checks + aarch64/qemu + DCE build + fuzz + the `make smoke` CLI regression);
CI mirrors it.

## Data ingestion & curation (the 1.5.x arc) — ships next

> An infra/data sub-arc, not an architecture milestone — it ships BEFORE M16 (it is
> the immediate next work). The driver (from v1.5.1 / X016): a tiny attn11 is
> **data-rich and capacity-poor** — 4 MB already exceeds what a 40 K–250 K-param
> model absorbs (the C4 run saw only ~8% of one epoch in 600 steps), and `g_data`
> stored **one i64 per token (8 B/token)** against a **256 MB single-allocation cap**
> (`ALLOC_MAX`), so the corpus ceiling was ~32 MB (**1.5.3 packed it to u8/u16,
> lifting that ~4–8× and raising `MAX_CORPUS_BYTES` to 64 MB**). So **"data quality >
> volume"** (the frontier survey) governs the near term, packing bought cheap
> headroom, and the RAM-independent path (streaming) waits for a model big enough to
> need it. Each item is tooling/data + a logged X-entry; the no-flag binary stays
> byte-identical except 1.5.3 (a transparent storage change) and 1.5.5 (the audit).

### 1.5.2 — Quality-curating sampler (sharpen quality) — ✅ shipped (X017)

Upgraded `scripts/c4_sample.py` to a curating sampler: exact + prefix
de-duplication, **multi-shard sampling** (`--shards N`), and prose/register filters
(letter/digit ratio, terminal punctuation, avg word length, repetition, long-token
spam). Tooling + data only; defaults reproduce the raw slice byte-for-byte. **Gate
met**: the quality filter cut eval bits/byte **3.43 → 3.23 (−5.9%)** at iso-compute
(same shard). Finding (X017): multi-shard *diversity* raised bits/byte for a tiny
model — **diversity/volume is a scale lever**, not a tiny-model one (it lands with
the model-scale work, M16+), so curate for *quality* now.

### 1.5.3 — Token-packing unlock — ✅ shipped (X018)

Stored the token stream **packed** — `u8` for byte-level (vocab ≤ 256), `u16` for BPE
(vocab ≤ 768) — instead of one i64 per token, removing the 8×/4× `g_data` bloat and
raising `MAX_CORPUS_BYTES` **4 MB → 64 MB** (the u16 store is then 128 MB, half the
256 MB per-alloc cap). A `g_data_w` width global + width-generic accessors
(`gd_ld`/`gd_st`, `_tw_ld`/`_tw_st`) over the corpus buffer; only `g_data` is packed
(the T-sized batch buffers stay i64); `bpe_learn` self-widens a u8 corpus to u16;
`tok_encode` gained an output-width arg (i64 prompt scratch vs the packed corpus
rebuild). See [`../architecture/006-packed-token-store.md`](../architecture/006-packed-token-store.md).
**Gate met**: default/u8, BPE-64/u16, and `--preset` runs **byte-identical** to 1.5.2;
**977** grad-checks green on x86_64 AND aarch64/qemu; lint + fuzz (100 random corpora
+ BPE round-trip) + `make smoke` green; a 6 MB corpus (byte + BPE) loads/trains, a
65 MB corpus rejects cleanly. Finding (X018): the headroom is real and free (ids ≤
767 fit u8/u16), but diversity/volume remains a *scale* lever — this lifts the ceiling
1.5.4 will fill.

### 1.5.4 — Curation at scale — ✅ shipped (X019)

Used 1.5.3's higher ceiling to curate a **24 MB / 12-shard** C4-en corpus (6× the old
4 MB cap, impossible before the packed store) and ran the scaled data+capacity
experiment vs the 4 MB curated baseline: two model sizes (default ≈53 K, preset
≈232 K) × two corpora, BPE 256, matched compute. Tooling + data + a logged run; binary
unchanged. **Gate met** (an honest result): eval bits/byte default 3.232/3.405 (4/24
MB), preset 2.666/2.741. Capacity dominates (default→preset −17.5%/−19.5%), and the
**diversity/volume penalty halves with capacity** (default +5.4% → preset +2.8% going
4→24 MB) — the first attn11 evidence that more clean data starts paying off as the
model grows; the 4 MB default cell reproduces X017's 3.232 bit-for-bit (curation +
1.5.3 packing both transparent). Honest caveat: bits/byte-on-own-corpus understates
data's *generalization* value — a held-out **`--eval-corpus FILE`** flag (re-encode a
disjoint corpus through the loaded tokenizer) is the clean way to score it, a small
additive follow-on deferred to keep 1.5.4 binary-unchanged.

### 1.5.5 — Hardening / audit / security pass (P(-1)) — ✅ shipped (closes the arc)

The standard pre-minor hardening (CLAUDE.md P(-1)): cleanliness, a benchmark baseline,
and a **security audit** of the surface the 1.5.x arc added — the raised corpus cap +
packed-store bounds (1.5.3), the curation script's untrusted-input handling
(1.5.1/1.5.2/1.5.4), and the diffusion path (1.5.0). Adversarial multi-agent review
(the ninth audit): five read-only dimensions, each medium+ finding adversarially
verified. **Verdict GO, 0 blockers**; four dimensions clean, **one** confirmed finding
fixed — a **gzip-bomb / unbounded line buffer** in `c4_sample.py` (`for line in gz`
materialized a whole line, so a no-newline runaway shard member OOMs), fixed with a
bounded `iter_lines()` (chunked read, 8 MB per-line cap, drop+resync) and
regression-verified (normal extraction unchanged, a 12 MB bomb completes with no OOM,
re-curation byte-identical). The Cyrius binary is unchanged (Python-side fix). Report:
[`../audit/2026-06-13-1.5.x-hardening-audit.md`](../audit/2026-06-13-1.5.x-hardening-audit.md).
**Gate met**: audit filed, finding fixed + regression-verified, `make release` green —
the 1.5.x arc is closed. Deferred follow-on noted by X019: a held-out
**`--eval-corpus FILE`** flag (the clean way to score data's generalization value) —
**✅ shipped in 1.5.6 (X020)** below.

### 1.5.6 — Held-out (cross-corpus) eval — ✅ shipped (X020)

The deferred X019 follow-on: a **`--eval-corpus PATH`** flag that scores the model on a
**disjoint** corpus, re-encoding it through the loaded tokenizer (same byte vocab / BPE
merges, no vocab-order check) and running the active objective's eval. The
generalization metric X019 flagged as missing: bits/byte-on-own-corpus understates
data's value in absolute terms; a held-out pass measures it directly. Additive and
surgical — the no-flag run stays **byte-identical** to 1.5.5; RNG-neutral; combines with
`--eval` and works under `--gen-only --load`; failures (missing/empty/too-short file)
set a non-zero exit (mirrors `--save`) and never crash. New `test_eval_held` pins the
core invariant (held-out of the OWN bytes reproduces `eval_corpus()` bit-for-bit, byte
+ BPE): **977 → 986** checks, green x86_64 + aarch64/qemu; lint + fuzz + `make smoke`
(new held-out case) green. Run as **X020** (the first attn11 cross-corpus generalization
numbers — see [`experiments.md`](experiments.md)): the own → held-out gap is **< 1.3%**
at both scales, so sub-epoch training barely memorizes and own-corpus bits/byte is
already a near-unbiased generalization proxy (validating X016–X019). The data-volume
held-out win (train-4MB vs train-24MB, eval a third disjoint set) is the **X021**
follow-on, landing at M16+ where a model can actually overfit a small corpus.

### Streaming token-shard ingestion — ✅ shipped (v1.6.2)

The RAM-independent large-corpus path: **pre-encode** a corpus to a token-shard once
(`--encode-shard`), then **sample windows by offset** (`--stream-corpus`) — decoupling
corpus size from RAM (GB+). Shipped as **lseek+read by offset** through a single
64 K-token chunk cache inside `gd_ld` (NOT `mmap` — the lib `mmap` wrapper is x86_64-only
and the agnos target has no file-backed mmap; lseek+read is portable, so the cross-arch
gate holds). Because `gd_ld` is the only corpus-token reader, `sample_window`/`eval_corpus`/
the model are untouched and byte-identity is structural. `scripts/c4_sample.py --emit-shard`
is the GB-scale byte-level producer (byte-identical to `--encode-shard`). **Gate met**:
streamed training reproduces the in-memory run BIT-FOR-BIT (byte + BPE, across chunk
boundaries) on x86_64 **and** aarch64/qemu; RSS is flat ~13 MB as the corpus grows
(bounded RAM, the GB precondition); the no-flag run is byte-identical; lint + fuzz
(500 shard rounds) + `make smoke` green. New file `src/stream.cyr`,
[`../architecture/007-streaming-token-shards.md`](../architecture/007-streaming-token-shards.md).
**Fast-follows** (additive), both now shipped: resume-from-stream (`--load` +
`--stream-corpus`) — **✅ v1.6.3** (bit-for-bit deterministic; the loader requires the
shard's full tokenizer to match the checkpoint's, `-15` otherwise) — and BPE GB-scale
shards (`--stream-encode`) — **✅ v1.6.4** (a bounded-RAM streaming encoder: learn merges
on a bounded prefix, then chunked `tok_encode` with a `2*BPE_MAX_TOKLEN` raw-byte carry
that makes the chunked encode byte-identical to a whole-corpus encode). The c4 emitter
stays byte-level; `attn11 --stream-encode` is the BPE GB path.

### Data-volume held-out win (X021) — ✅ shipped (v1.6.5)

The open experiment from X020, the one `--eval-corpus` was built to run. It needs the
**overfit** regime X020 lacked (sub-epoch barely memorizes, so own≈held); X021 forces it
with a small overfittable slice vs a larger disjoint one at matched compute, scored on a
third disjoint slice (`scripts/x021-heldout.sh`; **no new binary surface** — rides the
shipped `--eval-corpus` + `c4_sample.py`, binary byte-identical to 1.6.4). **Result (an
honest win, not a null):** at **preset** capacity, training on **4 MB** beats **256 KB** by
**−14.2% held-out bits/byte** (2.383 vs 2.776) — the data-volume generalization win X019
hypothesized; at **default** (tiny) capacity a **tie** (−0.06%), so the win is a **capacity
lever**. The overfit regime is reached (preset·256 KB own→held gap **+40.4%**), exposing
that own-corpus bits/byte **inverts the truth above the overfit threshold** (own looks
better while held-out is far worse) yet stays honest below it (validating X020). The scale
deviates from the roadmap's literal 4 MB vs 24 MB — attn11's largest (preset) cannot
overfit 4 MB of diverse C4, so 256 KB vs 4 MB is the scale where the question is decidable;
documented in X021. **Closes the data-ingestion story (X016→X021).**

## The 1.x architecture arc

Past 1.0 the surface is **additive-only**, so each frontier experiment ships as
an opt-in 1.x minor — new `--flags`, a new checkpoint version with permanent
back-compat, the default run byte-identical. The build order is a **value ÷ risk**
call, not a dependency chain: the axes (attention/KV, FFN density, sequence
mixer, training objective, precision) are orthogonal and re-orderable. Each
milestone below graduates one frontier experiment (the **E-series**, logged in
[`experiments.md`](experiments.md)), is independently shippable, and lands ONE
change at a time behind its own grad-check / bit-identity gate.

> **Shipped (M12–M15, v1.2.0–v1.5.0):** the attention/KV axis (MLA + the
> `--pos-kind` RoPE switch), the FFN-density axis (MoE, `--experts`), the second
> sequence-mixer family (`--attn-kind {lin,ssm}` + the any-mixer per-layer
> `--attn-every` hybrid), then 1.4.5 hardening + 1.4.6 benchmarking to close the
> architecture arc, and **M15** the char-diffusion *training objective*
> (`--objective diffusion`, v1.5.0) — the first objective departure — plus v1.5.1's
> C4 data-ingestion tooling. Detail lives in [`CHANGELOG.md`](../../CHANGELOG.md),
> ADRs 0007–0013, and [`experiments.md`](experiments.md) (X005–X019). **The
> data-ingestion & curation 1.5.x arc (the section above) is complete; the plan ahead
> is M16, into whose 1.6.x group the two open data items above fold.**

### M16 — Ternary (BitNet-style) training (v1.6.0) — E6

**Weights in {−1, 0, +1}** with a straight-through estimator — the precision
ladder's algorithmic endpoint, and a *natural fit* for an everything-is-i64
language: ternary matmul collapses to integer adds.

**Increment 1 — ✅ shipped (v1.6.0, X022, ADR 0014).** `--ternary`: each quantized
weight matrix becomes `W_eff = γ·clamp(round(W/γ),−1,+1)` (γ = absmean), master
weights stay f64, the STE backward is the `dx`-through-`W_eff` + `dW` pass-through
that rosnet's `linear_bwd` gives for free (it never reads `W` for `dW`). Scope: MHA
+ dense MLP + uniform + AR + learned-abs (the other axes are documented fast-follows).
Grad-checked **where defined** (`dx` FD at 1e-5; the STE `dW` pinned bit-exact;
full-model smooth-param FD), checkpoint **v8** (slot [23], `-48`/`-49` rejects),
default/preset/BPE runs **byte-identical**, **986 → 1010** checks green x86_64 +
aarch64/qemu. Accuracy vs f64 logged as X022 (ternary lowers capacity → worse
bits/byte at reference scale, as expected — the deliverable is "it learns + is
grad-checked", not a win at this scale).

**Increment 2 — ✅ shipped (v1.6.1, X023, the i64-add matmul + bench; closes the M16
gate).** `x·W_eff = γ·(x·t)`, `t∈{−1,0,+1}`, collapses the multiply to add/subtract/skip
plus one γ-scale per output. Two reference kernels `ternary_matmul_fwd`/`ternary_matmul_dx`
(`ops.cyr`), grad-checked (forward + `dx` pinned against the SIMD-f64 `W_eff` path at
maxrel 0, `dx` FD'd; **1010 → 1014** checks) and **benched head-to-head**. Honest result
(X023): the collapse is **~3× slower** (matmul) / **~2.4× slower** (end-to-end) than the
SIMD-f64 path on x86_64 — `f64v_fmadd` is 4-wide while the collapse is scalar add/sub/skip,
so the integer-add win needs activation quantization and/or non-FMA hardware (the orthogonal
1.58-bit *memory* win is real, unmeasured here). The default ternary forward **keeps the
SIMD-f64 path**; the collapse ships as the grad-checked + benched reference kernel (wired
into no run — ternary *and* default runs byte-identical to 1.6.0). The remaining ternary
fast-follows (mla/ssm/lin/MoE/rope/diffusion) and activation quantization are additive.

> **The 1.6.x group's two data items are both done:** **streaming token-shard ingestion**
> — **✅ v1.6.2** (`--encode-shard` / `--stream-corpus`), with resume-from-stream (✅ v1.6.3)
> and BPE GB-scale shards (✅ v1.6.4) as the fast-follows — and the **data-volume held-out
> win (X021)** — **✅ v1.6.5** (an honest win: more clean data generalizes −14.2% better on
> held-out at capacity; the overfit regime shows own-corpus bits/byte inverts above it).
> **The 1.6.x group is complete; M16 and its group are closed. Next is M17 (RL).**

### M17 — Reinforcement learning (REINFORCE) — ✅ shipped (v1.7.0)

**Last in the milestone chain, by design** — an orthogonal *training-objective* layer
over the finished AR trunk. `--objective rl`: on-policy **REINFORCE** (Williams 1992) —
sample rollouts from the policy (temperature 1), score each with a deterministic reward
(count of `--rl-target C` per rollout), weight the log-prob gradient by the advantage
`(R − b)` (`b` = EMA baseline). The realization that made it a *small* milestone:
`∇log π(a) = −∇CE(a)`, so the policy gradient is the **existing softmax-CE backward over
the sampled rollout scaled by `(R − b)`** — no new forward, no new backward op, no
checkpoint bump (the model stays plain AR; an RL image is a normal v5). The scale is
injected at `D_logits` in `model_backward`, gated on `g_rl` (AR/diffusion + the no-flag
run byte-identical). **Gate met** (X024, an honest win): grad-checked three ways
(`test_rl_op`: RL grad == advantage × AR grad at maxrel 0; FD vs `advantage × CE`; sign +
zero limits) + the RL-vs-SFT comparison — the policy moves decisively toward the reward
(target-char freq 9–19% → ~99.7%) with the documented SFT→RL alignment tax (corpus
bits/byte 0.24 → ~13, naive reward hacking). ADR 0015; runner `scripts/m17-rl.sh`.
**PPO/GRPO + richer rewards (valid text, length/format targets)** are the documented
heavier follow-on — REINFORCE earns the question, not yet the heavier machinery.

### M18 — GPU backend (sequencing TBD) — E-infra

**Moved in from Out-of-scope per the user (2026-06-13).** A GPU *compute* backend
for the same hand-derived forward/backward — an execution target, not a new
dependency. The sovereign path: the f64 tensor ops (matmul, attention, LM head,
Adam) dispatch to the **[mabda](https://github.com/MacCracken/mabda)** GPU
foundation (already vendored in `lib/mabda.cyr`) with **ai-hwaccel** for device
detection — **no cuBLAS / cuDNN / autodiff**; the "everything-is-i64, hand-derived,
grad-checked" invariant is device-independent. The CPU scalar/SIMD path stays the
reference and the bit-exact oracle: every GPU kernel is gated by matching the CPU
result to f64 tolerance (the finite-difference discipline, one level up), and the
no-flag run stays CPU + byte-identical. **Gate**: each GPU op validated against the
CPU reference within tolerance. Sequencing is TBD relative to M15–M17 (the user
reads it as "a few updates away"); no version pinned yet. **This milestone unblocks
benchmark phase B4 (the GPU competitor comparison).**

## Competitor benchmarking (B-series)

> The existing `scripts/bench-history.sh` + `bench-history.csv` track is
> **self-referential** — attn11's own tokens/sec across its own versions. This
> series adds the missing axis: **throughput vs external references.** It is a
> measurement/infra track, not an architecture milestone — it runs continuously
> and is re-runnable per release, so it carries no single version tag.

**Two headline axes** (per the user 2026-06-13):

1. **Honest raw throughput** — tokens/sec at a *matched model config*, reported
   straight even where attn11 loses (it will, to OpenMP-multicore llm.c and to any
   GPU competitor), with a **context column**: thread count, shared-lib deps
   (`ldd`), total shippable bytes, single-static-ELF (y/n), peak RSS.
2. **Normalized: throughput-at-zero-dependencies** — the axis where attn11 is
   alone: tokens/sec carrying *no* BLAS/libc/CUDA, one static ELF (~312 KB). The
   raw table is shown, but the *story* leads with the dependency-closure framing.

**Competitors** (each pinned to a specific upstream commit/tag for reproducibility):

| competitor | lang | comparable on | deps |
|------------|------|---------------|------|
| **llm.c** (`train_gpt2`, CPU) | C | **training** step tok/s | libc + libm + OpenMP (multi-core) |
| **nanoGPT** | PyTorch | training **and** decode | PyTorch + (CUDA) — GB-scale |
| **llama2.c** (`run`) | C | **decode/gen** tok/s only (no train) | libc + libm |
| **micrograd** | Python | training, from-scratch peer / sanity floor | CPython runtime |

**Fairness rules** (the harness asserts these — no cherry-picking):

- *Matched config* — same vocab, `d_model`, `n_layers`, `n_heads`, context `T`,
  batch. The harness maps attn11's `--preset` to each competitor's config and
  **requires the printed parameter count to match within tolerance** before a row
  is accepted.
- *Same host, pinned* — one CPU (`taskset`), perf governor, warmup + a
  run-of-record; host CPU + thread count + date stamped into every CSV row.
- *attn11 single-thread FIRST* — its honest scalar baseline, then the SIMD path;
  competitors' thread counts (llm.c OpenMP, PyTorch MKL) are recorded in the
  context column, never hidden.
- *Two surfaces, separate tables* — (a) **training** step throughput
  (fwd+bwd+opt), (b) **decode** throughput (attn11's KV-cached gen vs llama2.c /
  nanoGPT generate).
- *No vendoring* — competitors are cloned + built at pinned refs into a gitignored
  `bench/` dir (license + size); the harness records the ref it built.

**Phases:**

- **B0 — harness scaffold.** `scripts/compete-bench.sh`: clone+build each
  competitor at its pinned ref, run the matched config, emit a CSV row
  (`competitor, ref, tokens_per_sec, surface, threads, deps_bytes, peak_rss, host, date`).
  New `competitor-bench.csv` (the self-bench `bench-history.csv` stays as-is).
- **B1 — CPU training throughput.** attn11 (1-thread → SIMD) vs llm.c-CPU vs
  nanoGPT-CPU vs micrograd, matched config. First external table → `docs/benchmarks.md`.
- **B2 — decode throughput.** attn11 KV-cached gen vs llama2.c `run` vs nanoGPT
  generate.
- **B3 — the normalized story.** Add the zero-deps / shippable-bytes / single-ELF
  context columns + write the headline framing into `docs/benchmarks.md`.
- **B4 — GPU comparison** *(rides M18)*. attn11-GPU vs nanoGPT-GPU vs llama2.c
  CUDA, same matched config, a `backend` column folded into the same tables.

**Gate**: reproducible (every competitor at a pinned ref; config-match asserted by
param count; host/threads/date in every row) and *complete* — every config run
gets reported, no dropped or cherry-picked rows (the "no silent caps" discipline).

## Sequencing intent

The remaining order is **value ÷ risk** and re-orderable (the axes are
orthogonal). The **data-ingestion & curation 1.5.x arc** is **complete** (1.5.2
quality-curating sampler ✓ → 1.5.3 token-packing ✓ → 1.5.4 curation at scale ✓ →
1.5.5 hardening/audit ✓, which closed the arc) — cheap, high-ROW infra that improved the data the *existing*
models see and lifted the corpus ceiling, before any new model-scale work. The
*precision* departure — **ternary training (M16, E6)** — is **complete** (v1.6.0 the
grad-checked STE X022 → v1.6.1 the i64-add matmul + bench X023, closing the gate). Its
**1.6.x group is now complete**: **streaming token-shard ingestion** (v1.6.2,
`--encode-shard` / `--stream-corpus`) with resume-from-stream (v1.6.3) and BPE GB-scale
shards (v1.6.4) as the fast-follows, and the **data-volume held-out win (X021, v1.6.5)** —
an honest win (more clean data generalizes −14.2% better on held-out at capacity). With
M16 + its group closed, **RL (M17, E9, v1.7.0)** — the last milestone in the chain —
**shipped** (REINFORCE as reward-weighted softmax-CE; the policy moves toward the reward,
X024). **The full M12–M17 milestone chain is now complete.** What remains is the GPU
*compute* backend (**M18, E-infra**, sequencing TBD — it unblocks the B4 GPU benchmark)
and the competitor-benchmarking **B-series** (a continuous measurement track); both are
infra, not new training science. The architecture / objective / precision / RL frontier
experiments (E1–E9) have all graduated.
The E-series is informed by the
June-2026 frontier survey (`ai-ml-frontier-2026-expanded.docx`, repo root) —
data quality > volume, the KV cache as the central inference object, SSM/attention
hybrids, diffusion LMs, the precision ladder — which attn11 adapts by building
**reference implementations of the ideas**, not by chasing the hardware. An
experiment graduates into a milestone only when it earns one; results land in
[`experiments.md`](experiments.md).

## Out of scope

- The **serving / engine** layer — vLLM/SGLang, FP4 tensor cores, photonics — is
  *observed, not chased*; only its algorithmic ideas translate here. (Note: a GPU
  *compute* backend is **no longer** out of scope — it moved in as **M18** above,
  per the user 2026-06-13. What stays out is the serving stack, not the device.)
- A **CUDA / cuBLAS / cuDNN** dependency — when the GPU backend (M18) lands it
  goes through the sovereign **mabda** + **ai-hwaccel** path, never a vendor
  BLAS/autodiff stack. The "no BLAS / no autodiff" invariant is device-independent.
- Distributed / multi-process training.
- A general autodiff engine — gradients stay hand-derived and grad-checked.
- Windows / macOS as first-class training targets (cross-build only, if at all).
