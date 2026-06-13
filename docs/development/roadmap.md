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

Current: **v1.5.0**. The v1.0 surface is frozen and additive-only
([`STABILITY.md`](../STABILITY.md)); the reusable numeric core lives in
**[rosnet](https://github.com/MacCracken/rosnet)** + **[tyche](https://github.com/MacCracken/tyche)**
(v1.1.0). The 1.x architecture arc through M14 has shipped (the attention/KV, FFN-
density, and sequence-mixer axes), and **M15** (the char-diffusion *training
objective*, v1.5.0) is the first objective departure; **next up is M16** (below).
For *what* shipped, see [`CHANGELOG.md`](../../CHANGELOG.md) (release narrative),
[`experiments.md`](experiments.md) (the X-series), and [`state.md`](state.md) (the
live snapshot — current flags, counts, perf). This file is the plan ahead only.

## Versioning

`VERSION` is the single source of truth (`cyrius.cyml` derives it via
`${file:VERSION}`); bumps go through [`scripts/version-bump.sh`](../../scripts/version-bump.sh),
which also rewrites `CFG_VERSION` and stubs the CHANGELOG. The major is `1`, so
releases are stable, additive-only tags (no v2 fork is planned — the architecture
arc rides as 1.x minors). The local release gate is `make release` (lint + x86
grad-checks + aarch64/qemu + DCE build + fuzz + the `make smoke` CLI regression);
CI mirrors it.

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
> (`--objective diffusion`, v1.5.0) — the first objective departure. Detail lives
> in [`CHANGELOG.md`](../../CHANGELOG.md), ADRs 0007–0013, and
> [`experiments.md`](experiments.md) (X005–X015). **The plan ahead starts at M16.**

### M16 — Ternary (BitNet-style) training (v1.6.0) — E6

**Weights in {−1, 0, +1}** with a straight-through estimator — the precision
ladder's algorithmic endpoint, and a *natural fit* for an everything-is-i64
language: ternary matmul collapses to integer adds. **Gate**: the STE backward
documented + grad-checked where defined; accuracy vs the f64 baseline; the
i64-add matmul benched against the SIMD-f64 path.

### M17 — Reinforcement learning (v1.7.0) — E9

**Last in the chain, by design.** RL is an orthogonal *training-objective* layer,
not an architecture: it runs on whatever trunk exists (any `--attn-kind` /
mixer / precision), so it graduates after the architecture families are in place
and can fine-tune the best of them. The reference target is **policy gradient
(REINFORCE)** at char scale: sample a rollout from the current policy, score it
with a deterministic reward function, and weight the log-prob gradient by
`(R − b)` (advantage, `b` a moving-average baseline). The gradient
`∇ log π(a)·(R − b)` **reuses the existing softmax-CE backward** reweighted per
token — so the hand-derived backward is a small, grad-checkable delta over the
supervised path, exactly attn11's wheelhouse. Reward is a simple in-process
function at this scale (e.g. "is the sample valid-Cyrius / matches a target
pattern / hits a length target"), not a learned reward model. **Gate**: the
reward-weighted backward grad-checked against finite differences; a documented
RL-vs-SFT comparison (does the policy move toward the reward) logged as an
X-series entry. PPO/GRPO (clipped ratio, group baselines) noted as a heavier
follow-on only if REINFORCE earns it.

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
orthogonal): the *training-objective* and *precision* departures ride next — the
char-diffusion objective (M15, E5) then ternary training (M16, E6) — with RL (M17,
E9) last because it is an objective layer over a finished trunk, and the GPU
*compute* backend (M18, E-infra) sequenced TBD (it unblocks the B4 GPU benchmark).
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
