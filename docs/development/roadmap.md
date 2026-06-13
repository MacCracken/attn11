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

Current: **v1.4.4**. The v1.0 surface is frozen and additive-only
([`STABILITY.md`](../STABILITY.md)); the reusable numeric core lives in
**[rosnet](https://github.com/MacCracken/rosnet)** + **[tyche](https://github.com/MacCracken/tyche)**
(v1.1.0); the **1.x architecture arc** is underway — **M12, M13, and M14 are
complete**: the attention/position axes (MLA core v1.2.0, latent KV-cache decode
v1.2.1, coupled RoPE v1.2.2, decoupled RoPE v1.2.3), the **FFN-density axis**
(Mixture of Experts v1.3.0 — `--experts N --expert-topk K`, checkpoint v5, ADR
0008), **two non-softmax sequence mixers** (gated linear attention v1.4.0 —
`--attn-kind lin`, ADR 0009; the selective SSM v1.4.2 — `--attn-kind ssm`, ADR
0010; with 1.4.1 a refactor sweep between them), and the **per-layer mixer hybrid**
— rung c (v1.4.3, `--attn-every K`, checkpoint v6, ADR 0011) then rung d (v1.4.4,
any-mixer via a padded stride, ADR 0012). See [`CHANGELOG.md`](../../CHANGELOG.md) for
the full shipped narrative back to v0.1.0 and the M0–M11 / v1.0-cut history.

## Versioning

`VERSION` is the single source of truth (`cyrius.cyml` derives it via
`${file:VERSION}`); bumps go through [`scripts/version-bump.sh`](../../scripts/version-bump.sh),
which also rewrites `CFG_VERSION` and stubs the CHANGELOG. The major is `1`, so
releases are stable, additive-only tags (no v2 fork is planned — the architecture
arc rides as 1.x minors). The local release gate is `make release` (lint + x86
grad-checks + aarch64/qemu + DCE build + fuzz); CI mirrors it.

## The 1.x architecture arc

Past 1.0 the surface is **additive-only**, so each frontier experiment ships as
an opt-in 1.x minor — new `--flags`, a new checkpoint version with permanent
back-compat, the default run byte-identical. The build order is a **value ÷ risk**
call, not a dependency chain: the axes (attention/KV, FFN density, sequence
mixer, training objective, precision) are orthogonal and re-orderable. Each
milestone below graduates one frontier experiment (the **E-series**, logged in
[`experiments.md`](experiments.md)), is independently shippable, and lands ONE
change at a time behind its own grad-check / bit-identity gate.

> **M12, M13, and M14 are all complete.** M12 (MLA + RoPE, v1.2.0–v1.2.3) shipped
> the full `--attn-kind` × `--pos-kind` switch; M13 (Mixture of Experts, v1.3.0)
> shipped the FFN-density axis (ADR 0008, X009); **M14** shipped the second
> sequence-mixer family across four rungs — gated linear attention (a, v1.4.0, ADR
> 0009, X010), the selective SSM (b, v1.4.2, ADR 0010, X011), the per-layer hybrid
> (c, v1.4.3, ADR 0011, X012), and any-mixer hybrids via a padded stride (d, v1.4.4,
> ADR 0012, X013) — with 1.4.1 a refactor sweep along the way. See the CHANGELOG
> and X005–X013 for the shipped narrative. **Next: 1.4.5 hardening pass (P(-1))**,
> then the arc continues below (M15+).

### M14 — A second sequence-mixer family (v1.4.0+) — E4

**Linear attention → selective SSM → hybrid**, the survey's structural shift
(hybrids with ~10–25% attention layers beat pure transformers). Ladder:

- **(a) gated linear-attention/RetNet-style block** — **DONE (v1.4.0, ADR 0009).**
  `--attn-kind lin`: constant-`nh·hd²`-state retention, parameter-free, `attn_kind = 2`,
  per-op grad-check ~1e-9. X010 = the MHA/MLA/linear comparison.
- **(b) a minimal selective-SSM block** — **DONE (v1.4.2, ADR 0010).** `--attn-kind
  ssm`: a Mamba-lite diagonal SSM with input-dependent Δ/B/C, `attn_kind = 3`
  reusing Wq/Wo + `latent_dim` (state N, no checkpoint bump); hand-derived BPTT
  through the data-dependent scan (~1e-7), constant `C·N` decode cache. X011 adds
  it to the comparison (best bits/byte at reference scale).
- **(c) a per-layer `kind` config** — **DONE (v1.4.3, ADR 0011).** `--attn-every K`:
  the global `g_attn_kind` becomes a per-layer `g_layer_kind` read only by the three
  `_attn_block_*` dispatch helpers (the 1.4.1 refactor localized it). Restricted to
  layout-compatible kinds {mha, gqa, lin} so the per-block stride stays uniform — the
  hybrid is parameter-free, and the decode cache scales with the attention fraction.
  First checkpoint format bump (**v6** carries the per-layer pattern). X012 = the
  attention-fraction sweep.
- **(d) any-mixer hybrids** — **DONE (v1.4.4, ADR 0012).** Lifts (c)'s {mha, gqa,
  lin} restriction so MLA/SSM layers join a hybrid (full attention ⊕ SSM, the
  survey's strongest pairing). Each block's K/V region is PADDED to the max `_kvw`
  over the present kinds (`_kvw_hyb`) — uniform stride, so only `_kv_weight_size()`
  + the per-layer init/cache gates change, not every `_o_*` offset. v6 unchanged
  (carries the per-layer kinds). Mixed SSM/MLA ⊕ MHA backward grad-checked (~1e-4);
  X013 = the attention ⊕ SSM ratio sweep. **M14 complete.**

**Gate**: every new backward grad-checked; perplexity vs the iso-param transformer
on the vidya corpus, logged as an X-series entry.

### M15 — Char-diffusion objective (v1.5.0) — E5

**A dLLM at reference scale** — same trunk, different *training objective*: a
masked-denoising loss (drop the causal mask, predict masked bytes,
confidence-aware parallel decode). Tests the survey's "diffusion LMs are super
data learners" thesis honestly at tiny scale: AR vs diffusion on repeated epochs
of the same small corpus. **Gate**: grad-check the masked-CE path; a
matched-compute AR-vs-diffusion comparison logged as an X-series entry.

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

The order is **value ÷ risk** and re-orderable (the axes are orthogonal):
decoupled RoPE finishes the lowest-risk KV-cache arc M12 opened; MoE is the
requested scale-up knob; the mixer / objective / precision departures (E4–E6)
are the biggest architectural changes and ride later; RL (E9) is last because it
is an objective layer over a finished trunk. The E-series is informed by the
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
