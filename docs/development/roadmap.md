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

Current: **v1.3.0**. The v1.0 surface is frozen and additive-only
([`STABILITY.md`](../STABILITY.md)); the reusable numeric core lives in
**[rosnet](https://github.com/MacCracken/rosnet)** + **[tyche](https://github.com/MacCracken/tyche)**
(v1.1.0); the **1.x architecture arc** is underway — **M12 and M13 are complete**:
the attention/position axes (MLA core v1.2.0, latent KV-cache decode v1.2.1,
coupled RoPE v1.2.2, decoupled RoPE v1.2.3 — the full `--attn-kind {mha, mla}` ×
`--pos-kind {learned, rope, rope-decoupled}` switch) and the **FFN-density axis**
(Mixture of Experts v1.3.0 — `--experts N --expert-topk K`, checkpoint v5, ADR
0008). See [`CHANGELOG.md`](../../CHANGELOG.md) for the full shipped narrative back to v0.1.0
and the M0–M11 / v1.0-cut history.

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

> **M12 and M13 are complete.** M12 (MLA core + latent KV-cache decode + coupled +
> decoupled RoPE, v1.2.0–v1.2.3) shipped the full `--attn-kind` × `--pos-kind`
> switch; M13 (Mixture of Experts, v1.3.0) shipped the FFN-density axis
> `--experts N --expert-topk K` — a frozen top-K router with a Mixtral-style
> renormalized combine + a Switch load-balance aux loss (both grad-checked, ADR
> 0008), checkpoint v5, and the X009 density sweep. See the CHANGELOG and
> X005–X009 for the shipped narrative. The arc continues below.

### M14 — A second sequence-mixer family (v1.4.0) — E4

**Linear attention → selective SSM**, the survey's structural shift (hybrids with
~10–25% attention layers beat pure transformers). Ladder: (a) a gated
linear-attention/RetNet-style block (easiest hand-derivable backward — constant
state, linear compute), then (b) a minimal selective-SSM block (BPTT through the
scan), then (c) a per-layer `kind` config to interleave mixer types and sweep the
**hybrid ratio** at our scale. This is a large departure (a non-attention mixer),
so it rides after MLA + MoE. **Gate**: every new backward grad-checked; perplexity
vs the iso-param transformer on the vidya corpus, logged as an X-series entry.

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

- GPU / accelerator backends — attn11 is a CPU, scalar-f64 (then SIMD) reference
  implementation. (The survey's engine/serving layer — vLLM/SGLang, FP4 tensor
  cores, photonics — is *observed, not chased*; only its algorithmic ideas
  translate here.)
- Distributed / multi-process training.
- A general autodiff engine — gradients stay hand-derived and grad-checked.
- Windows / macOS as first-class training targets (cross-build only, if at all).
