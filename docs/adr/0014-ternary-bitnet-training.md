# 0014 — Ternary (BitNet-style) weight quantization with a straight-through estimator

**Status**: Accepted
**Date**: 2026-06-14

## Context

M16 (E6) is the precision-ladder endpoint: weights in **{−1, 0, +1}** with a
per-matrix scale (BitNet b1.58, arXiv:2402.17764). It is a *natural* fit for an
everything-is-i64 language — a ternary matmul collapses to integer add / subtract /
skip — and the reference question is whether real gradient-based learning still
works when the forward weight is quantized to ~1.58 bits.

The constraints are the 1.x arc's: opt-in and **additive** (the no-flag run
byte-identical, every older checkpoint still loads), every new hand-derived
backward passes a finite-difference grad-check *where it is defined*, and the
ternary-vs-f64 comparison is honest. Two facts shape the design. First,
`linear_fwd`/`linear_bwd` live in the vendored `lib/rosnet.cyr`, which attn11 may
**not** modify — so quantization must be wrapped at the attn11 layer. Second,
ternary quantization is **not differentiable** (round/clip are piecewise
constant), so the weight gradient cannot be a literal finite difference of the
quantized forward; the standard answer is a straight-through estimator (STE).

## Decision

Add `--ternary` (default off): BitNet-style **fake-quant** training. The master
weights stay **f64** (`g_params` / Adam moments unchanged); the forward replaces
each quantized weight matrix `W(K,N)` with `W_eff = γ · t`:

- `γ = absmean(W) = (Σ|Wᵢⱼ|) / (K·N)` — a fixed row-major sum (bit-reproducible
  cross-arch), and
- `tᵢⱼ = clamp(round(Wᵢⱼ / γ), −1, +1)` ∈ {−1, 0, +1}, so `W_eff ∈ {−γ, 0, +γ}`.

Scope for v1.6.0 is deliberately narrow — **MHA only, dense MLP, uniform (no
hybrid), AR, learned-absolute positions** — so M16 lands one idea behind a tight
grad-check (mirroring how M15 diffusion scoped to MHA+dense+uniform first). The
load-bearing choices:

1. **Two attn11-local wrappers** `qlinear_fwd`/`qlinear_bwd` (in `ops.cyr`) around
   rosnet's kernels. With `g_ternary == 0` they are exact passthroughs, so the
   default run is **byte-identical**. With ternary on, `W` is quantized into a
   pre-allocated scratch `g_qscratch` (sized `C·F`, the largest quantized matrix,
   reused sequentially — only one quantized matmul is live at a time) and the
   matmul/backward run against `W_eff`. The 16 quantized call sites are the MHA
   Q/K/V/O projections (`attn_fwd`/`_bwd`/`_fwd_row`) and the dense MLP
   (`Wfc`/`Wproj`) in train / decode / eval.

2. **The STE is free, given rosnet's `linear_bwd`.** `dx = dy · W_effᵀ` uses the
   quantized weight (the true forward path); `dW = xᵀ·dy` is the STE pass-through —
   and `linear_bwd` already computes `dW` **without reading `W`**, so passing
   `W_eff` yields exactly the gradient the master weight wants (round/clip act as
   identity for the weight gradient). No new gradient math in the kernel. The same
   master `W` is quantized in both the fwd and bwd of a step (Adam updates only
   after backward), so `W_eff` is consistent within a step.

3. **Full precision stays full precision**: embeddings, the weight-tied LM head,
   LayerNorm γ/β, and all biases are NOT quantized (BitNet keeps these f64). The
   tied head riding `tokemb` is therefore unquantized.

4. **Grad-checked where defined, pinned where not** (`test_ternary_quant`): the
   `dx` path FD-checks at 1e-5 (the quantized forward is smooth in `x` — `W_eff` is
   fixed as `x` varies), the full-precision bias FD-checks, and the STE `dW` is
   **pinned bit-for-bit** against a plain `linear_bwd` `dW` (its definition). A naive
   FD of `dW` through the quantized forward would be ~0 with spikes at the
   quantization steps — *not* the STE gradient, which is exactly why it is a pin,
   not an FD check (mirroring the existing exact-equality pins — rope `dX`, posemb
   off-path). `test_model_ternary` FD-checks the smooth parameters (embeddings,
   final LN, biases) end-to-end through the quantized stack.

5. **Checkpoint v8**: one `g_ternary` field at slot [23] (with `objective` at [22],
   both written; a v8 image is AR so objective == 0). A ternary model writes v8;
   non-ternary AR still writes v5, a hybrid v6, diffusion v7 — all byte-identical.
   v≤7 synthesize `ternary = 0`. Code `-48` rejects a hostile `ternary ∉ {0,1}`,
   `-49` a ternary image paired with mla/lin/ssm/rope/MoE/diffusion. The **weight
   blob is unaffected** — the master weights are serialized as f64 verbatim.

## Consequences

- **Positive**: a grad-checked ternary training mode on the shared trunk, zero
  regression to the f64 path (byte-identical default + preset + BPE runs verified),
  full checkpoint back-compat, and an honest ternary-vs-f64 accuracy axis (X-series).
  The fake-quant-into-scratch design needed no change to the vendored kernel and no
  new hand-derived gradient.
- **Negative**: increment 1 reuses the f64 `linear_fwd`, so it does **not yet**
  realize the integer-add speedup — it re-quantizes `W` each matmul (redundant f64
  work) purely for correctness. Ternary lowers capacity, so at reference scale
  bits/byte is worse than f64 (expected; the deliverable is "it learns + is
  grad-checked", not a win at this scale). One extra scratch allocation (`C·F`)
  under ternary.
- **Neutral**: the **i64-add ternary matmul kernel** (`x·W_eff = γ·(x·t)`, the
  multiply collapsing to add/subtract/skip) shipped as **increment 2 (v1.6.1, X023)** —
  two reference kernels `ternary_matmul_fwd`/`ternary_matmul_dx` in `ops.cyr`,
  grad-checked (the forward + `dx` pinned against the SIMD-f64 `W_eff` path at maxrel 0,
  `dx` FD'd) and **benched head-to-head**. The honest result: on x86_64 the collapse is
  **~3× slower** (matmul) / **~2.4× slower** (end-to-end) than the SIMD-f64 path, because
  `f64v_fmadd` retires 4 fused multiply-adds per instruction while the collapse is scalar
  add/subtract/skip — the wide-SIMD f64 multiply is already cheaper per element than a
  scalar add. The integer-add advantage needs **activation quantization** (int8 absmax
  acts → a literal integer matmul; the heavier follow-on scoped out below) and/or hardware
  without wide FMA; the orthogonal **memory** win (1.58 bits/weight) is real and unmeasured
  by this kernel. So the default ternary forward **keeps the SIMD-f64 path** and the
  collapse ships as the grad-checked + benched reference kernel (wired into no run; ternary
  runs stay byte-identical to 1.6.0). MLA/SSM/lin/MoE/RoPE/diffusion ternary remain clean
  additive fast-follows.

## Alternatives considered

- **Quantize inside `linear_fwd`/`linear_bwd`** — rejected: `lib/rosnet.cyr` is
  vendored and must not be modified; the attn11-local wrapper is the sovereign path.
- **FD-checking the STE `dW` against the quantized forward** — rejected as
  meaningless: the quantizer is piecewise constant, so that FD is ~0 a.e. with
  boundary spikes. The STE `dW` is pinned by its definition (pass-through) instead.
- **Quantizing the activations too (full BitNet b1.58 absmax int8 acts)** —
  rejected for v1.6.0: attn11 keeps activations f64 (the reference is about the
  *weight* precision endpoint); activation quant is a separate, heavier follow-on.
- **Threading `ternary` through every `model_init_*` signature** — rejected: it
  would touch ~22 call sites across the test/bench/fuzz harnesses; instead
  `g_ternary` is a global set by `main`/the loader before construction (the
  `g_data_w` precedent), reset on every load, with a `model_init_full` backstop.
- **Quantizing the LM head / embeddings** — rejected: BitNet keeps the embedding
  and output projection full precision; the head is weight-tied to `tokemb` here, so
  quantizing it would also corrupt the embedding gradient.
- **Per-output-channel scales (γ per row)** — rejected for the reference: b1.58
  uses a single per-tensor absmean; per-channel is a refinement that adds a scale
  vector and bookkeeping for a marginal reference gain.
