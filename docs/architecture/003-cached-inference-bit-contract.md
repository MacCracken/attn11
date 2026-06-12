# 003 — The cached-inference bit contract

**What's true**: attn11 has two forward implementations, and they must produce
**bit-identical** logits for the same tokens at the same positions:

- the **batch path** — `model_forward` (training) / `model_eval_window`
  (generation reference): every op over all `n` window rows;
- the **row path** — `model_fwd_row` + an attention-kind-specific row function
  over exactly one new row: `attn_fwd_row` (MHA/GQA, full-K/V cache) or
  `attn_mla_fwd_row` (MLA, latent cache — 1.2.1), each calling the shared
  `attn_core_fwd_row`. Coupled RoPE (1.2.2) rides inside both, rotating the new
  row's Q/K by its absolute position (`docs/architecture/005`).

`test_kv_generation` pins the MHA/GQA contract; `test_kv_mla` and `test_kv_rope`
pin the MLA and RoPE variants — logits compared bit-for-bit at every prefix
length and across context-shifts. It holds for a non-obvious reason, and it
constrains every future kernel change (and every new `--attn-kind`/`--pos-kind`).

## Why bit-identity is even possible

In a causal pre-norm transformer, row `i` of every activation depends only on
rows `≤ i` of the input, and every op in the stack is **row-local** except
attention (which is causal):

- `linear_fwd` computes each output row independently — a 1-row call performs
  the *same multiply/add sequence* as row `i` of an n-row call;
- `ln_fwd`, `gelu_fwd`, the residual adds, and the tied head are per-row /
  per-element;
- attention's row `i` reads K/V rows `0..i` — which the cache holds, computed
  earlier from inputs that (by induction over rows) were bit-identical.

So equality is achievable — but only if both paths run the **same arithmetic
in the same order** per row.

## What the contract forbids

- **Diverging kernels.** `attn_fwd_row` mirrors `attn_fwd`'s row loop exactly:
  the same `hd4` 4-wide `f64v_fmadd` chunking with the same scalar tail, the
  same accumulator reset, the same tree reduction of the 4 lanes, the same
  softmax (max, exp, normalize) order, the same AV axpy order over `j`. If a
  future change re-blocks, re-orders, or fuses any of these in ONE path, the
  bit contract breaks. Change both together or not at all.
- **Asymmetric "harmless" extras.** The batch eval path skips dropout instead
  of applying an all-ones mask. That is bit-neutral only because `x · 1.0 ≡ x`
  bitwise for every finite f64 (and eval masks are exactly 1.0). Any future
  always-on per-element transform (scaling, clamping, noise) must appear in
  both paths.
- **Padding-dependent semantics.** The row path never sees other rows, so
  nothing may depend on `n` except attention's `j ≤ pos` bound. (This is why
  0.7.0 removed the old left-padding sampler — pad rows would have to be
  cached too.)

## Practical notes

- `f64v_fmadd` lowers to `mulpd`+`addpd` on x86_64 (bit-identical to scalar
  mul/add — pinned by `test_simd_contract`) and to fused `fmla` on aarch64.
  Both paths use the same builtin in the same pattern, so the contract holds
  on **both** architectures even though aarch64 differs from x86_64.
- The contract is *per row at a given position*. A context-shift moves tokens
  to new positions, so their K/V must be **recomputed** (re-prime), never
  copied/shifted — see ADR 0005.
- When adding a kernel (new mixer, new activation): write the batch op first,
  grad-check it, then derive the row op by *deleting the row loop*, nothing
  else. If you can't express the row op that way (e.g. cross-row
  normalization), it cannot be KV-cached bit-identically — flag it in the
  design instead of approximating silently.
