# 009 — GPU layernorm (bit-exact, 3-pass tiled) and the transcendental wall

*What's true about `gpu_ln_fwd` (M18, v1.8.1) and why the rest of the forward isn't on
GPU yet. Builds on [008](008-gpu-matmul-spirv.md) (the matmul + the generated-SPIR-V /
host-tiling / native-dispatch mechanics, all reused here).*

## Which forward ops can be BIT-EXACT on the native f64 GPU, and which cannot

The 1.8.0 invariant is precious: a `--gpu` run is **byte-identical** to the no-flag run
(GPU is an execution target; the CPU stays the oracle). Preserving it means each GPU op
must match its CPU reference *bit-for-bit*. Whether that's possible per op:

| op | CPU reduction order | needs transcendental? | bit-exact on GPU? |
|----|--------------------|----------------------|-------------------|
| matmul (`linear_fwd`) | sequential AXPY per output | no | **yes** (1.8.0) |
| **`ln_fwd`** | **sequential** (`mu+=x`, `v+=(x-mu)²`) | no (sqrt/div only) | **yes** (1.8.1) |
| `head_fwd`, attn QK dots | SIMD 4-lane partials + **tree** sum | no | no — order differs |
| softmax (attn, masked-CE) | sequential | **`f64_exp`** | no — see below |
| GELU | elementwise | **`f64_tanh`** = exp-based | no — see below |

`ln_fwd` is the clean one: its reductions are **sequential** (a plain `+=` loop over C),
so a tiled sequential-sum GPU kernel reproduces the exact accumulation order, and it uses
only `sqrt`/`div`/`mul`/`add`/`sub` — all bit-exact on mabda's native path (X025 proved the
sqrt/div core). So `gpu_ln_fwd` is byte-identical to `ln_fwd`, and the `--gpu` checkpoint
stays identical.

## The transcendental wall (why softmax + GELU are NOT on GPU)

`f64_exp` is an **x86 hardware builtin** — `lib/math.cyr` carries only an *aarch64
polyfill* (`_f64_exp_polyfill`), and `f64_tanh` = `ganita_f64_tanh` = `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)`
is built on it. Two independent blockers follow:

1. **No bit-exact match.** Reproducing the x86 `exp` builtin bit-for-bit in SPIR-V is
   infeasible (it's hardware, ~80-bit internal). An in-shader *polynomial* exp would match
   only within a tolerance — so a softmax/GELU GPU op would **break the byte-identical
   invariant** (the forward would drift at every exp, accumulating).
2. **No device transcendentals.** mabda's native f64 path has no `exp`/`log` (spirv-val
   rejects them on `double`) — they must be hand-rolled from the proven primitives
   (add/mul/fma/div/sqrt/ldexp) regardless.

So softmax + GELU are a **separately-gated** future increment: an in-shader polynomial exp
validated within a *tolerance* (the dual-precision discipline ADR 0016 anticipated — except
the tolerance is f64-tight, not f32), accepting that those ops make the `--gpu` run
*tolerance-close*, not byte-identical. `head_fwd` and the attention QK dots are bit-exact in
principle but need a kernel that replicates the CPU's **4-lane-partial + tree** reduction
order (`f64v_fmadd` into `acc[4]` then a horizontal tree-sum) — not the sequential order the
matmul/ln kernels use. All three are deferred; 1.8.1 ships only the op that keeps the
invariant.

## ln_fwd is 3-pass host-tiled (the 256-id cap forbids a per-row unroll)

A per-row layernorm unrolled over C costs ~8·C ids (load + sum + sub + square + sum +
normalize); even C=32 (≈256) breaches mabda's public-compile cap (`NATIVE_SHADER_CAP_IDS
= 256`, see [008](008-gpu-matmul-spirv.md)). So the contraction is **host-tiled** like the
matmul, in three passes with per-row scalar buffers:

1. **pass 1 — tiled Σx → `S[m]`** (`_gpu_build_mean`, bindings x, S): one thread per row,
   `S[m] += Σ_{j<GPU_TK} x[m·C + c0 + j]` (RMW). Host loops `c0` in `GPU_TK` steps. Then
   **host** `mean[m] = S[m]·invC` (in place in S; also written to the caller's `mean[]`).
2. **pass 2 — tiled Σ(x−mu)² → `V[m]`** (`_gpu_build_var`, bindings x, mean, V): reads
   `mean[m]`, `V[m] += Σ (x−mean)²` (RMW). Then **host** `rstd[m] = 1/√(V[m]·invC + eps)`
   (in place in V; written to `rstd[]`).
3. **pass 3 — elementwise normalize → y** (`_gpu_build_norm`, bindings x, mean, rstd,
   gamma, beta, y): one thread per element (grid M·C), `m = idx/C`, `c = idx%C`,
   `y[idx] = (x[idx]−mean[m])·rstd[m]·gamma[c] + beta[c]`.

Dispatches are serialized, so pass *k* sees pass *k−1*'s writes (no barrier needed). The
per-row scalars are computed **host-side** with the exact ops `ln_fwd` uses (`*invC`,
`1/√(·+eps)`), so the result is bit-exact end to end. `mean[m]`/`rstd[m]` are written for
the CPU backward (which 1.8.1 leaves on CPU), so a `--gpu` forward + CPU backward is a fully
consistent step. `C` must be a multiple of `GPU_TK` (16) — default 32 / preset 64 qualify;
other C cleanly falls back to CPU.

## Buffers and kernel reuse

The persistent GTT buffer set grew 3 → **7** (matmul's x/W/y plus gamma, beta, and the two
per-row scalar buffers S/MU and V/RS), via a clean `_gpu_alloc1` + `_gpu_teardown` pair that
replaced the matmul-only allocation ladder (matmul re-validated — no 1.8.0 regression). ln
reuses matmul's `x` (input) and `y` (output) buffers (ops never run concurrently). The three
ln kernels share `_gpu_pre` (the variable-binding-count preamble) and the same `_gpu_op*`
emitters and `(kind,C,c0)`-keyed shader cache as the matmul (kind tag in the high bits keeps
ln keys from colliding with matmul tile keys). Validation: `tests/gpu_ln.cyr`
(`make gpu-test`).
