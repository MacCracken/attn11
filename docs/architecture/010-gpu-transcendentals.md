# 010 — GPU transcendentals: the in-shader f64 `exp`, GELU, and the tolerance gate

*What's true about `gpu_gelu_fwd` + `_gpu_emit_exp` (M18, v1.8.2). Builds on
[008](008-gpu-matmul-spirv.md) (SPIR-V/dispatch mechanics) and
[009](009-gpu-layernorm-and-the-transcendental-wall.md) (the wall this crosses).*

## Why this is the first NON-bit-exact GPU op

f64 `exp` is an **x86 hardware builtin** (`lib/math.cyr` carries only an aarch64
`_f64_exp_polyfill`), and mabda's native path has no f64 transcendentals (spirv-val rejects
`exp`/`log` on `double`). So an in-shader `exp` must be hand-rolled, and it **cannot** match
the x86 builtin bit-for-bit. GELU (`= 0.5·x·(1+tanh(inner))`, `tanh` built on `exp`) and
softmax therefore can't preserve the byte-identical `--gpu` invariant that matmul (1.8.0) and
layernorm (1.8.1) hold.

**The gate split.** Transcendental ops ride a **separate flag**: `g_gpu_tc` / `--gpu-tc`
(implies `--gpu`). Plain `--gpu` stays matmul + layernorm only and remains byte-identical to
the no-flag run; `--gpu-tc` opts into the full-er forward at a *tolerance*. So the strong
bit-exact invariant is never silently lost — you choose it.

## The in-shader f64 exp (`_gpu_emit_exp`)

Transliterated from `_f64_exp_polyfill` (the aarch64 CPU path), built only from f64 primitives
mabda's gfx9 backend proves (X025): FMul, FAdd, FSub, ConvertFToS, and GLSL `Ldexp` (53).

1. **Range-reduce.** `y = x·log2(e)`; `n = round(y)` via the **magic-number trick**
   `(y + 1.5·2⁵²) − 1.5·2⁵²` (IEEE round-to-nearest-even as an f64 — no `Round` ext-inst,
   which mabda doesn't lower); `r = x − n·ln2`, so `|r| ≤ ln2/2 ≈ 0.347`.
2. **Polynomial.** 11-term Taylor for `exp(r)` via Horner (`p = c_k + r·p`, `c_k = 1/k!`),
   the exact coefficient bit-patterns copied from the polyfill.
3. **Scale.** `n_i = ConvertFToS(n)` (exact — `n` is integral); `exp = ldexp(p, n_i) = p·2^{n_i}`.

Measured **max relative error ≈ 2.3e-13** vs the CPU `f64_exp` over `[−16, 0]` (the softmax
range; `x ≤ 0`). The emitter uses fixed const ids (LOG2E, LN2, MAGIC, c10..c0) and is emitted
**inline** wherever exp is needed (gfx9_compile is single-function — no `OpFunctionCall`).

## GELU

`inner = c·(x + a·x³)` (`c = √(2/π)`, `a = 0.044715`, computed host-side to the exact bits of
`_gelu_c`/`_gelu_a` and baked as constants), then `tanh(inner) = (e^{inner} − e^{−inner}) /
(e^{inner} + e^{−inner})` (exp emitted twice, then FSub/FAdd/**FDiv** — FDiv proven X025),
then `0.5·x·(1+tanh)`. One elementwise kernel (one thread per element, grid = n; `gelu_fwd` is
flat n-element, not M×C). id_bound ≈ 110, well under the 256-id cap (no tiling needed). It
reuses the matmul/ln `x`/`y` GTT buffers. Measured **~3e-14 absolute** error vs `gelu_fwd`.

## The tolerance gate (`allclose`, not relative error)

GELU has zero-crossings (`gelu(x) ≈ 0` for `x` slightly negative). A **pure relative-error**
gate blows up there (`|ref|→0` while `|gpu−cpu|` stays ~1e-14), so `tests/gpu_gelu.cyr` uses
**`allclose`**: pass iff `|gpu − cpu| ≤ atol + rtol·|cpu|` for every element (`atol = rtol =
1e-10`). This is the dual-precision discipline ADR 0016 anticipated — except the tolerance is
f64-tight (~1e-13 headroom of ~1000×), not f32. Two checks back it: (1) per-element allclose
on default/preset/decode widths; (2) a **statistical** check — a `--gpu-tc` training run's loss
+ eval bits/byte track the CPU run to 5-decimal print precision and never NaN (the ~1e-13
per-op divergence is invisible at that precision).

## Scope

`--gpu-tc` adds GELU. Still CPU: **softmax** (attention scores + masked-CE — needs the exp
emitter *plus* max/sum reductions, like ln's tiling) and **`head_fwd`/QK dots** (bit-exact in
principle but a SIMD tree-reduction order). Those are 1.8.3. The exp emitter (`_gpu_emit_exp`)
is the reusable foundation softmax will build on. AGNOS: the whole GPU path (incl. these
transcendental kernels) is `#ifndef CYRIUS_TARGET_AGNOS`-guarded out; only the dead-code mabda
dep compiles there (via the `SYS_IOCTL` stub, see [008](008-gpu-matmul-spirv.md)).
