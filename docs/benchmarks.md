# attn11 benchmarks

Hot-path timings for the default model (V=25, C=32, F=128, T=16, nh=4, NL=3;
39 488 params), x86_64 Linux. Reproduce with `cyrius build tests/attn11.bcyr
build/bench && ./build/bench`.

## Method

`tests/attn11.bcyr` times each op with `now_ns()` (a `clock_gettime`
CLOCK_MONOTONIC_RAW syscall) deltas around N-iteration loops (N = 200–2000), so
the ~120 ns/call clock overhead amortizes to nothing. Numbers are stable to
within a few percent across runs.

## Where the time goes (baseline)

A forward pass is **matmul-dominated**: the MLP's two linears are ~49%,
attention's Q/K/V/O projections ~29%, the tied LM head ~1.5% — together ~80% is
`linear_fwd`. GELU's `f64_tanh` is the largest non-matmul cost (~16%); LayerNorm
and softmax are negligible. Backward is ~1.9× forward and even more
matmul-bound (`linear_bwd` computes both `dx` and `dW`). Per scalar FMA the
software-float floor is ~3.6 ns (an `f64_mul` + `f64_add` builtin pair).

## SIMD vectorization (0.4.0)

The matmul inner loops were vectorized 4-wide over the contiguous output
dimension using the packed builtin `f64v_fmadd` (memory accumulators; a scalar
tail handles dimensions not a multiple of 4). On this toolchain `f64v_fmadd`
lowers to `mulpd`+`addpd` — it rounds identically to scalar `f64_mul`+`f64_add`
(verified 100 000/100 000 random cases, pinned by `test_simd_contract`).

Numerics, precisely:

- **AXPY paths** — forward `y`, `dW`, and the attention AV/`dQ`/`dK`/`dV`
  accumulations preserve the exact scalar reduction order, so they are
  **bit-identical** to the scalar build (`linear_fwd` matches a scalar reference
  with 0 bit-differences across 5 shapes, including non-multiple-of-4 tails).
- **DOT paths** — `dx` and the attention scores/`dP` use 4-lane partial sums +
  a tree reduction, so their summation *order* differs from a sequential scalar
  dot; results match only within floating-point rounding (~1e-16, far inside the
  grad-check tolerance).

Either way, **gradients are unchanged** (the grad-check suite passes for head
dims 6/8/10 and the production sizes) and training converges identically. GELU
(`f64_tanh`) and Adam are not vectorized (no packed transcendental; Adam is <1%
of a step).

| op | 0.3.0 scalar (ns) | 0.4.0 SIMD-4 (ns) | speedup |
|----|-------------------|-------------------|---------|
| `linear_fwd` T×C×F   | 242 719   | 62 517    | **3.88×** |
| `model_forward`      | 2 829 841 | 1 355 030 | 2.09× |
| `model_backward`     | 5 387 111 | 2 230 614 | 2.42× |
| fwd+bwd step         | 8 251 254 | 3 639 374 | 2.27× |
| `adam_step` (per opt step) | 773 842 | 776 688 | — (not vectorized) |
| **tokens/sec** (b=16) | **1 939** | **4 396** | **2.27×** |

A default 2000-step run drops from ~262 s of model compute to ~115 s.

## KV-cached generation + GQA (0.7.0)

Generation was the remaining O(T·rows) path: the 0.6.0 sampler recomputed the
full window for every token. 0.7.0 adds per-layer K/V caches and a single-row
cached forward (bit-identical to the uncached reference — see
`docs/architecture/003`); a full window context-shifts (drop oldest T/2,
re-prime) instead of sliding per token (ADR 0005). Greedy decode, 500 tokens,
default config:

| path | ns/token | tokens/sec |
|------|----------|------------|
| uncached (window forward per token) | 1 050 579 | 951 |
| **KV-cached** (row per token + amortized re-prime) | **170 392** | **5 868 (6.16×)** |

GQA (`n_kv_heads`) makes the cache size a knob — and trims the K/V projection
compute (default config; MQA = `nkv 1`):

| nkv | KV cache bytes | fwd+bwd step (ns) | gen cached (ns/tok) |
|-----|----------------|-------------------|---------------------|
| 4 (MHA, default) | 24 576 | 3 680 703 | 170 392 |
| 2 (GQA)          | 12 288 | —         | —       |
| 1 (MQA)          |  6 144 | 3 456 801 | 164 282 |

At ctx 16 the cached path is re-prime-bound (a window recompute every 8
tokens), so the speedup grows with context length — the E3 ctx-64 preset is
the natural next measurement. Training numbers are unchanged from 0.6.0
within noise (fwd+bwd ~3.7 ms, ~4 350 tok/s b=16; pin 6.1.33).

History is tracked in [`bench-history.csv`](../bench-history.csv).

## Next perf levers (future)

- Vectorize the tied LM head (matters as vocab grows with larger corpora).
- A packed `tanh` approximation for GELU (its ~16% share grows now that matmul
  is faster).
- Cache-blocking / register-tiling the matmul for larger `d_model`.
- A batched prefill (a window forward that also fills the K/V caches, instead
  of `keep` single-row calls) if the context-shift re-prime cost starts to
  matter at larger T.
