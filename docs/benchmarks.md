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
tokens), so the speedup grows with context length — measured next.

## The `--preset` config + BPE training cost (0.7.1)

The M7 `--preset` (ctx 64 / d_model 64 / 8 heads / 4 layers; 205 760 params
at the embedded corpus) confirms the prediction above: at T=64 the
context-shift re-prime amortizes over 32 tokens instead of 8, and the
uncached reference simultaneously gets more expensive, so the cached
advantage widens from 6.2× to **23×** (greedy, 200 tokens, x86_64, pin
6.1.34):

| config | path | ns/token | tokens/sec |
|--------|------|----------|------------|
| default (ctx 16) | uncached | 1 068 001 | 936 |
| default (ctx 16) | KV-cached | 174 689 | 5 724 (**6.1×**) |
| preset (ctx 64) | uncached | 15 564 530 | 64 |
| preset (ctx 64) | KV-cached | **672 747** | **1 486 (23.1×)** |

Preset training: fwd+bwd step 65.0 ms → ~984 tok/s (b=16) — ~17.6× the
default config's step cost for 5.2× the params and 4× the context (the
attention T²-terms and the wider matmuls compound).

BPE merge training (`--bpe K`, 0.7.1) is a one-shot pre-training cost:
**~110 ms** for a 256 KB corpus at K=128 (`bpe_learn 256KB K=128` in the
bench) — negligible against minutes of training. Default-config training
numbers are unchanged from 0.6.0 within noise (fwd+bwd ~3.7 ms, ~4 350
tok/s b=16).

History is tracked in [`bench-history.csv`](../bench-history.csv).

## MLA latent KV-cache decode (1.2.1, M12.2)

MLA (`--attn-kind mla`, ADR 0007) stores ONE low-rank latent `c` (`d_c` per
token per layer) as the persistent decode cache instead of full per-head K/V,
up-projecting to K/V on read — bit-identical to the uncached reference
(`test_kv_mla`, both arches). The compression is the headline; at the default
config (NL=3, T=16, C=32, hd=8), d_c = 16 (= C/2):

| cache kind            | total bytes | vs MHA |
|-----------------------|-------------|--------|
| MHA full K/V (nkv=4)  | 24 576      | 1.0×   |
| GQA (nkv=2)           | 12 288      | 2.0×   |
| MQA full K/V (nkv=1)  | 6 144       | 4.0×   |
| **MLA latent** (d_c=16) | **6 144** | **4.0×** |

MLA reaches MQA's footprint (4× under MHA) but keeps **full heads** — d_c is the
single knob (d_c = 8 → 3 072, 8× under MHA). Greedy decode, default config:

| path | ns/token | tokens/sec |
|------|----------|------------|
| uncached MLA (window forward per token) | ~951 700 | ~1 050 |
| **latent-cache** (row per token + amortized re-prime) | **~206 100** | **~4 852 (~4.6×)** |

The ~4.6× is from not recomputing the window per token; it is **not** at
MHA-cached parity because the reference re-up-projects the cached latents to K/V
each step. Absorbing `W_UK` into `W_Q` (attend latents directly) is the compute
optimization (and a further memory win) — future work, since it reorders the
accumulation and needs its own bit-identity story. Full trail:
[experiments.md X006](development/experiments.md).

## Coupled RoPE overhead (1.2.2, M12 increment 4)

`--pos-kind rope` rotates Q/K by absolute position inside every attention. The
rotation is parameter-free but adds, per dimension-pair, a Maclaurin `cos`/`sin`
of the base angle plus a binary-exponentiation to the position (the portable
trig path — `f64_sin`/`f64_cos` are x86-only; see
[architecture/005](architecture/005-rope-portable-trig.md)). Default config,
vs the learned-abs baseline:

| path                     | learned-abs | rope     | overhead |
|--------------------------|-------------|----------|----------|
| fwd+bwd step             | ~3.53 ms    | ~3.61 ms | **+2.3%** |
| gen cached (ns/token)    | ~163 000    | ~179 000 | **+10%**  |

The training step barely moves (~2%; attention's rotation is small against the
matmuls); cached decode pays ~10%/token. The cached-vs-uncached bit-identity
contract is unaffected (`test_kv_rope`). Full trail:
[experiments.md X007](development/experiments.md).

## SIMD tied LM head (0.8.0 → 0.8.1, M9 lever 1)

`head_fwd_row` (the weight-tied output projection, `logits(V) = f_row(C) @
tokemb^T`) was a scalar dot product while the linear layers had been 4-wide
since 0.4.0. It is `O(V·C)` per row and runs in **every** training forward and
**every** generated token, so its cost grows with the vocabulary — negligible
at the V=25 default, but ~17% of the forward at a BPE-scale V=768. Vectorized
with the same `f64v_fmadd` 4-lane accumulator + scalar tail as the matmul
(shared by the training and cached-generation paths, so the bit-identity gate
holds):

| `head_fwd` (V=768, C=64, T=64) | ns/op | speedup |
|--------------------------------|-------|---------|
| scalar (0.8.0)                 | ~9 700 000 | — |
| 4-wide SIMD (0.8.1)            | **~3 590 000** | **2.7×** |

Default-config training is unchanged within noise (the V=25 head is tiny);
the win lands on BPE/large-vocab and generation throughput. Grad checks, the
KV bit-identity gate, and resume-determinism stay green on x86_64 and aarch64;
a new `head SIMD == scalar dot (C=6 tail)` test exercises the `C % 4 ≠ 0` tail
that no other config hits (mutation-verified to catch a dropped tail).

## M9 perf levers — outcome (concluded at 0.8.1)

One lever per release, each measured before shipping (full trail in
[experiments.md X004](development/experiments.md)):

- ~~Vectorize the tied LM head~~ ✅ **0.8.1** (2.7× at V=768; above) — the one
  win.
- ❌ Packed `tanh` for GELU — **rejected**: the exact one-exp form is only
  ~15% faster per call (noisy) and GELU is ~8% of a step, so ~1–2% — below
  step-bench noise (`f64_exp` is cheap on this toolchain).
- ❌ Matmul cache-blocking / register-tiling — **rejected**: an m-blocked
  `linear_fwd` is bit-identical but ~15% *slower* at the preset shape; attn11's
  weight matrices are cache-resident, so blocking only adds accumulator traffic.
- ❌ Batched prefill — **rejected**: a batched window forward vs `keep`
  single-row re-prime benched ~1% (within noise); the re-prime is irreducible
  work, not call overhead.

The residual matmul gap to SIMD peak (~10–15%) is structural — the
memory-accumulator pattern forced by the never-reassign-a-SIMD-var rule
([architecture/001](architecture/001-tensors-and-floats.md)) and the 2-wide
`f64v_fmadd` lowering — and would need toolchain support (true AVX/FMA
builtins), not a v0.8.x code change.
