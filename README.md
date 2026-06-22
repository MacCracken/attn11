# attn11

> A from-scratch, dependency-free **GPT-style transformer — trained** — written in
> [Cyrius](https://github.com/MacCracken/cyrius), the AGNOS "assembly-up" systems
> language where *everything is i64* and floating point is IEEE-754 `f64`
> bit-patterns driven by `f64_*` builtins.

No BLAS. No libc. No autodiff framework. The forward pass, the backward pass
(hand-derived gradients), and the optimizer are all written by hand on raw
`f64` arrays. The whole thing compiles to a single static ELF.

## What it does

`attn11` trains a small char-level transformer to predict the next character in
an embedded corpus, then samples from it:

- **Tokenizer** — char-level by default (embedded or `--corpus`/`--stdin`), or
  opt-in **BPE** (`--bpe K`)
- **Embeddings** — token embedding + positions selectable via `--pos-kind`
  (learned-absolute / coupled RoPE / decoupled RoPE)
- **Block** — pre-norm `LayerNorm → causal sequence mixer → residual → LayerNorm
  → FFN → residual`, where the mixer is `--attn-kind {mha, mla (latent-KV), lin
  (gated-linear), ssm (selective-SSM)}` — or a per-layer **hybrid** (`--attn-every`)
  — and the FFN is a dense GELU MLP or a top-K **MoE** (`--experts N`)
- **Head** — final `LayerNorm` + weight-tied LM head → softmax cross-entropy
- **Objective** — autoregressive by default, or a masked **diffusion** LM
  (`--objective diffusion`: bidirectional, confidence-ordered parallel decode)
- **Precision** — full `f64`, or opt-in **ternary** `{−1, 0, +1}` weights
  (`--ternary`, BitNet-style fake-quant with a straight-through estimator)
- **Training** — hand-written backprop + **Adam**; loss printed as it descends
- **Generation** — autoregressive (or diffusion) sampling from the trained weights
- **Execution** — CPU by default (the reference oracle); opt-in **GPU** (`--gpu`, Linux / AMD GFX9)
  runs the f64 tensor ops on-device **bit-exact** via the **[rosnet](https://github.com/MacCracken/rosnet)**
  GPU backend layered on [mabda](https://github.com/MacCracken/mabda)'s native-AMD f64 SPIR-V — an
  execution *target*, not a different model (a `--gpu` checkpoint is byte-identical). See
  [`docs/guides/gpu.md`](docs/guides/gpu.md).

Correctness of the hand-derived gradients is gated by **finite-difference
gradient checks** (see `tests/`), the standard tool for verifying backprop.

## Why "everything is i64" matters here

Cyrius has no float type. An `f64` is its 64-bit IEEE-754 pattern carried in an
i64 register; arithmetic goes through builtins (`f64_add`, `f64_mul`, `f64_exp`,
`f64_tanh`, `f64_sqrt`, …). A "tensor" is just a heap pointer to a flat,
row-major run of `f64` values; shapes are tracked in code. See
[`docs/architecture/`](docs/architecture/) for the conventions.

## Build

```sh
cyrius deps                               # resolve stdlib deps
cyrius build src/main.cyr build/attn11    # compile to a static ELF
./build/attn11                            # train + sample
./build/attn11 --help                     # the full CLI
cyrius test                               # grad checks + smoke tests
```

See [`docs/guides/getting-started.md`](docs/guides/getting-started.md) for the
full CLI (`--corpus`, `--load`/`--save` checkpoints, `--preset`, `--bpe`, `--eval`,
`--eval-corpus` (held-out cross-corpus eval), the objective/precision axes
`--objective {ar,diffusion}` (with `--decode-steps`/`--decode-schedule`) and
`--ternary`, and the architecture axes `--attn-kind {mha,mla,lin,ssm}`, `--pos-kind
{learned,rope,rope-decoupled}`, `--experts N` (MoE), `--attn-every K` (per-layer
hybrid)) and [`docs/STABILITY.md`](docs/STABILITY.md) for the frozen surface.

## Model size

The default config is intentionally tiny so it trains on CPU in seconds — see
[`docs/development/state.md`](docs/development/state.md) for the live
hyperparameters (vocab, `d_model`, context, heads, layers).

## License

GPL-3.0-only
