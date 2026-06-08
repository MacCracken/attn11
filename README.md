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

- **Tokenizer** — char-level over a fixed embedded corpus
- **Embeddings** — token embedding + learned positional embedding
- **Block** — pre-norm `LayerNorm → causal multi-head self-attention → residual
  → LayerNorm → MLP (GELU) → residual`
- **Head** — final `LayerNorm` + weight-tied LM head → softmax cross-entropy
- **Training** — hand-written backprop + **Adam**; loss printed as it descends
- **Generation** — autoregressive sampling from the trained weights

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
cyrius test                               # grad checks + smoke tests
```

## Model size

The default config is intentionally tiny so it trains on CPU in seconds — see
[`docs/development/state.md`](docs/development/state.md) for the live
hyperparameters (vocab, `d_model`, context, heads, layers).

## License

GPL-3.0-only
