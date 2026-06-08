# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.1.0** — first working model. A one-block GPT-style transformer trains
end-to-end and generates text. Built 2026-06-08.

## Toolchain

- **Cyrius pin**: `6.1.3` (in `cyrius.cyml [package].cyrius`)

## What works

End-to-end, on x86_64 Linux:

- Char-level tokenizer over an embedded corpus
- Token + learned positional embeddings
- One pre-norm Transformer block:
  `LayerNorm → causal multi-head self-attention → residual → LayerNorm → MLP (GELU) → residual`
- Final `LayerNorm` + weight-tied LM head → softmax cross-entropy
- Hand-written backprop through every op (verified — see Tests)
- **Adam** optimizer (flattened parameter vector, bias-corrected)
- Mini-batch grad accumulation + training loop with loss logging
- Autoregressive generation (greedy + temperature sampling)

Default run (`./build/attn11`, ~90 s): loss `~3.2 → ~0.15` over 2000 steps;
sampled output reproduces real corpus phrases.

## Default hyperparameters (`src/main.cyr`)

| name        | value | note                         |
|-------------|-------|------------------------------|
| vocab `V`   | 25    | unique chars in the corpus   |
| `d_model` C | 32    |                              |
| context `T` | 16    |                              |
| heads `nh`  | 4     | head dim = C/nh = 8          |
| MLP `F`     | 128   | = 4·C                        |
| params      | 13952 | total trainable f64          |
| optimizer   | Adam  | lr 3e-3, β 0.9/0.999         |
| steps/batch | 2000 / 16 |                          |

## Source (`src/`, ~1200 LOC)

- `tensor.cyr` — f64-array helpers, deterministic PRNG (xorshift64 + splitmix
  seeding, Marsaglia-polar normal), float printing
- `ops.cyr` — linear, layernorm, GELU (tanh approx), softmax cross-entropy
  (forward + backward)
- `attn.cyr` — causal multi-head self-attention (forward + backward), one
  pre-allocated arena for caches + temporaries
- `model.cyr` — packed parameter vector, embeddings, tied head, full
  forward/backward wiring, Adam
- `train.cyr` — corpus, tokenizer, batch sampling, training loop, generation
- `main.cyr` — config + orchestration

## Tests

- `tests/attn11.tcyr` — **16 finite-difference gradient checks**: every op
  (linear, layernorm, GELU, softmax-xent), attention, and the full model.
  All pass (`cyrius test`).

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math,
  matrix, random

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). Natural extensions: multiple stacked blocks,
attention/MLP biases, dropout, learning-rate schedule, larger corpus / BPE,
checkpoint save/load, aarch64 validation (f64_exp/ln use polyfills there).
