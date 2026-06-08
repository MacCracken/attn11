# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.4.0** — performance (roadmap M3). 4-wide SIMD matmul (`f64v_fmadd`) on the
linear + attention hot paths: **~2.27× faster training** (1939 → 4396 tok/s),
gradients/convergence unchanged. Built 2026-06-08.
(0.3.0: file/stdin corpus, checkpoints + deterministic resume, fuzzing.
0.2.0: stacked layers, residual-init scaling, grad clipping, LR schedule.)

## Toolchain

- **Cyrius pin**: `6.1.5` (in `cyrius.cyml [package].cyrius`)

## Performance

4-wide SIMD (`f64v_fmadd`) matmul. Default config, x86_64:
fwd+bwd step 8.25ms → 3.64ms, **tokens/sec 1939 → 4396 (2.27×)**. See
[`benchmarks.md`](benchmarks.md) + [`../../bench-history.csv`](../../bench-history.csv).

## What works

End-to-end, on x86_64 Linux:

- Char-level tokenizer over an embedded corpus
- Token + learned positional embeddings
- **`n_layers` stacked** pre-norm Transformer blocks, each:
  `LayerNorm → causal multi-head self-attention → residual → LayerNorm → MLP (GELU) → residual`
- Final `LayerNorm` + weight-tied LM head → softmax cross-entropy
- Hand-written backprop through every op and the full residual stack (verified)
- **Adam** + **global-norm gradient clipping** + **LR warmup→cosine** schedule
- GPT-2 residual-projection init scaling (`1/sqrt(2·n_layers)`)
- Config-gated **attention biases** and **residual dropout** (dropout
  auto-disabled in eval/generation)
- Mini-batch grad accumulation; training logs loss / lr / grad-norm
- Autoregressive generation (greedy + temperature sampling)
- **Corpus from file/stdin** (`--corpus`/`--stdin`): `O_NOFOLLOW`, `fstat`
  size-cap, byte-level adaptive vocab
- **Checkpoints** (`--save`/`--load`): validated header (magic/version/config/
  size, all checked before allocation) + bit-for-bit **deterministic resume**
  (weights, Adam moments, step, PRNG state)
- CLI: `--corpus --stdin --load --save --steps --gen-only`

Default run (`./build/attn11`, 3 layers): loss `~3.2 → ~0.13` over 2000 steps;
sampled output reproduces real corpus phrases.

## Default hyperparameters (`src/main.cyr`)

| name        | value | note                              |
|-------------|-------|-----------------------------------|
| vocab `V`   | 25    | unique chars in the corpus        |
| `d_model` C | 32    |                                   |
| context `T` | 16    |                                   |
| heads `nh`  | 4     | head dim = C/nh = 8               |
| layers `NL` | 3     | stacked pre-norm blocks           |
| MLP `F`     | 128   | = 4·C                             |
| attn bias   | on    | Q/K/V/O biases (config-gated)     |
| dropout     | 0.0   | residual dropout (config-gated)   |
| params      | 39488 | total trainable f64 (3 layers, biases) |
| optimizer   | Adam  | β 0.9/0.999, global-norm clip 1.0 |
| lr schedule | warmup 100 → cosine | base 3e-3 → min 3e-4  |
| steps/batch | 2000 / 16 |                               |

## Source (`src/`, ~1200 LOC)

- `tensor.cyr` — f64-array helpers, deterministic PRNG (xorshift64 + splitmix
  seeding, Marsaglia-polar normal), float printing
- `ops.cyr` — linear, layernorm, GELU (tanh approx), softmax cross-entropy
  (forward + backward)
- `attn.cyr` — causal multi-head self-attention (forward + backward), one
  pre-allocated arena for caches + temporaries
- `fileio.cyr` — secure file I/O (`O_NOFOLLOW`, `fstat` size, looped read/write),
  stdin reader
- `model.cyr` — per-layer packed parameters (block stride + `_o_*`/`PL`/`GL`
  helpers), per-layer activation caches (residual-stream array), embeddings,
  tied head, full N-layer forward/backward, grad clipping, Adam
- `train.cyr` — byte-level tokenizer, corpus (embedded/file/stdin), batch
  sampling, LR schedule, resumable training loop, generation
- `persist.cyr` — validated checkpoint serialize/load (in-memory + file)
- `main.cyr` — CLI arg parsing + orchestration

## Tests

- `tests/attn11.tcyr` — **47 checks**: finite-difference gradient checks (every
  op incl. dropout, attention at head dims 6/8/10 incl. biases, 2-layer full
  model), the `f64v_fmadd`==scalar SIMD bit-contract, **bit-for-bit
  resume-determinism** (dropout off AND on), and checkpoint rejection smokes
  (truncated / bad magic / absurd config / NaN+1.0 dropout / vocab-mismatch).
  All pass (`cyrius test`).
- `tests/attn11.bcyr` — benchmark harness (timings + tokens/sec).
- `tests/attn11.fcyr` — fuzz harness: 500 mutated-checkpoint rounds + 100 random
  corpora; loaders reject malformed input without crashing.
- The M2 (persistence) and M3 (SIMD) code each passed an adversarial multi-agent
  review; all confirmed findings fixed and regression-tested. See CHANGELOG.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math,
  matrix, random, args

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). M2 (0.3.0) is complete: file/stdin corpus,
byte-level tokenizer, validated checkpoints + deterministic resume, loader
fuzzing. Next is M3 (0.4.0): SIMD hot paths + benchmarks (`benchmarks.md`).
Optional BPE tokenizer remains deferred (byte-level covers the M2 gate).
