# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.5.0** — portability & robustness (roadmap M4). aarch64 cross-build + qemu
validation (grad checks pass; 250-step run matches x86 to display precision),
NaN/inf training guard, soak/leak test, crash-atomic checkpoint save. Built
2026-06-08.
(0.4.0: 4-wide SIMD matmul, ~2.27× faster. 0.3.0: corpus loading, checkpoints +
deterministic resume. 0.2.0: stacked layers, grad clipping, LR schedule.)

## Toolchain

- **Cyrius pin**: `6.1.5` (in `cyrius.cyml [package].cyrius`)

## Performance

4-wide SIMD (`f64v_fmadd`) matmul. Default config, x86_64:
fwd+bwd step 8.25ms → 3.64ms, **tokens/sec 1939 → 4396 (2.27×)**. See
[`benchmarks.md`](benchmarks.md) + [`../../bench-history.csv`](../../bench-history.csv).

## What works

End-to-end, on Linux x86_64 **and aarch64** (cross-build + qemu; grad checks
pass on both, training matches to display precision):

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
- **NaN/inf training guard** (stops cleanly instead of poisoning weights)
- Autoregressive generation (greedy + temperature sampling)
- **Corpus from file/stdin** (`--corpus`/`--stdin`): `O_NOFOLLOW`, `fstat`
  size-cap, byte-level adaptive vocab
- **Checkpoints** (`--save`/`--load`): validated header (all checked before
  allocation) + bit-for-bit **deterministic resume**; **crash-atomic save**
  (temp + fsync + rename, prior checkpoint preserved on failure)
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

- `tests/attn11.tcyr` — **52 checks**: finite-difference gradient checks (every
  op incl. dropout, attention at head dims 6/8/10 incl. biases, 2-layer full
  model), the SIMD bit-contract, resume-determinism (dropout off/on), checkpoint
  rejection smokes, a **soak/leak** test (`alloc_used` constant) and a **NaN
  guard** test. All pass on x86_64 AND aarch64 (`cyrius test`; aarch64 via qemu).
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

See [`roadmap.md`](roadmap.md). M4 (0.5.0) complete: aarch64 validation, NaN
guard, soak test, atomic checkpoint save. Next is **M5 (0.6.0): AGNOS kernel
port** (`--agnos`) — de-Linux the raw `syscall(...)` sites (exit/write/fsync/
rename) behind portable stdlib wrappers. Optional BPE tokenizer still deferred.
