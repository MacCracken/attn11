# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.6.0** — AGNOS kernel port (M5): trains + checkpoints + samples as a ring-3
app under the booted AGNOS kernel, **bit-for-bit identical** checkpoint vs the
Linux run (fixed seed, CPU held constant); one source tree for Linux x86_64 /
aarch64 / AGNOS; toolchain pin 6.1.6 → 6.1.31; `scripts/agnos-smoke.sh` run
gate; statement-call entry epilogues (cycc argv-capture workaround,
`docs/architecture/002`). (0.5.1 — standards conformance. 0.5.0 — aarch64
validation, NaN/inf guard, soak test, crash-atomic save. 0.4.0: 4-wide SIMD
matmul, ~2.27× faster. 0.3.0: corpus loading, checkpoints + deterministic
resume. 0.2.0: stacked layers, grad clipping, LR schedule.)

## Toolchain

- **Cyrius pin**: `6.1.31` (in `cyrius.cyml [package].cyrius`) — bumped from
  `6.1.6` during M5: the old pin predated the 6.1.13/6.1.14 HIGH-sev agnos
  codegen fixes (indirect calls / argc-argv on the agnos target)

## Performance

4-wide SIMD (`f64v_fmadd`) matmul. Default config, x86_64:
fwd+bwd step 8.25ms → 3.64ms, **tokens/sec 1939 → 4396 (2.27×)**. See
[`benchmarks.md`](../benchmarks.md) + [`../../bench-history.csv`](../../bench-history.csv).

## What works

End-to-end, on Linux x86_64, **aarch64** (cross-build + qemu; grad checks pass
on both, training matches to display precision), and **the AGNOS kernel**
(ring-3, booted in QEMU: trains + checkpoints + samples; checkpoint
bit-for-bit identical to Linux at fixed CPU — `scripts/agnos-smoke.sh`):

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
- The M2 (persistence), M3 (SIMD), and M5 (AGNOS port) code each passed an
  adversarial multi-agent review; all confirmed findings fixed and
  regression-tested. See CHANGELOG + [`../audit/`](../audit/).
- The M5 run gate (`scripts/agnos-smoke.sh`) is a developer-side check (needs
  the agnos/gnoboot/agnoshi sibling repos); CI gates the `--agnos` build +
  static-ELF shape only.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math,
  ganita, args (post-0.5.1: `random` dropped — unused, and its `sys_getrandom`
  doesn't exist on AGNOS; `matrix` dropped — unused, duplicates ganita's
  `mat_*` at 6.1.31; `ganita` added — `f64_tanh`/`f64_pow` live there now)

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). M5 (0.6.0) complete: AGNOS port with the
bit-for-bit run gate green. Next is **M6 (0.9.0 → v1.0.0): freeze &
consumer** — freeze the config/CLI surface, land one downstream consumer (or
example pipeline) against a tagged build, final security audit, first
non-prerelease tag. The **vidya corpus pipeline is the leading example-
pipeline candidate** (first run logged as X001 in
[`experiments.md`](experiments.md): 488 KB of Cyrius, loss 4.76 → ~1.55,
samples reproduce Cyrius idiom). Post-1.0 sequencing now lives in the
roadmap's **frontier track** (E1 KV-cached generation → E6 ternary training),
mapped from the June-2026 survey. Loose ends: an `attn11` row upstream in
`agnos/scripts/stage-tools.sh`; the cycc argv-capture issue
([`issues/`](issues/2026-06-10-cyrius-agnos-capture-after-gvar-init-call.md))
getting fixed upstream so the `docs/architecture/002` workaround can retire.
BPE tokenizer folded into frontier-track E3.
