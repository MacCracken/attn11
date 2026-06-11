# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.7.0** — inference efficiency (M6, frontier E1+E2 graduated): KV-cached
generation (**6.2× faster sampling**, bit-identical to the uncached
reference) + grouped-query attention (`n_kv_heads` config, KV cache up to 4×
smaller at `nkv=1`); checkpoint format v2 (+`nkv`, v1 still loads) with a
pre-allocation bound on hostile configs; toolchain pin 6.1.31 → 6.1.33
(argv-capture fix landed upstream; `docs/architecture/002` retired);
training at the default config unchanged. (0.6.0 — AGNOS kernel port (M5),
bit-for-bit checkpoint vs Linux, toolchain pin 6.1.6 → 6.1.31. 0.5.1 —
standards conformance. 0.5.0 — aarch64 validation, NaN/inf guard, soak,
crash-atomic save. 0.4.0: 4-wide SIMD matmul, ~2.27× faster. 0.3.0: corpus
loading, checkpoints + deterministic resume. 0.2.0: stacked layers, grad
clipping, LR schedule.)

## Toolchain

- **Cyrius pin**: `6.1.33` (in `cyrius.cyml [package].cyrius`) — bumped from
  6.1.31 during M6: cycc 6.1.32 fixed attn11's agnos argv-capture issue
  (r15-parked init rsp; the old `_agnos_init_rsp` global is gone), and a
  new-compiler/old-lib mismatch reproduces `argc()==0` under the kernel —
  the run gate caught it. Pin and `lib/` snapshot must move together.

## Performance

4-wide SIMD (`f64v_fmadd`) matmul. Default config, x86_64:

- Training: fwd+bwd step ~3.7 ms, **~4 350 tokens/sec** (b=16) — unchanged
  from 0.6.0 within noise.
- Generation (0.7.0): uncached 1 050 579 ns/token → **KV-cached 170 392
  ns/token (6.2×, 951 → 5 868 tok/s)**, greedy, default config.
- KV cache bytes (default config): 24 576 at `nkv=4` → 12 288 (`nkv=2`) →
  6 144 (`nkv=1`).

See [`benchmarks.md`](../benchmarks.md) + [`../../bench-history.csv`](../../bench-history.csv).

## What works

End-to-end, on Linux x86_64, **aarch64** (cross-build + qemu; all checks pass
on both), and **the AGNOS kernel** (ring-3, booted in QEMU; bit-for-bit
checkpoint vs Linux at fixed CPU — `scripts/agnos-smoke.sh`):

- Char-level tokenizer over an embedded corpus
- Token + learned positional embeddings
- **`n_layers` stacked** pre-norm Transformer blocks, each:
  `LayerNorm → causal multi-head self-attention → residual → LayerNorm → MLP (GELU) → residual`
- **Grouped-query attention** (0.7.0): `n_kv_heads ≤ n_heads` shares K/V
  heads across query-head groups (`nkv = nh` = classic MHA, the default;
  `nkv = 1` = MQA); K/V projections are `C × Ckv`
- Final `LayerNorm` + weight-tied LM head → softmax cross-entropy
- Hand-written backprop through every op and the full residual stack
  (verified; incl. grouped dK/dV accumulation)
- **Adam** + **global-norm gradient clipping** + **LR warmup→cosine** schedule
- GPT-2 residual-projection init scaling (`1/sqrt(2·n_layers)`)
- Config-gated **attention biases** and **residual dropout** (dropout
  auto-disabled in eval/generation)
- Mini-batch grad accumulation; training logs loss / lr / grad-norm
- **NaN/inf training guard** (stops cleanly instead of poisoning weights)
- **KV-cached autoregressive generation** (0.7.0): per-layer K/V caches, one
  cached row per token, context-shift (drop oldest T/2 + re-prime) when the
  window fills; **bit-identical** to the uncached reference path; greedy +
  temperature sampling
- **Corpus from file/stdin** (`--corpus`/`--stdin`): `O_NOFOLLOW`, `fstat`
  size-cap, byte-level adaptive vocab
- **Checkpoints** (`--save`/`--load`): validated v2 header (all checked before
  allocation; v1 still loads as `nkv=nh`) + bit-for-bit **deterministic
  resume**; **crash-atomic save** (temp + fsync + rename)
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
| kv heads    | 4     | = nh (full MHA; `nkv < nh` = GQA) |
| layers `NL` | 3     | stacked pre-norm blocks           |
| MLP `F`     | 128   | = 4·C                             |
| attn bias   | on    | Q/K/V/O biases (config-gated)     |
| dropout     | 0.0   | residual dropout (config-gated)   |
| params      | 39488 | total trainable f64 (3 layers, biases) |
| optimizer   | Adam  | β 0.9/0.999, global-norm clip 1.0 |
| lr schedule | warmup 100 → cosine | base 3e-3 → min 3e-4  |
| steps/batch | 2000 / 16 |                               |

## Source (`src/`, ~1500 LOC)

- `tensor.cyr` — f64-array helpers, deterministic PRNG (xorshift64 + splitmix
  seeding, Marsaglia-polar normal), float printing
- `ops.cyr` — linear, layernorm, GELU (tanh approx), softmax cross-entropy
  (forward + backward)
- `attn.cyr` — causal multi-head/GQA self-attention (forward + backward), one
  pre-allocated arena for caches + temporaries; `attn_fwd_row` (KV-cached
  single-row forward, bit-identical per row to `attn_fwd`)
- `fileio.cyr` — secure file I/O (`O_NOFOLLOW`, `fstat` size, looped read/write),
  stdin reader
- `model.cyr` — per-layer packed parameters (block stride + `_o_*`/`PL`/`GL`
  helpers, Ckv-dependent), per-layer activation caches, embeddings, tied head,
  full N-layer forward/backward, grad clipping, Adam; KV caches +
  `model_fwd_row` (cached row) + `model_eval_window` (uncached eval reference)
- `train.cyr` — byte-level tokenizer, corpus (embedded/file/stdin), batch
  sampling, LR schedule, resumable training loop, KV-cached generation
  (`gen_prime`/`gen_decode` + context-shift)
- `persist.cyr` — validated v2 checkpoint serialize/load (v1 accepted)
- `main.cyr` — CLI arg parsing + orchestration

## Tests

- `tests/attn11.tcyr` — **161 checks**: finite-difference gradient checks
  (every op incl. dropout; attention at head dims 6/8/10 and GQA/MQA at
  `nkv ∈ {1, 2, nh}` incl. `dWk`/`dWv`/`dbv`; the `|dbk| ≈ 0`
  softmax-shift-invariance pin; 2-layer full model at MHA and GQA), the SIMD
  bit-contract, the **parameter-layout tiling pin** (FD is blind to offset
  aliasing), the **alloc-accounting pin** (`model_alloc_bytes` ==
  `model_init`), resume-determinism (dropout off/on + MQA), checkpoint
  rejection smokes (+ `-18` pre-alloc bound, `-19` rng=0) + v2/GQA
  round-trip + **v1-compat load**, the **KV bit-identity suite** (prefill at
  every prefix + decode across context-shifts, greedy + temperature, at
  hd ∈ {4, 6, 8, 10} × nkv ∈ {1, 2, nh} incl. odd-T shifts), a **soak/leak**
  test and a **NaN guard** test. All pass on x86_64 AND aarch64
  (`cyrius test`; aarch64 via qemu).
- `tests/attn11.bcyr` — benchmark harness (training timings + tokens/sec,
  generation cached/uncached, KV bytes per nkv, MQA timings).
- `tests/attn11.fcyr` — fuzz harness: 500 mutated-checkpoint rounds (v2
  header fields incl. nkv/step) + 100 random corpora; loaders reject
  malformed input without crashing.
- The M2 (persistence), M3 (SIMD), M5 (AGNOS port), and M6 (KV-cache/GQA)
  code each passed an adversarial multi-agent review; all confirmed findings
  fixed and regression-tested. M6's review (50 agents, 15 raw → 9 confirmed)
  also drove the checkpoint pre-allocation bound and config-invariant
  hardening. See CHANGELOG + [`../audit/2026-06-11-kv-gqa-audit.md`](../audit/2026-06-11-kv-gqa-audit.md).
- The M5 run gate (`scripts/agnos-smoke.sh`) is a developer-side check (needs
  the agnos/gnoboot/agnoshi sibling repos); CI gates the `--agnos` build +
  static-ELF shape only.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math,
  ganita, args

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). M6 (0.7.0) complete: KV-cached generation +
GQA, both gates green. Next is either **E3 (scale preset + BPE)** as 0.8.0 —
the ctx-64/d_model-64 preset the vidya run motivated (X001), making
byte-vs-BPE measurable at iso-compute — or straight to **M7 (0.9.0 → v1.0.0):
freeze & consumer** (freeze the config/CLI surface, land the vidya example
pipeline against a tagged build, final audit, first non-prerelease tag).
Loose ends: an `attn11` row upstream in `agnos/scripts/stage-tools.sh`; BPE
tokenizer folded into frontier-track E3. (The cycc argv-capture issue is
**resolved** — fixed upstream in 6.1.32, pin bumped to 6.1.33,
`docs/architecture/002` retired as a load-bearing rule.)
