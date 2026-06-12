# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**0.8.1** — performance, M9 lever 1 (SIMD tied LM head): `head_fwd_row` (the
weight-tied output projection, `O(V·C)` per row, run in every training forward
and every generated token) was a scalar dot product while the matmul has been
4-wide since 0.4.0. Vectorized with the same `f64v_fmadd` accumulator + tail —
**2.7×** at V=768 (`head_fwd` 9.7 → 3.59 ms). The win scales with the vocab
(negligible at the V=25 default — default training unchanged; ~17% of the
forward at BPE-scale). Shared kernel, so the cached-vs-uncached bit-identity
gate is unaffected; a mutation-verified C=6 test covers the new `C % 4 ≠ 0`
tail. (0.8.0 — security sweep (M8): a survey→map hardening release;
checkpoint **format immunity** to the model-file-deser RCE genre confirmed; a
dropped `_file_size` arg crashing every **AGNOS `--load`** and checkpoint
**save broken on the aarch64 lane** (qemu `fsync` → `fdatasync`) both fixed;
`_atoi` saturation, merge-scratch pin, **CI supply-chain** hardening. See
[`../audit/2026-06-11-m8-security-sweep-audit.md`](../audit/2026-06-11-m8-security-sweep-audit.md).
0.7.1 — scale preset + BPE (M7, E3): `--preset` (ctx 64 / d_model 64; gen
**23×**), opt-in **BPE** (`--bpe K`, ADR 0006), checkpoint **v3**, `--eval`
bits-per-byte, pin 6.1.33 → 6.1.34; X003 byte-vs-BPE −11 to −13% bits/byte.
0.7.0 — inference efficiency (M6, E1+E2): KV-cached generation
(6.2×) + GQA, checkpoint v2, pin 6.1.31 → 6.1.33. 0.6.0 — AGNOS kernel port
(M5), bit-for-bit checkpoint vs Linux. 0.5.1 — standards conformance. 0.5.0 —
aarch64 validation, NaN/inf guard, soak, crash-atomic save. 0.4.0: 4-wide
SIMD matmul, ~2.27× faster. 0.3.0: corpus loading, checkpoints +
deterministic resume. 0.2.0: stacked layers, grad clipping, LR schedule.)

## Toolchain

- **Cyrius pin**: `6.1.34` (in `cyrius.cyml [package].cyrius`) — bumped from
  6.1.33 during M7, with the matching `lib/` snapshot. The pin and snapshot
  must always move together: cycc 6.1.32 fixed attn11's agnos argv-capture
  issue (r15-parked init rsp; the old `_agnos_init_rsp` global is gone) during
  M6, and a new-compiler/old-lib mismatch reproduces `argc()==0` under the
  kernel — the run gate caught it. (`docs/architecture/002` retired at ≥6.1.32.)

## Performance

4-wide SIMD (`f64v_fmadd`) matmul. x86_64:

- Training (default config): fwd+bwd step ~3.7 ms, **~4 300 tokens/sec**
  (b=16) — unchanged from 0.6.0 within noise. Preset (ctx 64 / d_model 64):
  fwd+bwd ~63 ms, ~1 000 tok/s — ~17× the default step for 5.2× the params
  and 4× the context.
- Generation, default config (0.7.0): uncached 1 050 579 ns/token →
  **KV-cached 170 392 ns/token (6.2×, 951 → 5 868 tok/s)**, greedy.
- Generation, **preset** (0.7.1): uncached 15 564 530 ns/token → **KV-cached
  672 747 ns/token (23×, 64 → 1 486 tok/s)** — the context-shift re-prime
  amortizes over T/2 = 32 tokens at ctx 64.
- KV cache bytes (default config): 24 576 at `nkv=4` → 12 288 (`nkv=2`) →
  6 144 (`nkv=1`).
- BPE merge training (`--bpe K`): one-shot ~110 ms for 256 KB at K=128.

See [`benchmarks.md`](../benchmarks.md) + [`../../bench-history.csv`](../../bench-history.csv).

## What works

End-to-end, on Linux x86_64, **aarch64** (cross-build + qemu; all checks pass
on both), and **the AGNOS kernel** (ring-3, booted in QEMU; bit-for-bit
checkpoint vs Linux at fixed CPU — `scripts/agnos-smoke.sh`):

- **Byte-level adaptive tokenizer** (default) + opt-in **simple BPE**
  (`--bpe K`, ≤512 merges; 0.7.1, ADR 0006): merges layer on the byte base
  vocab, frozen deterministic tie-break, pure i64 (bit-reproducible
  cross-arch), decode via a precomputed flat span table (no recursion)
- Token + learned positional embeddings
- **`n_layers` stacked** pre-norm Transformer blocks, each:
  `LayerNorm → causal multi-head self-attention → residual → LayerNorm → MLP (GELU) → residual`
- **Scale `--preset`** (0.7.1): ctx 64 / d_model 64 / 8 heads / 4 layers,
  with `--heads`/`--kv-heads`/`--layers` overrides for fresh models
  (magnitude-capped + alloc-pre-flighted in `model_init`, mirroring the
  checkpoint loader — file and CLI config gates share one invariant)
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
  size-cap, byte-level adaptive vocab (raw bytes retained for BPE re-encode)
- **`--eval`** (0.7.1): one deterministic, RNG-neutral pass over the corpus →
  CE/token + **bits-per-byte** (tokenizer-comparable); runs after `--save`,
  so checkpoints are bit-identical with or without it
- **Checkpoints** (`--save`/`--load`): validated **v3** header — tokenizer
  triple + merge table validated as a well-founded DAG with bounded expansion,
  all checked before allocation; **v1/v2 still load** (byte-level) — +
  bit-for-bit **deterministic resume** (BPE re-encodes the retained corpus);
  **crash-atomic save** (temp + fsync + rename)
- CLI: `--corpus --stdin --load --save --steps --gen-only --preset --heads
  --kv-heads --layers --bpe --eval`

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

`--preset` overrides to C 64 / T 64 / nh 8 / NL 4 (205 760 params at the
embedded corpus); `--heads`/`--kv-heads`/`--layers` override individual dims
(magnitude-capped: nh|C, nkv|nh, NL ≤ 128, C ≤ 4096, T ≤ 8192). `--bpe K`
raises V to `base + K` (≤ 768).

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
  `model_fwd_row` (cached row) + `model_eval_window` (uncached eval reference);
  `model_config_ok` (magnitude + divisibility caps) + `model_alloc_bytes`
  pre-flight guard the fresh-model path
- `train.cyr` — byte + **BPE** tokenizer (`bpe_learn`/`tok_encode`/
  `bpe_build_spans`/`tok_emit`), corpus (embedded/file/stdin, raw bytes
  retained), batch sampling, LR schedule, resumable training loop, KV-cached
  generation (`gen_prime`/`gen_decode` + context-shift), `eval_corpus`
  (CE/token + bits-per-byte)
- `persist.cyr` — validated **v3** checkpoint serialize/load (tokenizer triple
  + merge-table DAG/expansion validation; v1/v2 accepted as byte-level)
- `main.cyr` — CLI arg parsing (incl. `--preset`/`--heads`/`--kv-heads`/
  `--layers`/`--bpe`/`--eval`, null-guarded `_atoi`) + orchestration

## Tests

- `tests/attn11.tcyr` — **248 checks**: finite-difference gradient checks
  (every op incl. dropout; attention at head dims 6/8/10 and GQA/MQA at
  `nkv ∈ {1, 2, nh}` incl. `dWk`/`dWv`/`dbv`; the `|dbk| ≈ 0`
  softmax-shift-invariance pin; 2-layer full model at MHA and GQA), the SIMD
  bit-contract, the **SIMD-LM-head tail pin** (`C % 4 ≠ 0` at C=6 vs a scalar
  dot — mutation-verified; no other config exercises it), the **parameter-layout tiling pin** (FD is blind to offset
  aliasing), the **alloc-accounting pin** (`model_alloc_bytes` ==
  `model_init`, incl. V=300), the **config-magnitude-cap pin**
  (`model_config_ok` rejects out-of-range V/C/T/NL — the `--layers`
  heap-OOB regression), resume-determinism (dropout off/on + MQA + BPE), the
  **file-path round-trip** (`ckpt_save_file`/`ckpt_load_file` — the in-memory
  tests never touched the file loader; pins `_file_size`/`fdatasync`, the M8
  aarch64 save fix), checkpoint rejection smokes (+ `-18` pre-alloc bound,
  `-19` rng=0, the v3 `-32…-39` matrix incl. the merge-table forgery cascade)
  + v2/GQA/**v3** round-trip + **v1/v2-compat load**, the **BPE suite** (known-merge,
  round-trip, cross-arch determinism, generation bit-identity), the
  **eval/bits-per-byte** determinism + RNG-neutrality pin, the **KV
  bit-identity suite** (prefill at every prefix + decode across
  context-shifts, greedy + temperature, at hd ∈ {4, 6, 8, 10} ×
  nkv ∈ {1, 2, nh} incl. odd-T shifts, **preset shape**, and V=300), a
  **soak/leak** test and a **NaN guard** test. All pass on x86_64 AND aarch64
  (`cyrius test`; aarch64 via qemu).
- `tests/attn11.bcyr` — benchmark harness (training timings + tokens/sec,
  generation cached/uncached, KV bytes per nkv, MQA timings, **preset
  train+gen**, **`bpe_learn` cost**).
- `tests/attn11.fcyr` — fuzz harness: 500 mutated-checkpoint rounds (v2/v3
  header fields incl. nkv/step, + a **boundary-combination** mode: every size
  field at/over its cap at once) + **500 BPE-image rounds** (merge-slot
  clobber, (V,Vb,K) triple inconsistency, expansion-bomb rewrite, + the
  **max-vocab triple** `V=768/Vb=256/K=512`) + 100 random corpora + a **BPE
  round-trip** property; loaders reject malformed input without crashing.
- The M2 (persistence), M3 (SIMD), M5 (AGNOS port), M6 (KV-cache/GQA),
  **M7 (BPE/preset/v3)**, and **M8 (security sweep)** code each passed an
  adversarial multi-agent review; all confirmed findings fixed and
  regression-tested. M6's review (50 agents) drove the checkpoint pre-alloc
  bound; M7's (9 agents) caught the **`--layers` heap-OOB**; M8's survey→map
  (12 agents) confirmed checkpoint **format immunity** to the model-file-deser
  RCE genre and surfaced the **AGNOS `--load`** crash + the **aarch64 save**
  break (qemu `fsync`). See CHANGELOG +
  [`../audit/2026-06-11-m8-security-sweep-audit.md`](../audit/2026-06-11-m8-security-sweep-audit.md)
  + [`../audit/2026-06-11-m7-bpe-audit.md`](../audit/2026-06-11-m7-bpe-audit.md)
  + [`../audit/2026-06-11-kv-gqa-audit.md`](../audit/2026-06-11-kv-gqa-audit.md).
- The M5 run gate (`scripts/agnos-smoke.sh`) is a developer-side check (needs
  the agnos/gnoboot/agnoshi sibling repos); it now also exercises `--load`
  under AGNOS (M8). CI gates the `--agnos` build + static-ELF shape only.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — string, fmt, alloc, io, vec, str, syscalls, assert, bench, math,
  ganita, args

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). **M9 (v0.8.x) — performance — concluded.**
The **SIMD tied LM head** (0.8.1, 2.7× at V=768) was the one clean win; the
other three roadmap levers were prototyped and **rejected by measurement**
(X004): GELU-tanh marginal (~1–2%, `exp` is cheap), matmul cache-blocking
*slower* (matrices are cache-resident), batched prefill *no win* (~1%; the
re-prime is irreducible work, not overhead). The residual matmul gap to SIMD
peak is structural (the SIMD-var-reassign rule + 2-wide `f64v_fmadd`) and needs
toolchain support, not a v0.8.x change.

**Next — M10 (v0.9.0)**: freeze the config/CLI surface, finish docs, land the
vidya example pipeline (X001) against a tagged build — so **v1.0.0 is a clean
cut** (final audit + tag only).

Loose ends: an `attn11` row upstream in `agnos/scripts/stage-tools.sh`
(folds into M10 cleanup). The pin and `lib/` snapshot must move together on
every bump (now 6.1.34). (The cycc argv-capture issue is **resolved** — fixed
upstream in 6.1.32; `docs/architecture/002` retired as a load-bearing rule.)
