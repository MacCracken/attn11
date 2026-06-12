# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.2.3** — *Decoupled RoPE* (M12 increment 5, **closes M12**; ADR 0007).
`--pos-kind rope-decoupled --rope-dim d_rope` adds the faithful DeepSeek-V2
decoupled RoPE for MLA (arXiv:2405.04434): position rides a **separate `d_rope`
channel** that bypasses the latent compression. The score splits into a CONTENT
term (compressed per-head K, dim `hd`) + a POSITION term (rope channel, dim
`d_rope`), summed and scaled by `1/sqrt(hd + d_rope)`. Two bias-free projections
per block — `W_QR` (per-head rope query `C→nh·d_rope`) + `W_KR` (the **shared**
rope key `C→d_rope`, from `x` directly) — both RoPE-rotated by position (reusing
1.2.2's `rope_apply_*`). The novel hand-derived piece is the decoupled softmax/PV
backward (shared `K^R` → `dKr` accumulates across heads), grad-checked **bit-tight**
in isolation (`test_attention_mla_dec`) + full-model + cached-vs-uncached
**bit-identity** across context-shifts (`test_kv_dec`). The decode cache holds the
latent `c` **and** the shared `K^R`: `kv_cache_bytes = NL·T·(d_c + d_rope)·8` —
**7680 B** at `d_c=16, d_rope=4` (latent 6144 + rope 1536), ~3.2× under MHA,
carrying relative position faithfully. v4 value-fills `pos_kind=2` + `rope_dim`
(round-trips + hostile rejections, `test_ckpt_dec`); no format bump. **M12's
`--pos-kind` switch is now complete** (learned / rope / rope-decoupled). **572**
grad-check/property tests green on x86_64 AND aarch64/qemu; fuzz + lint green.
Toolchain pinned at cyrius 6.1.37.
(1.2.2 — *Coupled RoPE* (M12 increment 4): `--pos-kind rope` rotates Q/K by
absolute position on dense MHA/GQA (`rope_apply_fwd`/`rope_apply_bwd`,
interleaved pairs, RoFormer arXiv:2104.09864) so the score depends only on `m-n`.
Parameter-free (grad-checked bit-exact + relative-position pin + cached bit-
identity); posemb off-path → zero gradient. Portable trig (Maclaurin + complex
binary-exponentiation; `f64_sin`/`f64_cos` are x86-only —
[`../architecture/005-rope-portable-trig.md`](../architecture/005-rope-portable-trig.md)).
1.2.1 — *MLA latent KV-cache decode* (M12.2): the cached single-row MLA decode
path (`attn_mla_fwd_row`) stores ONE low-rank latent `c` (`d_c` per token,
per-layer `LA_c`) instead of full per-head K/V, up-projecting on read.
`attn_core_fwd_row` extracted so MHA/GQA and MLA share the cached single-row
kernel. `--attn-kind mla` generates through the KV cache, bit-identical to the
uncached reference; `kv_cache_bytes` reports the latent footprint — **6144 B at
d_c=16, 4× under MHA**, MQA's footprint at full heads.
1.2.0 — *Multi-Head Latent Attention* (M12, the first new architecture on the
1.x arc): `--attn-kind mla` factors K/V through a low-rank latent (down `C→d_c`,
up `d_c→C`, `--latent-dim`; full heads), the DeepSeek-V2 parameterization
(arXiv:2405.04434). A shared `attn_core_fwd`/`attn_core_bwd` was extracted so
MHA/GQA and MLA run the **identical** softmax/PV kernel; the MLA backward composes
from `linear_bwd` + the core (no novel hand-derived math). **Checkpoint v4**
records the architecture descriptor (`attn_kind`/`latent_dim`), round-trips
bit-for-bit, and v1/v2/v3 still load. 1.2.0 generated MLA via the uncached
reference path; M12.2 (above) added the latent KV-cache decode.)
(1.1.0 — *the extraction*: the reusable numeric core lifted to **rosnet** 0.1.0
(tensor/BLAS-1/matmul + gradient) and **tyche** 0.1.0 (deterministic PRNG),
resolved via `cyrius deps`; additive/internal, byte-identical, attn11 the
reference consumer. Still no BLAS/libc/autodiff (both libs pure-Cyrius
`f64`-in-`i64`).
1.0.0 — the clean cut, **first non-prerelease**: the **final audit** (5
adversarial dimensions: hostile-input, math, memory, frozen-surface, release —
**go on all five, 0 blockers**;
[`../audit/2026-06-11-v1.0-final-audit.md`](../audit/2026-06-11-v1.0-final-audit.md))
plus release-hygiene fixes (`--save` exits non-zero on failure; `version-bump.sh`
updates `CFG_VERSION`; SECURITY wording); no features over 0.9.0, the surface
declared frozen ([`STABILITY.md`](../STABILITY.md)), additive-only past 1.0.
0.9.0 — freeze, docs & cleanup (M10): the user-facing surface declared
**frozen** ([`STABILITY.md`](../STABILITY.md)); CLI hardened (`--help`/
`--version`, rejects unknown args + missing flag values); 5-dimension docs
audit; dead code removed (`secure_write_file`/`f_println_lbl`/`CFG_NKV`); the
**vidya example pipeline**
landed ([`examples/vidya-pipeline.md`](../examples/vidya-pipeline.md): preset +
488 KB corpus → loss 1.089, bits/byte 1.760 → 5 MB checkpoint → sample); and
the toolchain pin moved **6.1.34 → 6.1.37** (`lib/` resynced). A no-flag run,
the checkpoint format, and training behavior are unchanged. 0.8.1 —
performance, M9 lever 1: SIMD tied LM head, **2.7×** at V=768; the three other
M9 levers were measured and rejected (X004). 0.8.0 — security sweep (M8): a
survey→map hardening release;
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

- **Cyrius pin**: `6.1.37` (in `cyrius.cyml [package].cyrius`) — bumped from
  6.1.34 during M10 (`cyrius update` resynced the `lib/` snapshot; 248 checks
  green on both arches + the agnos build). The pin and snapshot must always
  move together: cycc 6.1.32 fixed attn11's agnos argv-capture issue
  (r15-parked init rsp; the old `_agnos_init_rsp` global is gone) during M6,
  and a new-compiler/old-lib mismatch reproduces `argc()==0` under the kernel —
  the run gate caught it, so every pin bump is followed by `cyrius update`.
  (`docs/architecture/002` retired at ≥6.1.32.)

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
- **Checkpoints** (`--save`/`--load`): validated **v4** header — tokenizer
  triple + merge table validated as a well-founded DAG with bounded expansion +
  the architecture descriptor (`attn_kind`/`pos_kind`/`latent_dim`/`rope_dim`),
  all checked before allocation; **v1/v2/v3 still load** — + bit-for-bit
  **deterministic resume** (BPE re-encodes the retained corpus); **crash-atomic
  save** (temp + fsync + rename)
- **Multi-head latent attention** (1.2.0, `--attn-kind mla`, ADR 0007): K/V
  factored through a low-rank latent (down `C→d_c`, up `d_c→C`; `--latent-dim`,
  default `d_model/2`; full heads). Shares the extracted `attn_core_*` kernel with
  MHA/GQA; grad-checked per-op + full-model; checkpoint v4 carries the descriptor.
  Trains + checkpoints + samples.
- **MLA latent KV-cache decode** (1.2.1, M12.2): cached single-row MLA generation
  (`attn_mla_fwd_row`) stores the `d_c` latent per token (per-layer `LA_c`) and
  up-projects to K/V on read; `attn_core_fwd_row` shared with the MHA/GQA cached
  path. Bit-identical to the uncached reference. `kv_cache_bytes` reports the
  latent footprint — 6144 B at `d_c = 16`, **4× under MHA**, MQA's footprint at
  full heads. The absorption compute optimization is future work.
- **Coupled RoPE** (1.2.2, `--pos-kind rope`, ADR 0007): rotary positional
  embeddings on dense MHA/GQA (`rope_apply_fwd`/`rope_apply_bwd`) — Q/K rotated by
  absolute position so the score depends on `m-n` only (RoFormer). Parameter-free;
  mutually exclusive with learned-abs (posemb off-path → zero gradient). Portable
  trig (no x86-only `f64_sin`/`f64_cos`; `docs/architecture/005`). Even head dim,
  MHA/GQA only. Grad-checked + cached/uncached bit-identical; `np`/layout unchanged.
- **Decoupled RoPE** (1.2.3, `--pos-kind rope-decoupled --rope-dim d_rope`, ADR
  0007, **closes M12**): the faithful DeepSeek-V2 form for MLA — position on a
  separate `d_rope` channel (`W_QR` per-head query + `W_KR` shared key, bias-free,
  RoPE-rotated), score = content + position, scale `1/sqrt(hd+d_rope)`. The decode
  cache holds the latent + the shared `K^R` (`kv_cache_bytes = NL·T·(d_c+d_rope)·8`,
  7680 B at d_c=16/d_rope=4). New decoupled core (`attn_dec_core_*`,
  `attn_mla_dec_*`); grad-checked per-op + full-model + cached/uncached
  bit-identical; v4 carries `pos_kind=2`/`rope_dim`.
- CLI: `--corpus --stdin --load --save --steps --gen-only --preset --heads
  --kv-heads --layers --attn-kind --latent-dim --pos-kind --rope-dim --bpe --eval`

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
raises V to `base + K` (≤ 768). `--attn-kind mla --latent-dim d_c` (1 ≤ d_c ≤ C,
default C/2) swaps the K/V projections for the low-rank latent (37 952 params at
d_c=16 vs MHA's 39 488). `--pos-kind rope` (1.2.2; MHA/GQA, even head dim) swaps
learned-abs positions for coupled RoPE — parameter-free, so `params` is unchanged.

## Source (`src/`, ~1500 LOC)

- `tensor.cyr` — attn11-local float printing (`f_print`) + `_putc`/`puts` (40
  lines); the f64-array helpers + dense matmul moved to **rosnet**, the PRNG to
  **tyche** (1.1.0 extraction)
- `ops.cyr` — layernorm, GELU (tanh approx), softmax cross-entropy (forward +
  backward); `linear_fwd`/`linear_bwd` now resolve from **rosnet**
- `attn.cyr` — the shared attention core `attn_core_fwd`/`attn_core_bwd` (causal
  scaled-dot-product softmax/PV), wrapped by `attn_fwd`/`attn_bwd` (MHA/GQA
  projections) and `attn_mla_fwd`/`attn_mla_bwd` (MLA low-rank latent down/up
  projections; 1.2.0); one pre-allocated arena; the shared cached single-row core
  `attn_core_fwd_row`, wrapped by `attn_fwd_row` (MHA/GQA KV cache) and
  `attn_mla_fwd_row` (1.2.1: latent KV-cache decode — store the `d_c` latent,
  up-project to K/V on read) — each bit-identical per row to its batch path; plus
  coupled **RoPE** (1.2.2) `rope_apply_fwd`/`rope_apply_bwd` (gated by `pos_kind`
  in `attn_fwd`/`attn_bwd`/`attn_fwd_row`) with portable trig
  (`_rope_unit_cossin`/`_rope_pow` — Maclaurin + complex binary-exponentiation,
  no x86-only `f64_sin`/`f64_cos`; `docs/architecture/005`); plus the **decoupled
  RoPE** core (1.2.3) `attn_dec_core_fwd`/`bwd` + `attn_mla_dec_fwd`/`bwd` +
  `attn_mla_dec_fwd_row` (two-term content+position score, shared `K^R`, the
  latent+rope decode cache)
- `fileio.cyr` — secure file I/O (`O_NOFOLLOW`, `fstat` size, looped read/write),
  stdin reader
- `model.cyr` — per-layer packed parameters (block stride + `_o_*`/`PL`/`GL`
  helpers; one `_kv_weight_size()` branches the K/V region MHA↔MLA, ADR 0007),
  per-layer activation caches (+ the MLA latent `LA_c`, doubling as the 1.2.1
  latent decode cache), embeddings, tied head, full N-layer forward/backward
  (`attn_kind`-branched), grad clipping, Adam; KV caches + `model_fwd_row`
  (cached row, `attn_kind`-branched: full-K/V for MHA/GQA, latent for MLA) +
  `model_eval_window` (uncached eval reference); `kv_cache_bytes` reports the
  latent footprint for MLA; `model_init_arch`/`model_config_ok_arch`/
  `model_alloc_bytes_arch` carry the descriptor (the `_arch` forms; the old
  names delegate as MHA). `model_config_ok_arch`/`model_init_arch` also carry
  `pos_kind` (1.2.2: gate rope→even-hd + MHA/GQA); `embed_fwd_n`/`embed_bwd`/
  `model_fwd_row` skip the learned posemb add/grad under RoPE
- `train.cyr` — byte + **BPE** tokenizer (`bpe_learn`/`tok_encode`/
  `bpe_build_spans`/`tok_emit`), corpus (embedded/file/stdin, raw bytes
  retained), batch sampling, LR schedule, resumable training loop, KV-cached
  generation (`gen_prime`/`gen_decode` + context-shift), `eval_corpus`
  (CE/token + bits-per-byte)
- `persist.cyr` — validated **v4** checkpoint serialize/load (tokenizer triple +
  merge-table DAG/expansion validation + the v4 architecture descriptor, codes
  `-40..-43`; `ckpt_expected_np_arch` mirrors the MLA/decoupled layout; `pos_kind=1`
  (coupled RoPE) accepted on even-hd MHA images, `pos_kind=2`+`rope_dim` (decoupled)
  accepted on MLA images with even `2≤d_rope≤C` — bounded before alloc; v1/v2/v3
  accepted)
- `main.cyr` — CLI arg parsing (incl. `--preset`/`--heads`/`--kv-heads`/
  `--layers`/`--attn-kind`/`--latent-dim`/`--pos-kind`/`--rope-dim`/`--bpe`/`--eval`,
  null-guarded `_atoi`) + orchestration

## Tests

- `tests/attn11.tcyr` — **572 checks**: finite-difference gradient checks
  (every op incl. dropout; attention at head dims 6/8/10 and GQA/MQA at
  `nkv ∈ {1, 2, nh}` incl. `dWk`/`dWv`/`dbv`; the `|dbk| ≈ 0`
  softmax-shift-invariance pin; 2-layer full model at MHA and GQA), the **MLA
  suite** (1.2.0: per-op `attn_mla_fwd`/`bwd` grad-check at 3 latent configs
  incl. the `|dbuk| ≈ 0` shift-invariance pin, full-model MLA grad-check, the
  **MLA parameter-layout/alloc/config pins**, and the v4 MLA checkpoint
  round-trip + `-42` descriptor-consistency rejections), the SIMD
  bit-contract, the **SIMD-LM-head tail pin** (`C % 4 ≠ 0` at C=6 vs a scalar
  dot — mutation-verified; no other config exercises it), the **parameter-layout tiling pin** (FD is blind to offset
  aliasing, MHA + MLA), the **alloc-accounting pin** (`model_alloc_bytes` ==
  `model_init`, incl. V=300 + MLA), the **config-magnitude-cap pin**
  (`model_config_ok` rejects out-of-range V/C/T/NL + the MLA `d_c`/`nkv` gates —
  the `--layers` heap-OOB regression), resume-determinism (dropout off/on + MQA + BPE), the
  **file-path round-trip** (`ckpt_save_file`/`ckpt_load_file` — the in-memory
  tests never touched the file loader; pins `_file_size`/`fdatasync`, the M8
  aarch64 save fix), checkpoint rejection smokes (+ `-18` pre-alloc bound,
  `-19` rng=0, the v3 `-32…-39` matrix incl. the merge-table forgery cascade)
  + v2/GQA/**v3** round-trip + **v1/v2-compat load**, the **BPE suite** (known-merge,
  round-trip, cross-arch determinism, generation bit-identity), the
  **eval/bits-per-byte** determinism + RNG-neutrality pin, the **KV
  bit-identity suite** (prefill at every prefix + decode across
  context-shifts, greedy + temperature, at hd ∈ {4, 6, 8, 10} ×
  nkv ∈ {1, 2, nh} incl. odd-T shifts, **preset shape**, and V=300), the
  **MLA latent KV-cache bit-identity suite** (1.2.1, `test_kv_mla`:
  cached-vs-uncached prefill + decode across shifts, greedy + temperature, at
  hd ∈ {6, 8, 10} × `d_c = C/2`/`d_c ∤ C`, odd T, 2-token window), the **coupled
  RoPE suite** (1.2.2: `test_rope_op` — bit-exact rotation backward +
  relative-position invariance; `test_attention_rope` — attention grad-check at
  hd ∈ {6,8,10} × MHA/GQA/MQA incl. the now-real K-bias gradient;
  `test_model_rope` — full-model wiring + posemb-zero-gradient pin; `test_kv_rope`
  — cached-vs-uncached bit-identity across shifts; `test_ckpt_rope` — v4
  `pos_kind=1` round-trip + odd-hd/mla+rope rejections;
  config-cap rope gates), the **decoupled RoPE suite** (1.2.3: `test_attention_mla_dec`
  — per-op decoupled grad-check (dWqr/dWkr + the shared-Kr backward, tight 1e-4) at
  3 d_c/d_rope configs; `test_model_mla_dec` — full-model wiring; `test_kv_dec` —
  cached-vs-uncached bit-identity across shifts incl. a non-even content head dim;
  `test_param_layout_mla_dec` — the W_QR/W_KR aliasing pin; `test_ckpt_dec` — v4
  `pos_kind=2`/`rope_dim` round-trip + decoupled hostile rejections; alloc-accounting
  + config-cap decoupled gates), a
  **soak/leak** test and a **NaN guard** test. All pass on x86_64 AND aarch64
  (`cyrius test`; aarch64 via qemu).
- `tests/attn11.bcyr` — benchmark harness (training timings + tokens/sec,
  generation cached/uncached, KV bytes per nkv, MQA timings, **preset
  train+gen**, **MLA latent-cache gen + the cache-bytes table** (latent vs
  MHA/MQA full-K/V), **RoPE train-step + cached-gen overhead**, **`bpe_learn`
  cost**).
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
- **[rosnet](https://github.com/MacCracken/rosnet) 0.1.0** — tensor storage +
  BLAS-1 + dense matmul/gradient (`linear_fwd`/`linear_bwd`, `t_*`); 1.1.0
  extraction, pinned in `cyrius.lock`
- **[tyche](https://github.com/MacCracken/tyche) 0.1.0** — deterministic
  statistical PRNG (`rng_seed`/`rng_u64`/`rng_uniform`/`rng_normal`); 1.1.0
  extraction, pinned in `cyrius.lock`

## Consumers

_None yet._

## Next

See [`roadmap.md`](roadmap.md). **v1.0.0 (clean cut), v1.1.0 (extraction),
v1.2.0 (MLA core), v1.2.1 (MLA latent KV-cache decode), v1.2.2 (coupled RoPE), and
v1.2.3 (decoupled RoPE) shipped — M12 is complete.** The surface is frozen
([`STABILITY.md`](../STABILITY.md)) and additive-only past 1.0; the numeric core
lives in **rosnet** + **tyche**. The 1.x architecture arc now has the full
attention/position axes: `--attn-kind {mha, mla}` × `--pos-kind {learned, rope,
rope-decoupled}` (checkpoint v4).

**Next on the 1.x architecture arc** (roadmap):
- **M13 (v1.3.0) — Mixture of Experts** (E8; sparse FFN with the
  `--experts {8,16,32,64,128,256}` density sweep, checkpoint v5), then E4–E6
  (mixers / diffusion / ternary) as M14–M16, and **M17** reinforcement learning
  (E9).

Loose ends: an `attn11` row upstream in `agnos/scripts/stage-tools.sh`
(`stage_one attn11 src/main.cyr attn11`) — a cross-repo edit (agnos maintainer's
side). The pin and `lib/` snapshot move together on every bump (now **6.1.37**,
resynced via `cyrius update`); rosnet/tyche resolve via `cyrius deps` (pinned in
`cyrius.lock`). (The cycc argv-capture issue is **resolved** — fixed upstream in
6.1.32; `docs/architecture/002` retired as a load-bearing rule.)
