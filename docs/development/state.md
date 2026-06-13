# attn11 — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.4.4** — *Any-mixer hybrids* (M14 rung d, E4, **completes M14**; ADR 0012).
Lifts 1.4.3's layout restriction so a hybrid interleaves ANY of the four mixers
`{mha, mla, lin, ssm}` — including full attention ⊕ the selective SSM (the survey's
strongest pairing, attn11's best single mixer). The trick: each block's K/V region
is **padded to the max `_kvw` over the kinds present** (`_kvw_hyb`), so the
per-block stride stays uniform — only `_kv_weight_size()` + the per-layer init/cache
gates change, NOT every `_o_*` offset (no per-layer-offset refactor). A smaller
kind tiles its weights and leaves a zeroed pad. `_hybrid_kinds_ok` keeps the
cross-cutting constraints (learned-abs, full heads, a valid shared latent iff any
mla/ssm). Checkpoint **v6 unchanged** (already carries the per-layer kinds, 1.4.3);
the loader sizes the padded block the same way (`ckpt_expected_np_kvw`). The mixed
SSM/MLA ⊕ MHA backward grad-checks (`test_model_hybrid_ssm`/`_mla`, ~1e-4);
mha/ssm + mha/mla cached decode bit-identical; padded mha/ssm v6 round-trip green.
Attention-fraction sweep (X013, base ssm): bits/byte within noise (0.218 pure-ssm →
0.279 pure-mha), the decode cache a continuous knob from constant `C·N` to ∝T K/V.
`{mha,gqa,lin}` hybrids stay exact (no pad). A no-flag run is byte-identical.
Verified: **907** checks x86_64 AND aarch64/qemu, agnos, fuzz, lint.
(1.4.3 — *Per-layer mixer hybrid* (M14 rung c, E4, **the interleaving lever**;
ADR 0011). `--attn-every K` places a full-attention (MHA) block at every K-th layer
and a gated-linear block elsewhere — the survey's "a few attention layers among
many cheap recurrent ones" structural shift. The global `attn_kind` becomes a
per-layer `g_layer_kind`, read ONLY by the three `_attn_block_*` dispatch helpers
(`_lk(L)`); uniform models get the global back, byte-identical. Restricted to
**layout-compatible kinds {mha, gqa, lin}** (gated-linear reuses MHA's projections,
ADR 0009), so `_kvw`/`_blk`/the offsets/`g_NP` are identical and the per-block
stride stays uniform — no per-layer offset refactor, and the hybrid is
**parameter-free**. What it buys is a knob on the decode cache: `kv_cache_bytes`
SUMS the per-layer caches, so the attention fraction sets how much is T-growing K/V
vs constant lin state (1/3 attention ⇒ half of pure-MHA's). First checkpoint
**format bump (v6)**: a hybrid persists its per-layer pattern (loader rejects an
invariant-breaking kind, `-46`); uniform models still write v5, byte-identical. The
mixed backward grad-checks (`test_model_hybrid` ~1e-5); cached decode bit-identical
per interleaving (`test_kv_hybrid`); v6 round-trip green (`test_ckpt_hybrid`).
Attention-fraction sweep (X012, default config, 1200 steps): bits/byte within noise
across ratios (all beat pure-MHA 0.279), cache scales with the fraction — a
"trains + grad-checks", not a scaling claim. Verified: **857** checks x86_64 AND
aarch64/qemu, agnos, fuzz, lint.
1.4.2 — *Selective SSM* (M14 rung b, E4, **the third sequence mixer**; ADR
0010). `--attn-kind ssm` adds a minimal Mamba-lite diagonal SSM: a per-channel
N-state recurrence `h_t = exp(Δ·A)·h_{t-1} + Δ·B·a`, `y = Σ C·h + D·a`, with
Δ/B/C all functions of the input (the *selective* scan). The milestone is the
**hand-derived BPTT through the data-dependent scan** — `test_ssm_core` grad-checks
every parameter + the input at **~1e-7**. Reuses Wq (W_dt) + Wo (output proj) and
the `latent_dim` field (= state size N), so it rides `attn_kind = 3` — **checkpoint
v5, no format bump**. The decode cache is the constant `C·N` state (a third
constant-cache mixer). Mixer comparison (X011, default config, 1200 steps): SSM
**bits/byte 0.218** — best of the four (vs linear 0.239, MLA 0.273, MHA 0.279) — at
38 048 params, cache constant in T (8× under MHA at the preset). Full-model +
cached bit-identity green; lands in its own `attn_ssm.cyr`. A no-flag run is
byte-identical. Verified: **801** checks x86_64 AND aarch64/qemu, agnos, fuzz, lint.
1.4.1 — *Refactoring sweep* (maintenance; no behavior change — the no-flag
run byte-identical, **727** checks unchanged on both arches, every checkpoint
round-trips). Reorganizes the mixer machinery so the M14 rungs are cheap:
(1) the `attn_kind` dispatch is now ONE point each — `_attn_block_fwd`/`_bwd`/
`_fwd_row` in `model.cyr` (was inlined in four functions); (2) the per-block param
arithmetic is shared pure helpers `_kvw`/`_mlpw` used by both the offset helpers
and the checkpoint validator (the model↔persist keep-in-sync hazard is gone);
(3) the gated-linear mixer moved to its own `attn_linear.cyr` (one-file-per-mixer
pattern; `attn.cyr` 1266→976); (4) the six `_gen_bits_*` test helpers collapsed
to one driver. `src/*.cyr` identical in effect to 1.4.0.
1.4.0 — *Gated linear attention* (M14 rung a, E4, **opens the second
sequence-mixer family**; ADR 0009). `--attn-kind lin` swaps the softmax/PV core
for a causal RetNet-style **retention recurrence** `S_t = γ_h·S_{t-1} + k_t⊗v_t`,
`out_t = (1/√hd)·S_t^T q_t`, fixed per-head decay `γ_h = 1−2^{−(3+h)}`
(parameter-free). It reuses the MHA Q/K/V/O projections, so it rides the existing
`attn_kind` slot (value 2) — checkpoint **v5, no format bump**. The headline: the
decode cache is the **constant** `nh·hd²` retention state, not a T-growing K/V. The
hand-derived backward needs no state caching (`dq` via a forward S-recompute,
`dk`/`dv` via a reverse `dS`); pure multiply/add, so `test_lin_core` grad-checks at
**~1e-9**, full-model + cached bit-identity green. Mixer comparison (X010, default
config, 1200 steps): linear **bits/byte 0.239** (vs MHA 0.279, MLA 0.273) at MHA's
exact param count (39 488), with a 6 144 B cache that is **constant in T** (16×
under MHA at the preset). A no-flag run is byte-identical. Verified green: **727**
checks x86_64 AND aarch64/qemu, the `--agnos` static-ELF build, fuzz, lint.
1.3.0 — *Mixture of Experts* (M13, E8, **opens the FFN-density axis**; ADR
0008). The dense GELU MLP in each block becomes **N experts + a top-K router**:
`--experts N --expert-topk K` (N in 1..256, default topk 2; `--experts 1` = the
byte-identical dense baseline). The milestone is the **router backward** —
a discrete top-K pick (frozen, lower-index tie-break, bit-reproducible cross-arch)
→ a Mixtral-style renormalized top-K softmax combine (gradient only to the
selected logits, straight-through) + a Switch-style load-balance aux loss
(`α·N·Σ fᵢ·Pᵢ`, dispatch counts held constant) — both hand-derived and
finite-difference grad-checked (`test_moe_op` 1e-4, `test_moe_aux` 1e-5, full
model 1e-3; cached-vs-uncached **bit-identity** `test_kv_moe`). Checkpoint **v5**
records `num_experts`/`topk` (v1–v4 load, synthesizing the dense MLP). The
density sweep (X009, `scripts/moe-sweep.sh`): total params scale ~linearly with N
(39 K → 1.62 M at N=64) while per-token-active stays ~65–71 K, **routing entropy
0.993–0.999** (load stays balanced), bits/byte best at N=8–16. The cyrius pin also
moved **6.2.1 → 6.2.2** (clean patch realign, byte-identical `./lib/` snapshot).
Verified green: **673** checks x86_64 AND aarch64/qemu, the `--agnos` static-ELF
build, fuzz, lint — `make release` exit 0.
1.2.4 — *Toolchain realignment + docs* (maintenance): pin **6.1.37 → 6.2.1**,
`./lib/` resynced, **572** checks green on both arches + agnos + fuzz; roadmap
trimmed forward-facing, handoff section added. `src/*.cyr` identical to 1.2.3.
1.2.3 — *Decoupled RoPE* (M12 increment 5, **closes M12**; ADR 0007):
`--pos-kind rope-decoupled --rope-dim d_rope` — the faithful DeepSeek-V2 form for
MLA (arXiv:2405.04434). Position rides a **separate `d_rope` channel** that
bypasses the latent; the score = CONTENT (compressed per-head K) + POSITION (rope
channel), scaled `1/sqrt(hd+d_rope)`. Two bias-free projections — `W_QR`
(per-head) + the **shared** `W_KR` — both RoPE-rotated. The decoupled softmax/PV
backward (shared `K^R` → `dKr` accumulates across heads) grad-checked bit-tight in
isolation + full-model + cached-vs-uncached bit-identity (`test_kv_dec`). Decode
cache = latent + shared `K^R` (`NL·T·(d_c+d_rope)·8` = **7680 B** at d_c=16/
d_rope=4, ~3.2× under MHA). v4 value-fills `pos_kind=2`+`rope_dim`; no format bump.
M12's `--pos-kind` switch complete (learned / rope / rope-decoupled).
1.2.2 — *Coupled RoPE* (M12 increment 4): `--pos-kind rope` rotates Q/K by
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

- **Cyrius pin**: `6.2.2` (in `cyrius.cyml [package].cyrius`) — bumped from
  6.2.1 in 1.3.0 to realign with the installed cycc (`cyrius update` resynced the
  `lib/` snapshot; it is byte-identical to the 6.2.1 snapshot — a clean patch
  realign — so only `cyrius.cyml` moved in the working tree; 673 checks green on
  both arches + the agnos build, no shadow/drift warnings). (1.2.4 had moved the
  pin 6.1.37 → 6.2.1.) The pin and snapshot must always move together: cycc
  6.1.32 fixed attn11's agnos argv-capture issue (r15-parked init rsp; the old
  `_agnos_init_rsp` global is gone) during M6, and a new-compiler/old-lib
  mismatch reproduces `argc()==0` under the kernel — the run gate caught it, so
  **every pin bump is followed by `cyrius update`** and a both-arches retest.
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
- **Gated linear attention** (1.4.0, default config): fwd+bwd step **~3.8 ms**
  (~6% over the dense ~3.6 ms); cached gen **~160 µs/token** (the O(hd²) state
  update beats the O(T·hd) cache scan). Decode cache **6 144 B, constant in T**
  (16× under MHA at the preset); bits/byte 0.239 vs MHA 0.279 at the same params
  (X010).
- **Selective SSM** (1.4.2, default config, N=16): fwd+bwd step **~5.6 ms** (the
  O(T·C·N) scan, ~1.56× the dense ~3.6 ms); cached gen **~258 µs/token**. Decode
  cache **12 288 B, constant in T** (8× under MHA at the preset). Best bits/byte of
  the four mixers at reference scale — **0.218** (X011).
- **Per-layer hybrid** (1.4.3, default config, mha/lin): the attention-fraction
  sweep is **parameter-identical** (39 488 each) — the hybrid only redistributes the
  decode cache: 6 144 B (0/3 attn) → 12 288 (1/3) → 18 432 (2/3) → 24 576 (3/3,
  pure MHA). bits/byte within noise across ratios (0.234–0.244, all under MHA's
  0.279). Hybrid fwd+bwd step **~3.7 ms** (≈ the linear step; two of three blocks
  linear). X012.
- **Any-mixer hybrid** (1.4.4, default config, mha ⊕ ssm, base ssm): bits/byte
  0.218 (pure ssm) → 0.224 (1/3 mha) → 0.219 (2/3) → 0.279 (pure mha) — within noise,
  near pure ssm. The decode cache is a continuous knob: 12 288 B (pure ssm, constant
  `C·N`) → 16 384 (1/3) → 20 480 (2/3) → 24 576 (pure mha, ∝T). Padding lifts the
  hybrid params to MHA's 39 488 (vs pure ssm's 38 048, +1 440). Hybrid fwd+bwd step
  **~5.0 ms** (between pure ssm ~5.6 ms and the dense step). X013.
- **MoE** (default config, 8 experts, top-2): fwd+bwd step **~6.9 ms** vs the
  dense ~3.6 ms (top-2 = two active expert MLPs + the `C→N` router), 215 648 params
  vs dense 39 488; cached gen ~273 µs/token. Per-token compute scales with `topk`,
  parameter count with `N` (X009 density sweep).
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
- **Mixture of Experts** (1.3.0, `--experts N --expert-topk K`, ADR 0008): the
  dense MLP becomes N experts (each `C→F→C`) + a bias-free router gate `C→N`;
  `--experts 1` is the byte-identical dense baseline. Forward: router logits →
  top-K (frozen lower-index tie-break) → renormalized top-K softmax → gate-weighted
  expert sum. Backward (`moe_fwd`/`moe_bwd`, `moe_aux_*` in `ops.cyr`): gradient to
  the selected logits only (straight-through pick) + the Switch load-balance aux
  loss (`α·N·Σ fᵢ·Pᵢ`, α=0.01, added to CE, off the eval path). Per-op +
  full-model grad-checked; cached decode bit-identical (MoE MLP is
  position-independent). Checkpoint v5 carries `num_experts`/`topk`. `--eval`
  reports total / per-token-active params + routing entropy. Trains + checkpoints
  + samples; the density sweep is X009 (`scripts/moe-sweep.sh`).
- **Gated linear attention** (1.4.0, `--attn-kind lin`, ADR 0009): a non-softmax
  sequence mixer — causal RetNet retention `S_t = γ_h·S_{t-1} + k_t⊗v_t`,
  `out_t = (1/√hd)·S_t^T q_t`, fixed per-head decay (parameter-free), over the MHA
  projections. New core `lin_core_fwd`/`bwd` + `attn_lin_fwd`/`bwd` (`attn.cyr`);
  hand-derived backward with no state caching (grad-checked ~1e-9). Cached decode
  (`lin_core_fwd_row`, per-layer `g_lin_state`) is the **constant** `nh·hd²` state
  — bit-identical to the batch scan. Rides `attn_kind=2` (checkpoint v5, no bump);
  full heads, learned-abs positions. Trains + checkpoints + samples (X010).
- **Selective SSM** (1.4.2, `--attn-kind ssm`, ADR 0010): a minimal Mamba-lite
  diagonal SSM — per-channel N-state `h_t = exp(Δ·A)·h_{t-1} + Δ·B·a`,
  `y = Σ C·h + D·a`, with Δ=softplus(a·W_dt)/B=a·W_B/C=a·W_C all input-dependent
  (selective). Core `ssm_fwd`/`bwd` + `ssm_fwd_row` (`attn_ssm.cyr`); hand-derived
  BPTT through the data-dependent scan (grad-checked ~1e-7). Reuses Wq (W_dt) + Wo
  (output proj) + `latent_dim` (= state N), so it rides `attn_kind=3` (checkpoint
  v5, no bump). Constant `C·N` decode cache (`g_ssm_state`); A inits to a negative
  ramp, D to 1. Best bits/byte of the four mixers at reference scale (X011).
- **Per-layer mixer hybrid** (1.4.3 rung c + 1.4.4 rung d, `--attn-every K`, ADR
  0011/0012): a full-attention (MHA) block every K-th layer, the `--attn-kind` base
  elsewhere. The global `attn_kind` becomes a per-layer `g_layer_kind` read only by
  the `_attn_block_*` dispatch helpers (`_lk(L)`). Rung c allowed {mha, gqa, lin}
  (shared `_kvw`, parameter-free); **rung d admits ANY mix of {mha, mla, lin, ssm}**
  by PADDING each block's K/V region to the max `_kvw` over the present kinds
  (`_kvw_hyb`) — uniform stride, no per-layer offset refactor, at the cost of a
  zeroed pad (and a few % params) on smaller-kind layers. `_hybrid_kinds_ok`: any
  kinds, learned-abs, full heads, a valid shared latent iff any mla/ssm.
  `kv_cache_bytes` sums the per-layer caches (mha K/V, lin/ssm constant state, mla
  latent). **Checkpoint v6** carries the per-layer pattern (`-46` on an invalid
  kind); the loader sizes the padded block via `ckpt_expected_np_kvw`; v≤5
  synthesize uniform. Mixed SSM/MLA ⊕ MHA full-model grad-check + cached-decode
  bit-identity green. Trains + checkpoints + samples (X012, X013).
- CLI: `--corpus --stdin --load --save --steps --gen-only --preset --heads
  --kv-heads --layers --attn-kind --latent-dim --attn-every --pos-kind --rope-dim
  --experts --expert-topk --bpe --eval` (`--attn-kind` takes `mha`/`mla`/`lin`/`ssm`;
  `--latent-dim` is the MLA latent / SSM state size; `--attn-every K` builds the
  per-layer hybrid over the `--attn-kind` base)

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
`--experts N --expert-topk K` (1.3.0; N ≤ 256, default topk 2) replaces the dense
MLP with N experts + a `C→N` gate: total params scale with N (215 648 at N=8 vs
39 488 dense) while per-token-active compute scales with topk; `--experts 1` is
the dense default (params unchanged). `--attn-kind lin` (1.4.0; full heads,
learned-abs) swaps the softmax core for the gated retention recurrence —
parameter-free (same 39 488 params as MHA), and the decode cache becomes the
constant `nh·hd²` state instead of a T-growing K/V.

## Source (`src/`, ~1500 LOC)

- `tensor.cyr` — attn11-local float printing (`f_print`) + `_putc`/`puts` (40
  lines); the f64-array helpers + dense matmul moved to **rosnet**, the PRNG to
  **tyche** (1.1.0 extraction)
- `ops.cyr` — layernorm, GELU (tanh approx), softmax cross-entropy (forward +
  backward); `linear_fwd`/`linear_bwd` now resolve from **rosnet**; plus the
  **MoE** router/combine (1.3.0) `moe_fwd`/`moe_bwd` (top-K pick + renormalized
  softmax + per-expert MLP, gradient to selected logits only) and the Switch
  load-balance aux (`moe_aux_fwd`/`moe_aux_dr`/`moe_aux_bwd`)
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
  latent+rope decode cache); `attn_arena_size` carries a `2·hd²` S/dS scratch for
  the gated-linear core. The pure `_kvw` per-block K/V-region size (1.4.1) is
  shared with persist's checkpoint validator (one layout source).
- `attn_linear.cyr` — the **gated-linear** mixer (1.4.0; split out of `attn.cyr`
  in 1.4.1, the one-file-per-mixer pattern): `lin_core_fwd`/`bwd` (retention
  recurrence `S_t = γ_h S_{t-1} + k_t⊗v_t`, fixed per-head decay) + `attn_lin_fwd`/
  `bwd` wrappers + `lin_core_fwd_row`/`attn_lin_fwd_row` (the constant-state cached
  decode). Included after `attn.cyr` in each entry.
- `attn_ssm.cyr` — the **selective SSM** mixer (1.4.2, ADR 0010; one-file-per-mixer):
  `ssm_fwd`/`ssm_bwd` (the data-dependent scan + the hand-derived BPTT through
  `exp(Δ·A)`) + `ssm_fwd_row` (constant `C·N`-state cached decode) + `_ssm_softplus`.
  Reuses Wq (W_dt) + Wo (output proj); A/W_B/W_C/D are the `attn_kind=3` K/V region.
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
  `model_fwd_row` skip the learned posemb add/grad under RoPE. **MoE** (1.3.0):
  `_mlp_weight_size()` branches the MLP region dense↔experts (ADR 0008, the second
  config-dependent region after `_kv_weight_size`), `_o_expert(e)`/`_o_Wgate()`
  index the experts + gate; the `_moe` forms (`model_init_moe`/`model_config_ok_moe`/
  `model_alloc_bytes_moe`) carry `num_experts`/`topk` (the `_arch` forms delegate as
  dense); the block MLP fwd/bwd + `model_fwd_row` + `model_eval_window` branch on
  `g_num_experts > 1`; `moe_entropy`/`moe_disp_*` report routing-entropy utilization.
  **Gated linear** (1.4.0): the per-layer retention state `g_lin_state` is the
  constant decode cache, and `kv_cache_bytes` reports it. **Mixer dispatch (1.4.1):**
  the mixer-kind branch lives in ONE place each — `_attn_block_fwd`/`_attn_block_bwd`/
  `_attn_block_fwd_row` — so a new mixer (SSM) touches those three, not four functions.
  **Per-layer hybrid (1.4.3 rung c + 1.4.4 rung d, ADR 0011/0012):** those three
  helpers read `_lk(L)` (the per-layer `g_layer_kind`, else the global) — uniform
  models byte-identical; `model_init_full` carries the per-layer array
  (`model_init_moe` is the uniform delegator). Rung d's `_kvw_hyb` pads each block's
  K/V region to the max `_kvw` over the present kinds (so the stride stays uniform
  for any mix), the init loop + caches gate per-layer on `_lk(L)`/`_any_kind(k)`,
  `_hybrid_kinds_ok` validates the relaxed constraints, and `model_alloc_bytes_hyb`
  takes the per-layer `kinds` pointer to count each present kind's caches
- `train.cyr` — byte + **BPE** tokenizer (`bpe_learn`/`tok_encode`/
  `bpe_build_spans`/`tok_emit`), corpus (embedded/file/stdin, raw bytes
  retained), batch sampling, LR schedule, resumable training loop, KV-cached
  generation (`gen_prime`/`gen_decode` + context-shift), `eval_corpus`
  (CE/token + bits-per-byte — pure CE, excludes the MoE aux term; accumulates the
  routing-entropy dispatch histogram)
- `persist.cyr` — validated checkpoint serialize/load (tokenizer triple +
  merge-table DAG/expansion validation + the v4 architecture descriptor, codes
  `-40..-43`, + the v5 **MoE** descriptor `num_experts`/`topk`, codes `-44`/`-45`;
  `ckpt_expected_np_moe` mirrors the MLA/decoupled/MoE layout; `pos_kind=1`
  (coupled RoPE) accepted on even-hd MHA images, `pos_kind=2`+`rope_dim` (decoupled)
  accepted on MLA images with even `2≤d_rope≤C`, `num_experts∈1..256`/`topk∈1..N`
  — all bounded before alloc; v1/v2/v3/v4 accepted, synthesizing dense MLP). **v6**
  (1.4.3): a hybrid appends NL per-layer kinds after the fixed header (`_ckpt_pl`),
  `_hybrid_kinds_ok`-validated on load (code `-46`); `CKPT_VER()` writes v6 only for
  a hybrid (uniform → v5, byte-identical); v≤5 synthesize the uniform pattern
- `main.cyr` — CLI arg parsing (incl. `--preset`/`--heads`/`--kv-heads`/
  `--layers`/`--attn-kind`/`--latent-dim`/`--attn-every`/`--pos-kind`/`--rope-dim`/
  `--experts`/`--expert-topk`/`--bpe`/`--eval`, null-guarded `_atoi`; `--attn-every K`
  builds the per-layer hybrid and dispatches to `model_init_full`) + orchestration

## Tests

- `tests/attn11.tcyr` — **907 checks**: finite-difference gradient checks
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
  + config-cap decoupled gates), the **MoE suite** (1.3.0: `test_moe_op` — per-op
  combine grad-check (dx/dWg/dWe, 1e-4) at 4 configs incl. top-1 and K=N;
  `test_moe_aux` — the load-balance aux backward vs FD (1e-5, non-uniform dispatch);
  `test_model_moe` — full-model wiring (dWe/dWgate/dWq/dtokemb, 1e-3); `test_kv_moe`
  — cached-vs-uncached bit-identity (top-1/top-2/K=N, odd T, 2-token window);
  `test_param_layout_moe` — the experts+gate tiling/aliasing pin; `test_ckpt_moe` —
  v5 round-trip + `-44`/`-45`/`-10` rejections; `test_ckpt_v4_compat`; MoE
  alloc-accounting + config-cap gates), the **SSM suite** (1.4.2: `test_ssm_core`
  per-op BPTT ~1e-7, `test_model_ssm`, `test_kv_ssm`, `test_ckpt_ssm`), the
  **per-layer hybrid suite** (1.4.3 + 1.4.4: `test_model_hybrid` — the MIXED mha/lin
  full-model grad-check ~1e-5; `test_model_hybrid_ssm`/`_mla` — the SSM/MLA ⊕ MHA
  mixed backward through the padded layout ~1e-4; `test_kv_hybrid` — cached-decode
  bit-identity across mha/lin, mha/ssm, mha/mla interleavings; `test_ckpt_hybrid` —
  the v6 per-layer round-trip (incl. a padded mha/ssm image) + `-46` rejects; hybrid
  config-cap + alloc-accounting pins for all kinds), a
  **soak/leak** test and a **NaN guard** test. All pass on x86_64 AND aarch64
  (`cyrius test`; aarch64 via qemu).
- `tests/attn11.bcyr` — benchmark harness (training timings + tokens/sec,
  generation cached/uncached, KV bytes per nkv, MQA timings, **preset
  train+gen**, **MLA latent-cache gen + the cache-bytes table** (latent vs
  MHA/MQA full-K/V), **RoPE train-step + cached-gen overhead**, **MoE train-step +
  cached-gen + param count**, **`bpe_learn` cost**).
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

See [`roadmap.md`](roadmap.md). **Shipped: v1.0.0 (clean cut) → v1.1.0
(extraction) → v1.2.0–1.2.4 (M12: MLA core, latent KV-cache decode, coupled +
decoupled RoPE; then 1.2.4 toolchain realign + docs) → v1.3.0 (M13: Mixture of
Experts) → v1.4.0 (M14 rung a: gated linear attention) → 1.4.1 (refactor sweep) →
v1.4.2 (M14 rung b: selective SSM) → v1.4.3 (M14 rung c: per-layer hybrid) → v1.4.4
(M14 rung d: any-mixer hybrids — completes M14).** The surface is frozen
([`STABILITY.md`](../STABILITY.md)) and additive-only past 1.0; the numeric core
lives in **rosnet** + **tyche**. The 1.x arc now has the attention/position axes
`--attn-kind {mha, mla, lin, ssm}` × `--pos-kind {learned, rope, rope-decoupled}`,
the FFN-density axis `--experts N --expert-topk K`, **two non-softmax mixers**
(gated linear attention + the selective SSM), and the **per-layer hybrid**
`--attn-every K` (any mix of the four). **M12, M13, and M14 are all complete.**

**Next — 1.4.5 hardening pass (P(-1)), then M15+.** With four feature releases
stacked (1.4.0–1.4.4, incl. checkpoint v6, per-layer dispatch, and the padded
hybrid layout), 1.4.5 is a consolidation/hardening pass: security/correctness audit
(input handling, the v6 load path, buffer/bounds), a benchmark baseline refresh, a
deep review of the 1.4.x complexity, and a doc/ADR audit (file in `docs/audit/`).
Then E5–E6 (diffusion objective / ternary) as M15–M16 and **M17** reinforcement
learning (E9). A vidya-scale bake-off across mixers AND hybrid ratios is the
standing X-entry (X012/X013 ran the ratio sweeps at reference scale).

### Handoff — how to pick this up

- **Build/test/release**: `make check` (lint + x86 grad-checks) for the fast loop;
  `make release` (lint + x86 + aarch64/qemu + DCE build + fuzz) is the full gate
  before tagging — both must exit 0. Quick refs at the top of
  [`CLAUDE.md`](../../CLAUDE.md) (Quick Start). `cyrius deps` resolves rosnet/tyche
  (pinned in `cyrius.lock`); a no-flag `./build/attn11` trains + samples.
- **The discipline that matters**: every hand-derived backward lands behind a
  finite-difference grad-check (`cyrius test`); kernel changes land in BOTH the
  batch and the cached single-row path or neither (the bit-identity contract,
  [`../architecture/003`](../architecture/003-cached-inference-bit-contract.md));
  the pin and `lib/` snapshot move together (`cyrius update` after any bump, then
  retest both arches); additive-only past 1.0 (new flags + a new checkpoint
  version with permanent back-compat). New backward op → grad-check it first,
  then plumb (the M12.5 increments are the worked example: per-op grad-check in
  isolation, then model wiring, then the cached path + bit-identity).
- **Where things live**: live state here; forward plan in
  [`roadmap.md`](roadmap.md); shipped narrative in [`../../CHANGELOG.md`](../../CHANGELOG.md);
  experiment evidence in [`experiments.md`](experiments.md); decisions in
  [`../adr/`](../adr/); non-obvious invariants in [`../architecture/`](../architecture/);
  the frozen surface in [`../STABILITY.md`](../STABILITY.md).

**Loose ends / known items**: (1) an `attn11` row upstream in
`agnos/scripts/stage-tools.sh` (`stage_one attn11 src/main.cyr attn11`) — a
cross-repo edit on the agnos maintainer's side, not actionable here. (2) The
**MLA absorption** compute optimization (fold `W_UK` into `W_Q` to attend latents
directly, avoiding the per-step K/V re-materialization in 1.2.1/1.2.3) is deferred
— it reorders accumulation, so it would ride its own bit-identity story; the MoE
combine has an analogous folding option but the same caveat. (3) A perplexity
bake-off (decoupled vs coupled vs learned MLA, and MoE density) on the vidya
corpus is the natural next X-series entry now that M13 has landed (X009 ran the
density sweep on the embedded corpus; the vidya-scale run is the follow-on). (4)
The MoE aux coefficient α is fixed at 0.01 (no `--aux-alpha` flag); an α sweep is
a small additive follow-on if it earns one. The pin is now **6.2.2** (realigned
1.3.0, byte-identical `lib/` snapshot); the cycc argv-capture issue is resolved
upstream (6.1.32; `docs/architecture/002` retired).
