# Audit — 0.7.0 (M6: KV-cached generation + GQA), 2026-06-11

**Scope**: the full 0.7.0 diff — E2 (GQA/MQA: `nkv`-parameterized attention,
Ckv-dependent parameter layout, checkpoint format v2 with v1 compat) and E1
(KV caches, `attn_fwd_row`/`model_fwd_row`/`model_eval_window`, the
context-shift sampler) plus the new tests/bench. Method: adversarial
multi-agent review (5 dimensions — gradient math, memory/arena safety,
persistence/hostile input, bit-identity/determinism, integration — each raw
finding independently judged by 3 verifier lenses instructed to refute;
≥2/3 upholds). 50 agents; 15 raw findings → 9 confirmed, 6 refuted.

## Confirmed findings → fixes

1. **[medium] A fully shape-valid hostile checkpoint could crash the loader
   via allocation failure** (two vectors: `np` up to the old 64M cap → the 4
   parameter vectors alone exceed the allocator's 256 MB; and the `nh·T·T`
   attention-arena term, which is independent of `np` — e.g. C=4, nh=4,
   T=8192 passes every shape check in an ~800 KB file but asks for a ~2 GB
   arena). `alloc()` returns 0 past its cap and `t_alloc` zero-filled from
   address 0 → SIGSEGV, violating persist.cyr's "rejected, never a crash"
   contract. **Fixed**: new `model_alloc_bytes()` (exact mirror of
   `model_init`'s allocations, sync-pinned by `test_alloc_accounting`)
   checked in `ckpt_load_buf` before any allocation (reject `-18`,
   `CKPT_MAX_MODEL_BYTES` 128 MB); `CKPT_MAX_NP` 64M → 4M;
   `CKPT_MAX_BYTES` 2 GB → 128 MB; `ckpt_load_file`/`ckpt_save_file`
   null-check their buffers (`-31`/`-21`); `t_alloc` aborts cleanly on
   allocation failure (defense in depth — a controlled abort, never a wild
   write); corpus loaders null-check (`-4`). Regression:
   `test_ckpt_gqa_roundtrip` ("oversized model (arena) rejected pre-alloc").

2. **[medium] `model_init` trusted its config** — `nkv` not dividing `nh`
   makes `kvb` walk past the Ckv-wide K/V rows (silent arena/KV corruption,
   wrong gradients, no error); `nkv > nh` divides by zero. Only the
   checkpoint path validated. **Fixed**: `model_config_ok()` enforced at the
   top of `model_init` (clean abort on programmer error; the loader still
   rejects hostile headers with `-16` first).

3. **[low] `_rng_state = 0` restorable from a checkpoint** — the xorshift64
   fixed point (stream sticks at 0 forever); `rng_seed` itself guards it, so
   0 can only come from a hostile image. **Fixed**: reject `-19`; regression
   in `test_ckpt_gqa_roundtrip`.

4. **[low] FD grad checks are structurally blind to parameter-layout
   aliasing** (forward and backward index through the same `_o_*` helpers, so
   overlapping offsets perturb and accumulate identically — FD stays green
   while the model silently trains a different architecture). **Fixed**:
   `test_param_layout` pins that every offset equals the previous tensor's
   end and the block tiles `_blk()` exactly, at a `Ckv ≠ C` config.

5. **[medium+low] The E1 bit-identity gate only ran at hd=8** — the SIMD
   scalar-tail loops in `attn_fwd_row` and the true `1 < nkv < nh` group
   mapping were never exercised by the cached-vs-uncached compare (verifiers
   probed those configs and found no live divergence — test gap, not a bug).
   **Fixed**: the gate now runs at hd ∈ {4, 6, 8, 10}, nkv ∈ {1, 2, nh},
   odd T (odd `GEN_SHIFT`), and a degenerate 2-token window.

6. **[low] Verification records (not bugs)**: the GQA backward was
   re-derived independently and matched line-by-line (grouped dK/dV
   accumulation, softmax backward, scale chain, all linear dims); the 13
   arena sub-pointers tile `attn_arena_size` exactly for all `nkv`; the
   `dbk ≈ 0` softmax-shift-invariance pin is mathematically right;
   `ckpt_expected_np` reduces to the v1 formula at `nkv = nh`.

## Refuted findings (recorded so they aren't re-litigated)

- `gen_prime`/`gen_decode` bounds claims (0/3 ×4): all callers guarantee
  `1 ≤ n ≤ T`; the slide invariant keeps `G_n < T` at every store; an
  unprimed decode isn't reachable from the public surface.
- "Fuzz can't reach the model_init alloc path" (1/3): true but moot once the
  pre-allocation bound rejects before init; the bound itself is regression-
  tested directly.
- "Mid-decode logits only pinned via sampled ids" (0/3): the per-prefix bit
  compare plus final-row bit compare brackets the decode loop; temperature
  sampling through the cumulative scan makes a mid-stream logit divergence
  surface in the id stream with overwhelming probability.

## Toolchain finding (from the run gate, not the review)

The M5 AGNOS run gate failed on first 0.7.0 execution: `--steps/--save`
silently ignored under the booted kernel (`argc()==0` symptom). Root cause:
**toolchain drift, not 0.7.0 code** — installed cycc 6.1.33 parks the agnos
init rsp in r15 at the entry landing (the 6.1.32 fix for the issue attn11
filed on 2026-06-10), but the vendored `lib/` snapshot at the old 6.1.31 pin
still read the removed `_agnos_init_rsp` global, which a 6.1.32+ compiler
never sets. Linux unaffected (different args path). **Fixed**: pin bumped
`6.1.31 → 6.1.33`, `lib/` re-synced; all gates re-run green, the run gate
passes (see CHANGELOG). The `docs/architecture/002` statement-call epilogue
workaround is no longer required at pin ≥ 6.1.32 (kept in code — harmless).

## Result

161 checks green on x86_64 and aarch64 (qemu); fuzz green (500 mutated
checkpoints + 100 random corpora, now over the v2 header incl. nkv/step/rng
fields); agnos build + run gate green at pin 6.1.33. All confirmed findings
fixed and regression-tested in this release.
