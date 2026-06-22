# Architecture notes

Non-obvious constraints, quirks, and invariants that a reader cannot derive from the code alone. Numbered chronologically — never renumber.

Not decisions (those live in [`../adr/`](../adr/)) and not guides (those live in [`../guides/`](../guides/)). An item here describes *how the world is*, not *what we chose* or *how to do something*.

## Items

- [001 — Tensors and floats](001-tensors-and-floats.md) — everything is i64; an
  `f64` is its bit pattern; tensors are flat row-major f64 arrays; the SIMD
  matmul convention (memory accumulators, never reassign a SIMD var); and the
  toolchain gotchas (long-literal mis-parse, byte-counting `print`, f64
  comparisons not being NaN-correct). **Affects**: all numeric code in `src/`.
- [002 — agnos entry epilogue](002-agnos-entry-epilogue.md) — **retired at
  pin ≥ 6.1.32** (upstream fix: r15-parked init rsp). Was: `r = main()`
  must be a top-level *statement*, never a `var` initializer: on agnos, a
  call-bearing gvar init runs before the compiler-emitted argv capture, so
  `main()` would see `argc()==0` and silently ignore every CLI flag.
  Residual rule: the pin and the `lib/` snapshot must move together.
  **Affects**: every entry file (`src/main.cyr`, `src/test.cyr`,
  `tests/attn11.{tcyr,bcyr,fcyr}`, `docs/examples/minimal_train.cyr`).
- [003 — The cached-inference bit contract](003-cached-inference-bit-contract.md) —
  every KV-cached row path (`attn_fwd_row` for MHA/GQA, `attn_mla_fwd_row` for
  MLA, both via `attn_core_fwd_row`, with RoPE riding inside) must stay
  arithmetically identical, per row, to its batch path; what that forbids when
  changing kernels. **Affects**: `src/attn.cyr`, `src/model.cyr`, `src/ops.cyr`
  — any forward-kernel change, and every new `--attn-kind`/`--pos-kind`.
- [004 — Harness include sets](004-harness-include-sets.md) — the bench
  (`tests/attn11.bcyr`) is the only entry point that omits `persist.cyr`, and
  cyrius compiles an undefined function to a runtime *trap* (not a build
  error), so a symbol `model.cyr` references must not live only in
  `persist.cyr` — rebuild **and run** all four harnesses after touching shared
  `src/`. **Affects**: `src/model.cyr`, `tests/attn11.bcyr`.
- [005 — RoPE's portable trigonometry](005-rope-portable-trig.md) — `f64_sin`/
  `f64_cos` are x86-only builtins (no aarch64 polyfill), so coupled RoPE builds
  cos/sin from a Maclaurin series (`θ_k ∈ (0,1]`, no range reduction) + complex
  binary-exponentiation to the integer position — computed directly from the
  position so the batch and cached-row paths stay bit-identical (note 003).
  **Affects**: `src/attn.cyr` (`rope_apply_*`, `_rope_unit_cossin`, `_rope_pow`).
- [006 — The packed corpus token store](006-packed-token-store.md) — `g_data` is
  the only corpus-sized buffer, so it alone is packed: one **u8** per token for
  byte-level (vocab ≤ 256) or **u16** for BPE (vocab ≤ 768), set by the `g_data_w`
  width global before the corpus load, via width-generic accessors (`gd_ld`/`gd_st`,
  `_tw_ld`/`_tw_st`). Everything else (the `g_T`-sized batch buffers, the vocab-sized
  BPE tables) stays i64 — the asymmetry is deliberate. This removed the 8×/4× bloat
  and raised `MAX_CORPUS_BYTES` 4 MB → 64 MB. **Affects**: `src/train.cyr` (`g_data`,
  `corpus_set`, `bpe_learn`, `tok_encode`, the window samplers). Shipped 1.5.3 (X018).
- [007 — Streaming token shards](007-streaming-token-shards.md) — the RAM-independent
  large-corpus path: `--stream-corpus` samples windows by **lseek+read at offset** through a
  single 64 K-token chunk cache inside `gd_ld` (not `mmap` — the lib wrapper is x86_64-only and
  agnos has no file-backed mmap). Because `gd_ld` is the only corpus-token reader, the model /
  samplers are untouched and stream-vs-in-memory byte-identity is structural. **Affects**:
  `src/stream.cyr`, `src/train.cyr` (`gd_ld`). Shipped 1.6.2.

> **The GPU backend notes (008–011) describe rosnet's `[lib.gpu]` backend** (`dist/rosnet-gpu.cyr`),
> **extracted whole from attn11 at 1.10.0** (ADR 0017). They were written M18/M19 when the backend was
> attn11-local, so they say `src/gpu.cyr` for that historical path — the SPIR-V / tiling / precision
> constraints they record are unchanged by the relocation. Future GPU-kernel work lands in rosnet.

- [008 — GPU matmul in SPIR-V](008-gpu-matmul-spirv.md) — what the generated-SPIR-V matmul
  kernel must obey: `gfx9_compile` takes **straight-line only** (no loops / phi) and caps at
  **256 ids**, so the dot is unrolled + **host-tiled** (`GPU_TK=16` RMW over `k`); the reserved-VA
  buffer layout; why the result is **bit-exact** vs CPU `linear_fwd`; the AGNOS dead-code gating.
  **Affects**: rosnet's GPU backend, `src/ops.cyr` (`qlinear_fwd` seam). M18 1.8.0.
- [009 — GPU layernorm and the transcendental wall](009-gpu-layernorm-and-the-transcendental-wall.md)
  — why `ln_fwd` is GPU-**bit-exact** (sequential reductions + sqrt/div only) but softmax / GELU /
  head are not (f64 `exp` has no bit-exact SPIR-V form; SIMD tree-reduction order differs) — the
  finding that split plain `--gpu` (bit-exact) from `--gpu-tc` (tolerance). M18 1.8.1.
- [010 — GPU transcendentals](010-gpu-transcendentals.md) — the hand-rolled in-shader f64 `exp`
  (`_gpu_emit_exp`: Cody-Waite + Taylor-Horner + `ldexp`, ~2.3e-13 vs CPU), GELU over it, and the
  `--gpu-tc` **tolerance gate** (`allclose atol=rtol=1e-10` + the statistical loss-tracks check)
  that protects the byte-identical `--gpu` invariant. M18 1.8.2–1.8.3.
- [011 — GPU attention](011-gpu-attention.md) — the fused causal-MHA core as **4 host-orchestrated
  passes** (a single per-(head,query) kernel exceeds the 256-id cap): the causal mask via
  float-compare + `OpSelect` (gfx9 lowers no integer compares), the finite **−1e8** −∞ sentinel,
  and the synth-id budget that forces the tiling. **Affects**: rosnet's GPU backend, `src/attn.cyr`
  (`attn_core_fwd` seam). M18 1.8.4.
