# GPU compute backend (`--gpu`) — M18

attn11's forward matmul can run on the GPU. It is **opt-in** (`--gpu`), the CPU stays the
reference oracle, and the GPU result is **bit-exact** vs the CPU — so a `--gpu` run produces
the same numbers (and the same checkpoint) as the default run. The GPU is an *execution
target*, not a different model.

> **Where the backend lives (1.10.0, ADR 0017):** the whole GPU backend was extracted to
> **[rosnet](https://github.com/MacCracken/rosnet)** as a mabda-gated `[lib.gpu]` profile
> (`dist/rosnet-gpu.cyr`) — attn11 consumes it via `[deps.rosnet]`; there is **no `src/gpu.cyr` in
> attn11 anymore**. Future GPU-kernel work (int8, the Nvidia bring-up) lands in rosnet, once, for every
> consumer. The `--gpu` surface and its byte-identity are unchanged — the move was a pure relocation.

## Requirements

- An **AMD GFX9 GPU** reachable as a DRM render node (`/dev/dri/renderD128`, no root needed
  if the node is mode 0666). The dev reference is an **AMD Cezanne (gfx90c)** APU.
- The toolchain resolves **rosnet 0.2.0** (the GPU backend bundle `dist/rosnet-gpu.cyr`) **+ mabda
  3.4.1** (`cyrius deps`); mabda's native SPIR-V→GFX9 f64 emitter is what runs the kernel —
  launcher-free, pure-Cyrius, no ROCm / Vulkan loader / vendor BLAS. (mabda's symbols ship unresolved in
  the rosnet bundle; attn11's `[deps.mabda]` supplies them — both auto-prepend on Linux targets.)

No GPU? `--gpu` prints that no device was found and runs entirely on the CPU. Everything
still works; nothing crashes.

## Use

```sh
cyrius build src/main.cyr build/attn11
./build/attn11 --gpu --steps 200            # forward matmul on the GPU
```

On start `--gpu` reports the device (`native AMD compute device online` / `f64 SPIR-V
compute supported`); after the run it reports how many forward matmuls actually dispatched
on-device, so a silent CPU fallback is never mistaken for a GPU run.

`--gpu` runs (all **bit-exact** vs the CPU oracle — a `--gpu` run is byte-identical to the
no-flag run, *including the optimizer state*):
- **matmul** at the `qlinear_fwd` seam — Q/K/V + output projection, MLP up/down (~80% of
  FLOPs) — 1.8.0.
- **layernorm** (`ln_fwd`) — run ~2×/layer — 1.8.1.
- **the Adam step** (`model_adam_step`) — the per-parameter optimizer update, the first
  backward-arc op; in-shader f64 `sqrt`/`div` are correctly-rounded so it is bit-exact — 1.8.5.
- **the LM-head backward** (`head_bwd`) — the tied-embedding gradients (D_f + gemb); the CPU
  head_bwd is pure-scalar sequential so the GPU tiles match bit-for-bit — 1.8.8 (engages for **any
  vocab** since 1.9.0 tail-tiled the D_f V-contraction; only `T%16≠0` still falls back).
- **the layernorm backward** (`ln_bwd`) — dx + dgamma/dbeta from the saved mean/rstd; rstd is a
  precomputed input so the reductions are sequential (no transcendental) → bit-exact — 1.8.9.
- **the RoPE Q/K rotation** (`rope_apply_fwd`/`bwd`, `--pos-kind rope`) — the angles depend only on
  `(channel-pair, position)`, so the host precomputes the cos/sin table (the trig stays on the CPU)
  and the GPU does only the rotation (f64 mul/add/sub) → bit-exact — 1.9.1 (decode rows=1 falls back).

**`--gpu-tc`** (implies `--gpu`) additionally runs **GELU** (1.8.2), the **LM head** (1.8.3,
`logits = f · tokembᵀ`), and the **full fused attention core** (1.8.4 — QK scores + the causal
softmax + the PV weighted-sum) at a *tolerance* (~1e-13), not bit-exact — GELU and the softmax
because the in-shader f64 `exp` ≠ the x86 hardware `exp`; the head because its sequential GPU
reduction differs from the CPU's SIMD-tree dot order. To keep plain `--gpu` byte-identical, these
are behind this **separate** gate: a `--gpu-tc` run tracks the CPU run to ~1e-13 (loss + eval
bits/byte match to print precision, never NaN) but is **not** byte-identical. It also runs the
**GELU / linear / attention backward** (1.8.6 / 1.8.7 / 1.8.10) at tolerance.

**The full training step runs end-to-end on the GPU (M18 milestone, 1.8.10) — at default AND preset
(1.9.0).** Every heavy op of a step — forward + backward + Adam — executes on-device: a
`--bpe 7 --gpu-tc` step dispatches matmuls + layernorms + Adam + head-bwd + ln-bwd (bit-exact) and
GELU + head + attention + linear-bwd + attn-bwd (tolerance), all on the GPU, no NaN. **1.9.0** tiled
the attention backward over T (`TK=16` RMW chunks → the old `T ≤ 20` gate is gone) and tail-tiled the
head's vocab, so a `--preset --gpu-tc` step (T=64) now runs end-to-end on-device too.

The attention core (`gpu_attn_core` at the `attn_core_fwd` seam) is **4 host-orchestrated
passes** — scores (`nh·T·T`) → rowmax → exp+sum → PV — because a single fused per-(head,query)
kernel exceeds the 256-id compile cap; it covers **causal MHA** (`nkv==nh`, `g_bidir==0`).

**Non-default model axes:** **`--ternary`** already runs on the GPU (it quantizes `W→W_eff` then
takes the same `qlinear→gpu_matmul` path — no separate kernel); **`--pos-kind rope`** runs its Q/K
rotation on the GPU bit-exact (1.9.1, above); **`--experts` (MoE)** runs its attention/projection
matmuls on the GPU, but the **routed expert MLPs stay on the CPU** — they are M=1 per-token-per-expert
matmuls (re-uploading each expert's weights `T·K`×/step would be a poor GPU fit; a batched-per-expert
gather is the deferred proper path).

What stays on CPU: **GQA / bidirectional (diffusion) attention** (fall back to the CPU core), the
**MoE routed expert MLPs** (above), and **very wide heads** (`hd > 32` — the untiled `dp` pass would
exceed the build budget, so the attention backward self-falls-back; `--preset --heads 1` is the corner
case). Everything else of a training step runs on the GPU at default and preset (RoPE since 1.9.1,
preset-scale since 1.9.0). Each op
self-falls-back to CPU if the device is absent, the contraction
dimension isn't a multiple of `GPU_TK` (16) / the tile `TK` (4), or a buffer/dispatch limit is
exceeded — all of attn11's default/preset shapes (matmul K ∈ {32,64,128,256}; ln C ∈ {32,64};
GELU any width; head C ∈ {32,64}; attention T ∈ {16,64}) run on-device.

## Verify it

```sh
make gpu-test     # build + run tests/gpu_matmul.cyr
```

This builds + runs the GPU validation suite — `tests/gpu_matmul.cyr` (matmul) and
`tests/gpu_ln.cyr` (layernorm) + `tests/gpu_adam.cyr` (the Adam step) + `tests/gpu_head_bwd.cyr`
(LM-head backward) + `tests/gpu_ln_bwd.cyr` (layernorm backward) bit-exact, plus
`tests/gpu_gelu.cyr`, `tests/gpu_head.cyr`, `tests/gpu_attn.cyr` (the fused attention core),
`tests/gpu_gelu_bwd.cyr` (GELU backward), `tests/gpu_linear_bwd.cyr` (linear backward — dx tolerance,
dW/db bit-exact), and `tests/gpu_attn_bwd.cyr` (the 6-pass attention backward) at tolerance —
validating each against its CPU
oracle (`linear_fwd`/`ln_fwd`/`gelu_fwd`/`head_fwd`/`attn_core_fwd`) across attn11's real shapes,
proving the device actually engaged, and checking the shape-not-tileable CPU fallback. On a box
with no AMD GPU they **skip cleanly** (exit 0) — so it is a standalone target, *not* part of
`make release` (the CI gate runs without a GPU). The end-to-end oracle check is byte-identity of
the checkpoints:

```sh
./build/attn11 --steps 40 --save /tmp/cpu.ckpt
./build/attn11 --gpu --steps 40 --save /tmp/gpu.ckpt
cmp /tmp/cpu.ckpt /tmp/gpu.ckpt && echo "identical"   # GPU forward is bit-exact
```

## Honest scale

This is the **sovereign GPU path working + validated**, not a speedup. At attn11's size
(default ≈39 K params), per-call host↔device transfer plus `ceil(K/16)` serialized
dispatches dominate; the native f64 path is **scalar VALU** (no matrix core — Cezanne has no
full-rate `V_MFMA_F64`), so it is a correctness/oracle path. A throughput win needs a much
larger model and/or matrix-core f64 hardware (a future mabda capability).

## How it works (pointers)

- Backend + kernel generator: rosnet's
  [`src/gpu.cyr`](https://github.com/MacCracken/rosnet/blob/main/src/gpu.cyr) → `dist/rosnet-gpu.cyr`
  (the `[lib.gpu]` profile, ADR 0017; vendored into attn11 as `lib/rosnet-gpu.cyr`). The architecture
  notes below describe this backend; they say `src/gpu.cyr` for its historical attn11-local path
  (M18/M19) — the constraints are unchanged by the relocation.
- The non-obvious constraints (no SPIR-V loops/phi, the 256-id compile cap → host-tiling,
  the reserved-VA layout, why matmul is bit-exact):
  [`../architecture/008-gpu-matmul-spirv.md`](../architecture/008-gpu-matmul-spirv.md).
- Layernorm (3-pass tiled reduction) + the transcendental wall:
  [`../architecture/009-gpu-layernorm-and-the-transcendental-wall.md`](../architecture/009-gpu-layernorm-and-the-transcendental-wall.md).
- The in-shader f64 `exp`, GELU, and the `--gpu-tc` tolerance gate:
  [`../architecture/010-gpu-transcendentals.md`](../architecture/010-gpu-transcendentals.md).
- The full fused attention core (4 passes, the causal-mask-without-int-compares + finite-−∞
  sentinel + synth-id-budget findings):
  [`../architecture/011-gpu-attention.md`](../architecture/011-gpu-attention.md).
- The boundary decision (kernels are rosnet's GPU backend layered on mabda, never in mabda)
  + the f32→f64 sequencing (inverted to f64-first on AMD):
  [ADR 0016](../adr/0016-gpu-backend-layered-on-mabda.md).
- On-hardware f64 proof + integration: experiments **X025 / X026 / X027**.
