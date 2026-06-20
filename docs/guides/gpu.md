# GPU compute backend (`--gpu`) — M18

attn11's forward matmul can run on the GPU. It is **opt-in** (`--gpu`), the CPU stays the
reference oracle, and the GPU result is **bit-exact** vs the CPU — so a `--gpu` run produces
the same numbers (and the same checkpoint) as the default run. The GPU is an *execution
target*, not a different model.

## Requirements

- An **AMD GFX9 GPU** reachable as a DRM render node (`/dev/dri/renderD128`, no root needed
  if the node is mode 0666). The dev reference is an **AMD Cezanne (gfx90c)** APU.
- The toolchain resolves **mabda 3.4.1** (`cyrius deps`), mabda's native SPIR-V→GFX9 f64
  emitter is what runs the kernel — launcher-free, pure-Cyrius, no ROCm / Vulkan loader /
  vendor BLAS.

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

What runs on GPU: the matmul at the `qlinear_fwd` seam — Q/K/V and the output projection,
and the MLP up/down (~80% of FLOPs). The tied-weight LM head, the attention core, and the
`lin`/`ssm` mixers stay on CPU. Backward + Adam stay on CPU until 1.8.2. Each matmul
self-falls-back to CPU if the device is absent, the contraction dimension `K` is not a
multiple of `GPU_TK` (16), or a buffer/dispatch limit is exceeded — all of attn11's
default/preset matmul shapes (K ∈ {32,64,128,256}) run on-device.

## Verify it

```sh
make gpu-test     # build + run tests/gpu_matmul.cyr
```

This validates `gpu_matmul_fwd` against the CPU `linear_fwd` oracle across attn11's real
shapes (bit-exact), proves the device actually engaged, and checks the K-not-tileable CPU
fallback. On a box with no AMD GPU it **skips cleanly** (exit 0) — so it is a standalone
target, *not* part of `make release` (the CI gate runs without a GPU). The end-to-end
oracle check is byte-identity of the checkpoints:

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

- Backend + kernel generator: [`src/gpu.cyr`](../../src/gpu.cyr).
- The non-obvious constraints (no SPIR-V loops/phi, the 256-id compile cap → host-tiling,
  the reserved-VA layout, why it's bit-exact):
  [`../architecture/008-gpu-matmul-spirv.md`](../architecture/008-gpu-matmul-spirv.md).
- The boundary decision (kernels are rosnet's GPU backend layered on mabda, never in mabda)
  + the f32→f64 sequencing (inverted to f64-first on AMD):
  [ADR 0016](../adr/0016-gpu-backend-layered-on-mabda.md).
- On-hardware f64 proof + integration: experiments **X025 / X026 / X027**.
