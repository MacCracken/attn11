# 0016 — GPU compute backend layered on mabda (not in it); precision-tiered f32→f64

**Status**: Accepted
**Date**: 2026-06-14

## Context

M18 (E-infra) adds a GPU *compute* backend for attn11's hand-derived forward/backward
— an execution target, not a new dependency. A mabda recon + an AMD-GPU f64/sovereign
research pass (2026-06-14) established the facts that force the design:

- **mabda is a GPU *foundation*, not a math library.** It provides device/context
  init, buffers (create/upload/synchronous readback), and a compute pipeline (WGSL →
  module → bind-group layout → `compute_dispatch`), with caches + a ping-pong buffer.
  It vendors **no** matmul/BLAS/softmax/attention kernels. The wgpu backend
  (Vulkan/Metal/DX12) works today; the native-AMD (DRM/PM4/GFX9) compute path is
  deferred upstream. mabda's stated purpose is graphics + generic compute.
- **rosnet is attn11's numeric core** (tensor / BLAS-1 / matmul + gradient), extracted
  to a shared crate at 1.1.0. The hot ops funnel through a clean seam: matmul (~80% of
  FLOPs) via `qlinear_fwd/bwd` over rosnet `linear_fwd/bwd`.
- **Precision: the f32 wall is WebGPU's, not AMD's.** WGSL has no f64 (gpuweb #2805,
  deferred), so the portable wgpu/WGSL path is f32-only. But **f64 is reachable on AMD**
  — via Vulkan-compute + SPIR-V (`shaderFloat64`; wgpu's `SHADER_F64`, native-Vulkan
  only, slow, possibly raw SPIR-V) or native CDNA MFMA (`V_MFMA_F64`, full-rate on
  Instinct; consumer RDNA is FP64-throttled). attn11 is f64 everywhere, CPU
  grad-checked at maxrel 1e-5.
- **Sovereign invariant.** attn11 links **no** vendor BLAS/cuDNN/autodiff. AMD's
  Composable Kernel / CK Tile and rocBLAS are MIT-licensed and may be *studied* (not
  linked) as kernel-design references; the AMD ISA PDFs + the LLVM AMDGPU backend give
  a ROCm-runtime-free assembler path.
- **Constraint:** `lib/` is vendored/gitignored (regenerated from the pin); attn11
  cannot modify `lib/mabda.cyr` or `lib/rosnet.cyr` in-repo.

The open questions: *where do the GPU tensor kernels live, at what precision, in what
order?*

## Decision

1. **The GPU tensor kernels are rosnet's GPU backend, layered ON mabda's primitives —
   NOT in mabda.** GEMM/attention/softmax/layernorm/gelu/Adam are tensor *math*
   (rosnet's domain); putting them in mabda would contaminate a graphics/compute-
   primitive foundation. mabda is **consumed, never modified**; its only M18-related
   work is the **device-layer** Vulkan-f64 capability (on mabda's own v3.2 roadmap).
   Because `lib/` is vendored, the kernels are **built attn11-local first**
   (`src/gpu*.cyr`, consuming `lib/mabda.cyr`), then **extracted** to rosnet's GPU
   backend once proven — the exact path the CPU core took to rosnet at 1.1.0. The
   GPU-vs-CPU oracle lives next to the f64 CPU reference it validates.

2. **Precision is tiered, CPU stays the reference.** The portable wgpu/WGSL path runs
   **f32**, gated by a new **dual-precision** test (GPU-f32 op vs CPU-f64 op within an
   f32 tolerance, ~1e-3..1e-5) **plus** a statistical gate (loss descends / never
   NaNs). The **f64 oracle is restored** on the f64-capable paths (Vulkan-SPIR-V, then
   native-CDNA MFMA). The CPU scalar/SIMD path remains the f64 reference and the
   byte-identical no-flag default; the GPU is `--gpu`-gated.

3. **Sequencing is LOCKED: f32 portable first, then the f64 track.**
   - **1.8.0** GPU foundation + matmul (portable f32, the working wgpu backend) →
     **1.8.1** rest of the forward (f32) → **1.8.2** backward + Adam (f32) + an honest
     GPU-vs-CPU perf X-entry.
   - **then the f64 track** (later 1.8.x / M19, gated on mabda 3.x's Vulkan-f64): re-run
     the op set on Vulkan-SPIR-V f64 to restore the f64 oracle; native-CDNA `V_MFMA_F64`
     for full-rate f64 once mabda's native compute dispatch matures.

### Alternatives considered

- **Put the kernels in mabda** — rejected: contaminates a graphics/compute-primitive
  library with ML math; the math is rosnet's domain (and the f32-vs-f64 oracle belongs
  with the f64 CPU reference, i.e. in rosnet, not the device layer).
- **Link a vendor BLAS (rocBLAS / hipBLASLt / Composable Kernel) as a dependency** —
  rejected: violates the sovereign "no BLAS / no autodiff" invariant. Instead these
  MIT/Apache libraries are *studied* as kernel-design + ISA references, never linked.
- **f64-first (skip the f32 portable path)** — rejected as the *start*: f64 needs the
  mabda 3.x Vulkan capability, is native-Vulkan-only, and is slow on consumer GPUs.
  Starting on the working f32 wgpu backend proves the rosnet-GPU layer + dispatch +
  validation harness cheapest; f64 is the destination, not the entry point.
- **A new standalone `gpu` crate beside rosnet** — rejected in favor of **rosnet's GPU
  backend**: rosnet *is* the tensor-math core, so CPU (SIMD) and GPU math stay unified
  there, with mabda as the device layer underneath both.

## Consequences

- mabda stays a clean, reusable GPU foundation (no ML baggage); the GPU tensor layer is
  itself reusable (rosnet's GPU backend, for any consumer — the rosnet model).
- M18 introduces exactly **one** new discipline: f32-tolerance dual-precision validation
  on the portable path. The f64 track returns to the bit-tight f64 oracle every prior
  milestone holds.
- A **cross-repo dependency** on mabda 3.x (the Vulkan-compute f64 device capability)
  gates the f64 track — recorded on mabda's own v3.2 roadmap (2026-06-14).
- **Open risk** (deferred to the native f64 follow-on): mabda's native-AMD path uses raw
  PM4-over-graphics-DRM, which is under-documented (Mesa/RADV + AMD PAL headers are the
  next references); the documented HSA/amdkfd AQL path is a different submission route.
- The no-flag CPU run stays byte-identical (the GPU is `--gpu`-gated); the prior f64
  grad-check suite is untouched. Sources / scoping: `docs/development/roadmap.md` (M18,
  the 1.8.x mini-arc) + the 2026-06-14 mabda recon and AMD-GPU research.
