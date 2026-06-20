# 008 — The GPU matmul: generated SPIR-V, host-tiled, on mabda's native-AMD f64 path

*What's true about `src/gpu.cyr` (M18, v1.8.0). Non-obvious constraints a future
change must not violate.*

## The working compute path is the LOW-LEVEL native one, not portable wgpu

mabda exposes two compute paths. The portable **wgpu/WGSL** path
(`compute_pipeline_new`, `gpu_compute_dispatch` over the wgpu backend) **does not run
compute** on this toolchain: `MABDA_WGPU_COMPUTE = 0` and the wgpu compute-dispatch slot
is a `GPU_ERR_OTHER` stub (ADR 0016, X025). The path that actually executes is
mabda's **native-AMD** route: `gpu_context_new_native()` → `gpu_shader_module_create_spirv`
(which runs the in-tree `gfx9_compile` SPIR-V→GFX9 emitter) → `gpu_compute_dispatch`
(native slot) over PM4/DRM. attn11 uses that path. WGSL is never authored; the kernel is
**SPIR-V we generate word-by-word** in `_gpu_build_tile`.

## `gfx9_compile` accepts STRAIGHT-LINE SPIR-V only — no loops, no phi

mabda's emitter rejects `OpLoopMerge` and `OpPhi` (`LOWER_ERR_CONTROL_FLOW`): a back-edge
breaks its forward linear-scan register allocator, and the MVP carries no merge-consumed
(phi) values. Forward `OpSelectionMerge` + `OpBranch`/`OpBranchConditional` lower, but a
value cannot cross a merge. **Consequence:** the matmul dot product cannot be a loop — it
is **unrolled**. The accumulator is a forward SSA chain (`acc_{k+1} = OpFAdd acc_k prod_k`),
never a loop-carried phi, which is exactly what the linear-scan accepts.

## The public native-compile path caps a module at 256 ids → HOST-TILE the contraction

`_native_shader_compile_spirv` (mabda) sizes its compile scratch for
`NATIVE_SHADER_CAP_IDS = 256` ids (also 512 MIR instrs, 8192-byte ISA). A full-K unroll
needs id_bound ≈ `8·K`, so K≈24 is the ceiling — too small (attn11 K reaches 128/256).
So the contraction is **host-tiled**: the kernel does a bounded `GPU_TK = 16`-term unroll
(id_bound ≈ 160) that **read-modify-writes** `y` (`acc = y[idx]; acc += Σ x·W; y[idx] = acc`),
and the host loops `k0` in steps of `GPU_TK`, dispatching `ceil(K/GPU_TK)` times. The host
**pre-fills `y` with the bias** (or 0) before the first tile, so every tile is a uniform
RMW with no first-tile special case. Dispatches are **serialized** (the native dispatch
waits for completion), so there is no race on `y`. **K must be a multiple of `GPU_TK`**;
other K cleanly falls back to CPU (attn11's default/preset shapes — 32/64/128/256 — are all
multiples of 16). Raising `GPU_TK` past ~28 would breach the 256-id cap.

## One thread per output element, 1-D dispatch (only `GlobalInvocationId.x`)

The kernel is dispatched as a 1-D grid of `M·N` workgroups of `LocalSize (1,1,1)`. Each
thread reads `idx = GlobalInvocationId.x`, then `m = idx / N` (`OpUDiv`), `n = idx % N`
(`OpUMod`); the output index is `idx` itself. mabda expands `GlobalInvocationId =
WorkgroupId·localsize + LocalInvocationId` and wires WorkgroupId via TGID SGPRs, so 1-D
multi-workgroup dispatch is the safe, proven shape (the e2e conformance kernels also use
the in-workgroup id). 2-D dispatch (using `gid.y`) would rely on TGID_Y enablement that
1.8.0 does not exercise. `M·N` must be ≤ `MABDA_MAX_DISPATCH_DIM` (65535).

Indices advance **incrementally** inside the unroll (`xi += 1`, `wi += N`), so the kernel
needs only the constants `{0, 1, K, N, k0, k0·N}` regardless of `GPU_TK`. `k0`/`k0·N` are
baked per tile, so a tile kernel is keyed by `(K, N, k0)` and cached (linear-scan, ≤128
entries) — each distinct tile compiles once.

## GTT buffers are CPU-visible; the data VAs must dodge mabda's reserved regions

`native_bo_create_gtt` returns `{handle, host_map}` — GTT memory is host-mapped (Cezanne
APU unified memory), so "upload" = `store64` into the map and "readback" = `load64`, no
copy. The three persistent data buffers (x, W, y; 4 MB each, allocated once in `_gpu_init`)
are mapped at **fixed GPU VAs `0x…110/111/112000000`**, chosen clear of mabda's reserved
native VAs: IB `0x…100200000`, **fence `0x…100600000`**, render targets `0x…101000000`
(16 MiB), textures `0x…180000000` (2 GiB). A 4 MB buffer based at `0x…100400000` would
straddle the fence VA — that is why the data band is up at `0x…11x000000`.

## Bit-exact, so a `--gpu` run reproduces the CPU run

The kernel accumulates in the **same k-order** as rosnet's `linear_fwd` (sequential
`y += x·W`), and uses separate `OpFMul`+`OpFAdd` (two roundings) — which on GFX9 scalar
`V_MUL_F64`/`V_ADD_F64` rounds identically to `f64_mul`/`f64_add`, and `linear_fwd`'s SIMD
`f64v_fmadd` is itself bit-identical to that scalar sequence. So `gpu_matmul_fwd` is
**bit-exact** vs `linear_fwd`, and a `--gpu` training run produces a **byte-identical
checkpoint** to the no-flag run (verified). The GPU is an execution target; the CPU stays
the oracle.

## Scope (1.8.0) and where it bites

Only the **forward** matmul at `qlinear_fwd` is on GPU (QKV / O / MLP up+down — the bulk
of FLOPs). The tied-weight LM head (`head_fwd`) and the attention-core / SSM / linear
mixers are not at this seam and stay CPU. Backward (`qlinear_bwd`) stays CPU until 1.8.2.
`--gpu` self-falls-back per shape, so partial coverage is invisible to correctness.

The cost: `qlinear_fwd` now references `gpu.cyr`, so every harness that includes `ops.cyr`
(main, `.tcyr`, `.fcyr`, `.bcyr`) pulls in `lib/mabda.cyr` (~1 MB dist → binary ~373 KB →
~1.34 MB). That is a transient binary-SIZE cost pending cyrius `dep-module-call`; it never
gated M18. With `g_gpu == 0` the path is the unchanged CPU code (byte-identical).

## AGNOS: mabda auto-prepends on EVERY target — gate the consumer, stub the one symbol

cyrius auto-prepends a declared dep's `modules` (`[deps.mabda] modules = ["dist/mabda.cyr"]`)
into **every** build, including `--agnos` — and there is no target-conditional dependency
syntax, nor a way to fetch a dep without auto-prepending it (dropping `modules` stops the
fetch entirely). So `#ifndef CYRIUS_TARGET_AGNOS` around an *explicit* `include "lib/mabda.cyr"`
does nothing: mabda is injected regardless (the explicit include was redundant — removed).

mabda is **Linux-only** (it `syscall(SYS_IOCTL, …)` for DRM), and the AGNOS ring-3 syscall
peer omits `SYS_IOCTL`, so the auto-prepended dep failed to compile for `--agnos`. The fix,
given the constraints (can't modify vendored `lib/`, can't make the dep conditional): on
AGNOS the entire GPU path is already gated out (`#ifndef CYRIUS_TARGET_AGNOS` on the
`src/gpu.cyr` include, the `--gpu` flag, and the `g_gpu`/`gpu_matmul_fwd` refs in
`qlinear_fwd`), so mabda's ioctl code is **dead code** there — it only needs to *type-check*.
A one-line `#ifdef CYRIUS_TARGET_AGNOS  var SYS_IOCTL = 16;  #endif` (in `main.cyr` and each
harness, before the includes) satisfies the reference; cyrius resolves top-level globals
whole-program, so the def is found even though mabda is prepended ahead of it. On Linux the
stdlib syscalls peer owns `SYS_IOCTL`, so the `#ifdef` avoids a duplicate definition; on
aarch64-Linux mabda compiles unchanged (`SYS_IOCTL` is in the aarch64 syscall peer). Result:
AGNOS builds (main + grad-check/fuzz/bench suites) are clean static ELFs, GPU-free.
