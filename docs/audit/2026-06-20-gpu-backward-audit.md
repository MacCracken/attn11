# Audit — M18 GPU backward + full-step close-out (1.8.5–1.8.11), 2026-06-20

Scope: the M18 GPU backend (`src/gpu.cyr`), focused on the **backward + Adam surface** added across
1.8.5–1.8.10 (the optimizer step, GELU/linear/head/layernorm/attention backward) and the shared
infra (the `_gpu_build_tile_dw` accumulating matmul-bwd, the `_gpu_acl6`/`_gpu_ecf6` helpers, the
6-pass attention backward). The forward surface (1.8.0–1.8.4) was audited at its cuts; this pass
re-reviews the device-buffer / index / gate surface now that the full training step runs on-device,
and records the bugs the per-op bit-exact tests caught during the arc.

## Verdict: GO — 0 residual blockers

Every heavy op of a training step runs on the GPU, gated and self-falling-back to the CPU oracle.
The GPU path touches **no untrusted input** — it operates only on the model's own f64 tensors
(params/grads/activations), never on file/network/arg data (those are parsed + validated on the CPU
before any tensor exists). All four bugs found during the arc were caught by the bit-exact /
allclose tests and fixed in-cut; none reached a release with wrong numbers (the byte-identity gate
on plain `--gpu` is the tripwire). The CPU remains the production path; the GPU is the validated
oracle / sovereign-stack milestone.

## Findings → dispositions (4 caught + fixed during the arc; 0 residual)

1. **Layering / `undefined g_NP` (1.8.5→1.8.6, fixed).** `gpu_adam_step` read model globals
   (`g_NP`/`g_params`/…) directly, so `src/gpu.cyr` depended on `src/model.cyr` and every standalone
   gpu test broke. It hid because `make release` doesn't build the gpu harnesses. Fixed: pass the
   buffers as params (`gpu_adam_step(params, grads, m, v, NP, lr, step)`); gpu.cyr is self-contained.
   **Hardening:** `make gpu-test` is now run after any `src/gpu.cyr` change (the release gate won't
   catch standalone-test breakage).
2. **Accumulation order (1.8.8, fixed).** Reusing dW's RMW-onto-uploaded-seed path for head_bwd's
   `gemb` gave a 68% bit-mismatch — `head_bwd` is `seed + pure-Σ` (not RMW-from-seed like
   `linear_bwd`). Fixed: zero-init + host-add. The bit-exact test caught it.
3. **`CMP_ERR_FLAT_NONVGPR` (1.8.9, fixed).** gfx9 rejects a load with a uniform (SGPR) address;
   `ln_bwd`'s loop-indexed loads (`gamma[c]`, `mean/rstd[m]`) were uniform. Fixed: taint the index
   with a VGPR-zero (`gid − gid`).
4. **`MIR_ERR_UNSUPPORTED_TYPE` (1.8.10, fixed).** `_gpu_ecf` hardcodes the f64-const result-type
   `%7` (= double in the attn preamble, = runtimearray in `_gpu_pre` where double is `%6`). Fixed:
   `_gpu_ecf6` (`%6`) for `_gpu_pre` kernels.

## Buffer / index / bounds review (this pass — clean)

- **Device-buffer bounds.** Every backward kernel's `OpAccessChain` index is bounded by the dispatch
  shape: linear-bwd dx/dW grids ≤ `M·K`/`K·N`; head-bwd `D_f`/`gemb` ≤ `T·C`/`V·C`; ln-bwd ≤ `M·C`;
  attn-bwd dP/sds ≤ `nh·T·T`, dQ/dV/dK ≤ `nh·T·hd`, and the dV/dK **gather** loops `i = 0..T-1` with
  P index `h·T·T + i·T + j ≤ nh·T·T` and the operand index `i·C + base + d ≤ T·C` — all in range. The
  pool BOs are 4 MB each and every op gates `bytes ≤ GPU_BO_MAX()` and `grid ≤ MABDA_MAX_DISPATCH_DIM`.
- **The unmasked gather is sound.** The dV/dK gather loops the full `i` range relying on
  `Pc[h,i,j]=0` for `j>i` (the forward zeroes the upper triangle); the masked terms contribute 0, so
  no out-of-causal-range gradient leaks. Pinned by `tests/gpu_attn_bwd.cyr` (dQ/dK/dV 0-diff).
- **256-id cap discipline.** The untiled attn-bwd reductions (D/G: 8 ids/iter + 2 UDiv synth) fit the
  cap only for small T → gated `T ≤ 20` (default engages; preset falls back to the CPU core). Tiled
  ops (matmul/ln-bwd) stay bounded for all shapes. No kernel is dispatched that didn't compile
  (`_gpu_get_*` returns 0 → host falls back).
- **Accumulator preservation.** `+=` gradients (dW/dgamma/dbeta/gemb) either RMW onto the uploaded
  running grad (matching the CPU's interleaved `+=` order, bit-exact) or zero-init + host-add
  (matching `seed + pure-Σ`) — per each CPU op's own order; verified 0-diff by the tests.

## Syscall / dependency surface

- The GPU path rides mabda's native-AMD DRM ioctls (Linux-only). It is `#ifndef CYRIUS_TARGET_AGNOS`-
  gated end-to-end (flag, include, all `g_gpu*` consumers), and the auto-prepended dep type-checks on
  AGNOS via the one-line `SYS_IOCTL` stub. AGNOS main+tcyr build clean; the GPU path is dead code there.
- No `sys_system`, no path traversal, no external-data trust on the GPU path.

## Cleanliness (P(-1))

`cyrius lint src/*.cyr` → 0 warnings. `cyrius test` → 1056 grad-checks pass (x86_64 **and**
aarch64/qemu — the CPU path is unchanged; all GPU ops are flag-gated). `make gpu-test` → 11/11
(matmul/ln/Adam/head-bwd/ln-bwd bit-exact; gelu/head/attn/gelu-bwd/linear-bwd/attn-bwd tolerance).
AGNOS main+tcyr build clean. `make fuzz` + `make smoke` green. Plain `--gpu` checkpoint
byte-identical to the no-flag run (the bit-exact invariant, the primary tripwire). Benchmark
baseline X039 / B5 (the honest 4–7× full-step loss).

## Coverage notes (known, documented, not blockers)

- Preset-scale attention backward is untiled (`T ≤ 20`) → T=64 falls back to the CPU core. Tiling is
  a follow-up; correctness is unaffected (CPU fallback).
- The tolerance ops (GELU/head/attention fwd+bwd, linear-bwd dx) are ~1e-13 vs the CPU's
  SIMD-tree/x86-exp; they ride `--gpu-tc` and never affect the byte-identical plain-`--gpu` path.
- GFX9-only (other GPUs unverified); scalar VALU f64 (correctness, not speed).
