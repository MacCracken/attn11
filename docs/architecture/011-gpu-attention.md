# 011 — GPU fused attention (the full forward on-device)

> **Location (1.10.0):** this backend was extracted whole to **rosnet** (`dist/rosnet-gpu.cyr`, ADR
> 0017); `src/gpu.cyr` below is its historical attn11-local path. The constraints are unchanged.

**Status:** current (M18 1.8.4). Builds on
[008 (matmul / the 256-id cap + host-tiling)](008-gpu-matmul-spirv.md),
[009 (layernorm / tiled reductions)](009-gpu-layernorm-and-the-transcendental-wall.md),
[010 (in-shader f64 exp / the `--gpu-tc` tolerance gate)](010-gpu-transcendentals.md).

## What's true

`gpu_attn_core` (`src/gpu.cyr`) runs the **causal multi-head attention core** — the body of
`attn_core_fwd` — on the native-AMD f64 SPIR-V path: QK scores, the causal softmax, and the PV
weighted-sum. It is hooked at `attn_core_fwd` (the one source of truth for the per-head
arithmetic, shared by MHA/GQA/MLA) and rides **`--gpu-tc`** (the in-shader `exp` makes it a
tolerance op, ~1.5e-13 vs the CPU oracle — never plain `--gpu`). With attention on-device, the
entire forward (matmuls + layernorm + GELU + LM head + attention) can run on the GPU; the CPU
`attn_core_fwd` remains the reference.

## Why four passes, not one fused kernel

A single per-(head, query) kernel would, for each of the ≤T keys, do a head-dim dot **and** an
in-shader `exp` (~25 ids each) plus the running softmax — ~900 SSA ids for T=16. That is far
past mabda's **256-id compile cap** (008). So attention is decomposed into four
host-orchestrated passes over the `Qc`/`Kc`/`Vc` the QKV matmuls already produced, using `Pc`
(the `nh·T·T` score buffer) as scratch:

| Pass | Grid | Work | Tiled? |
|------|------|------|--------|
| 1 scores | `nh·T·T` | `P[h,i,j] = scale·Σ_d q[i,h,d]·k[j,h,d]`, causal-masked | no (hd-unrolled) |
| 2 rowmax | `nh·T` | `M[h,i] = max_j P[h,i,j]` (softmax shift) | yes (over j) |
| 3 exp+sum | `nh·T` | `P[h,i,j] := exp(P−M)`, `L[h,i] = Σ_j` (`_gpu_emit_exp`) | yes (over j) |
| 4 PV | `nh·T·hd` | `concat[i,h,d] = (1/L)·Σ_j P[h,i,j]·V[j,h,d]` (RMW) | yes (over j) |

Between passes the host pre-inverts `L := 1/L` (so pass 4 multiplies, no in-shader FDiv) and,
after pass 4's RMW accumulation, does the final `concat *= 1/sum` normalize. Reductions are
**j-tiled** (`_gpu_attn_tk = 4`, one dispatch per tile) so each kernel's id_bound stays well
under the cap regardless of `T`.

## The causal mask (three constraints stacked)

1. **No integer compares.** `gfx9_compile` lowers `OpSelect` and the *floating* ordered
   compares (`OpFOrd*` → VOPC), but `_spirv_alu_to_mir` does **not** map standalone integer
   compares — an `OpULessThanEqual` returns `MIR_ERR_UNSUPPORTED_OP` (−20). So the `j≤i` test is
   built in floating point: `OpConvertSToF(i)`, `OpConvertSToF(j)`, `OpFOrdGreaterThanEqual`,
   feeding `OpSelect`.
2. **A finite −∞ sentinel.** Masked entries (`j>i`) are stored as **−1e8**, not 0 and not a true
   −∞. Zero would win the pass-2 row max when all real scores are negative; a magnitude past
   ~1e8 overflows the **i32** in `exp`'s range reduction (`ConvertFToS(round(x·log₂e))`),
   producing garbage instead of 0. −1e8 is far below any real attention score yet
   `exp(−1e8 − m)` underflows cleanly to 0 — so **only pass 1 masks**; passes 2–4 process the
   full row uniformly and the sentinels vanish (max ignores them, `exp→0`, PV adds nothing).
3. **Synth-id budget.** Decomposing the 1-D `GlobalInvocationId.x` into `(h,i,j)` needs integer
   div/mods, and each `UDiv`/`UMod` expands to a Newton-reciprocal macro whose **hidden** ids
   count against the 256 cap. A full-T-unrolled PV (id_bound 175) + 2 `UDiv` overflowed
   (`MIR_ERR_ID_OOR`, −25). Two fixes together: tiling every reduction drops the explicit
   id_bound to ~80, and host-side `1/L` removes the FDiv macro. (Replacing one `UMod` with
   `idx − (idx/hd)·hd` also trades a heavy macro for a cheap `mul`+`sub`.)

A **3-D dispatch** (`grid = (hd, T, nh)`, reading `gid.x/y/z` to skip the div/mods entirely)
was tried and **rejected**: mabda's native dispatch does not populate the `y`/`z` workgroup-id
SGPRs (uninitialised TGID → wild indices → it wrote the SPIR-V `"main"` bytes into the output).
Stay 1-D; pay for the decomposition with tiling.

## Scope and fallback

The GPU path handles **causal MHA only**: `nkv == nh` (so `kvb == base`, `Ckv == C`) and
`g_bidir == 0`. GQA (`nkv < nh`, the strided kv-head mapping), bidirectional diffusion
(`g_bidir == 1`), and any `T` not divisible by the tile `TK` self-fall-back to the CPU core
(`gpu_attn_core` returns −1; the caller in `attn_core_fwd` runs the CPU body). The bidir gate
lives in the caller so `src/gpu.cyr` stays free of `g_bidir` (keeps the standalone gpu tests
linkable). Default (T16·C32·nh4) and preset (T64·C64·nh8) are both plain causal MHA → on-device.

Buffers reuse the 7-BO GTT pool (009) by role: **Q=x, K=W, V=γ, P=β, M=S, L=V, concat=y**. `P`
is the largest (`nh·T·T·8` = 256 KB at preset) and fits the 4 MB per-BO budget.

## Why tolerance (not bit-exact), and what `--gpu` still guarantees

The softmax's `exp` is the in-shader polynomial (010), not the x86 hardware `exp`, so attention
is **tolerance** — it joins GELU and the LM head behind `g_gpu_tc`/`--gpu-tc`. Plain `--gpu`
does **not** run attention on the GPU, so a plain-`--gpu` run stays **byte-identical** to the
no-flag run (matmul + layernorm only, both bit-exact). A `--gpu-tc` run tracks the CPU to
~1.5e-13 per op; over many training steps that compounds into a small trajectory drift in
bits/byte (expected — it is not a reproducible/checkpoint path, by construction).

## Pointers

- Kernels + host orchestration: `gpu_attn_core`, `_gpu_attn_scores/_rowmax/_expsum/_pv`,
  `_gpu_pre_a` in rosnet's
  [`src/gpu.cyr`](https://github.com/MacCracken/rosnet/blob/main/src/gpu.cyr) → `dist/rosnet-gpu.cyr`
  (extracted at 1.10.0, ADR 0017; vendored into attn11 as `lib/rosnet-gpu.cyr`).
- Validation: [`tests/gpu_attn.cyr`](../../tests/gpu_attn.cyr) (`make gpu-test`).
- On-hardware proof + the bug trail (causal mask, synth overflow, 3-D dead end): experiment
  **X031** in [`docs/development/experiments.md`](../development/experiments.md).
