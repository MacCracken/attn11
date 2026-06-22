# 0017 — Extract the GPU backend to rosnet (whole-backend)

**Status**: Accepted
**Date**: 2026-06-21

## Context

[ADR 0016](0016-gpu-backend-layered-on-mabda.md) built the GPU compute backend
**attn11-local first** (`src/gpu.cyr`, consuming `lib/mabda.cyr`) with an explicit plan to
**extract it to rosnet's GPU backend once proven — "the same path the CPU core took to rosnet
at 1.1.0."** That condition is now met: M18 (v1.8.0–1.8.11) + M19 (v1.9.0–1.9.2) shipped the full
training step (forward + hand-derived backward + Adam) on the native-AMD f64 SPIR-V→GFX9 path,
validated bit-exact (matmul/ln/Adam/head-bwd/ln-bwd) or to ~1e-13 (the transcendental / SIMD-tree
ops) against the CPU oracle, with the perf honestly characterised (an oracle/coverage path, not a
speedup, on the Cezanne mobile APU — benchmarks B4/B5/B6, X032/X039/X042).

`src/gpu.cyr` is ~2200 lines in three strata: (1) a reusable **GPU foundation** — lazy native-AMD
device/ctx init, a 7-buffer GTT pool (host-mapped, CPU-visible), the hand-emitted SPIR-V primitives
(`_gpu_op*`, `_gpu_pre`/`_gpu_pre_a`, `_gpu_acl6`, `_gpu_ecf`, `_gpu_emit_exp`), the tiled-matmul
builders/getters (`_gpu_build_tile`/`_t`/`_dw`, `_gpu_get_tile*`), a shader cache, and the dispatch
helpers; (2) **generic tensor/NN ops** — `gpu_matmul_fwd`, `gpu_ln_fwd`, `gpu_gelu_fwd`/`bwd`,
`gpu_linear_bwd`, `gpu_ln_bwd`, `gpu_adam_step`; (3) **transformer-specific ops** — `gpu_attn_core`/
`bwd`, `gpu_head_fwd`/`bwd`, `gpu_rope_apply`.

rosnet is the extracted CPU numeric core (tensor/matmul + gradient), a shared AGNOS crate with
**other consumers** beyond attn11; its CPU `[lib]` bundle (`dist/rosnet.cyr`) must stay dependency-
clean. mabda is the GPU *foundation* (device/buffers/dispatch), **Linux-only** (DRM/PM4). The forced
decision: **where to cut the boundary when extracting.** The transformer ops lean heavily on the
shared foundation (all 7 BOs, the emit primitives, the tile builders, the cache, dispatch), so a
"clean" generic→rosnet / transformer→attn11 split would have to expose roughly half of `gpu.cyr` as
rosnet's public GPU API — a large surface, committed now, for an as-yet-hypothetical second consumer.

## Decision

**Extract the *whole* `src/gpu.cyr` (foundation + generic + transformer ops) into rosnet as a
mabda-gated `[lib.gpu]` profile (`dist/rosnet-gpu.cyr`), at rosnet 0.2.0; attn11 1.10.0 consumes it
and calls the public `gpu_*` functions from its seams.** rosnet's CPU `[lib]` bundle is unchanged and
**mabda-free**; only a consumer that opts into `dist/rosnet-gpu.cyr` pulls mabda. The GPU bundle ships
with mabda symbols **unresolved** (the consumer-included-bundle pattern `cyrius distlib` already
expects); attn11 keeps `[deps.mabda]` to supply them and adds the rosnet-gpu module. The relocated
kernels are byte-for-byte unchanged — **the gate is that `--gpu` stays byte-identical and `--gpu-tc`
stays within tolerance** across the move.

Whole-backend is chosen over the cleaner generic/transformer split for **simplicity and lower risk**:
the unit moves cohesively (pure relocation, no API carving), and the rosnet↔attn11 boundary can be
refined later (re-home the transformer ops to attn11 once a real second consumer pins down the
foundation API) without re-litigating the move. The transformer-specific kernels temporarily living
in rosnet's GPU bundle is the accepted cost.

## Consequences

- **Positive** — the proven GPU backend becomes a reusable AGNOS asset (any consumer gets GPU tensor
  ops); one source of truth for the kernels; attn11 slims toward only its transformer seams + CLI;
  ADR 0016's extraction promise is honoured; future mabda multi-vendor work (the Nvidia bring-up)
  lands once, in rosnet, for everyone.
- **Negative** — rosnet's GPU bundle holds transformer-specific kernels (attention/RoPE/head) — a
  boundary blur a generic tensor crate would rather not have; rosnet gains a Linux-only optional
  bundle + a transitive mabda surface for GPU consumers; cross-repo coupling tightens (attn11 1.10.0
  pins rosnet 0.2.0; a kernel fix is now a rosnet change + release before attn11 can take it).
- **Neutral** — a follow-up ADR may re-home the transformer ops to attn11 (the clean boundary) once
  the foundation API has a second consumer to design against; rosnet's GPU tests need mabda, wired so
  the CPU `[lib]` bundle is never contaminated (mirroring attn11's AGNOS-gating of the mabda include).

## Alternatives considered

- **Clean boundary (generic→rosnet, transformer→attn11)** — the ADR-0016-intended design and the
  right long-term split, but it forces exposing the GPU foundation as a large public API *now*, for
  an uncertain second consumer. Rejected as premature; explicitly revisitable (see Neutral).
- **Foundation-only (emit/device/tile infra → rosnet; all ops stay attn11)** — leaves the generic
  tensor ops (`matmul`/`ln`/`gelu`/`adam`/…, unambiguously rosnet's domain) stranded in attn11. A
  half-measure that satisfies neither boundary.
- **Stay attn11-local (don't extract)** — abandons the ADR 0016 plan and leaves the backend
  un-reusable. Rejected; the extraction was called now that the backend is proven.

See the mirror decision from rosnet's side:
[rosnet ADR 0001](https://github.com/MacCracken/rosnet/blob/main/docs/adr/0001-adopt-gpu-backend.md).
