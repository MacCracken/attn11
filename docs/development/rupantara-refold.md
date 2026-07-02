# attn11 → consume rupantara (kill the forward divergence)

**Status:** ✅ **DONE for the leaf ops (2026-07-02).** attn11 consumes rupantara's
`ru_*` leaf forward ops; the full grad-check suite is green in one binary.
Cross-repo with rupantara.

## Why

`rupantara` is the transformer **forward** extracted from attn11 (for anukūlana /
the murti load-seam). It *duplicated* attn11's CPU forward math → two copies that
drift. The extraction's whole point is a **single source of truth**; attn11 must
consume rupantara or the split made things worse. The **GPU** forward is already
the single shared source (`rosnet-gpu`, from M18) — only the **CPU** math was
doubled, so this is a bounded change.

## What was done (attn11 side)

- Added `[deps.rupantara]` (git + local `path = "../rupantara"` + tag,
  `dist/rupantara.cyr`) and `include "lib/rupantara.cyr"` after `lib/rosnet.cyr`
  in `src/main.cyr` **and** in the three test entries (`tests/attn11.{tcyr,fcyr,bcyr}`,
  which each carry their own include chain — the grad-check gate builds from
  `tests/attn11.tcyr`, not `main.cyr`).
- Routed the **CPU bodies of three pure leaf ops** to rupantara's `ru_*`, keeping
  every GPU / diffusion branch attn11-local:

| attn11 op (src) | delegates to | condition |
|---|---|---|
| `ln_fwd` (`ops.cyr`) | `ru_ln_fwd` | below the `if (g_gpu==1)` guard |
| `gelu_fwd` (`ops.cyr`) | `ru_gelu_fwd` | below the `if (g_gpu_tc==1)` guard |
| `attn_core_fwd` (`attn.cyr`) | `ru_attn_core_fwd` | **`g_bidir == 0`** (causal); the diffusion body stays attn11-local |

- Backward, objectives (AR/diffusion), MTP, MoE, dropout, RoPE, the row/decode
  path, and `model_forward` orchestration **stay in attn11** — only the three leaf
  CPU bodies are sourced from rupantara. The forward caches (`mean`/`rstd` from
  `ru_ln_fwd`, the `Pc` probabilities from `ru_attn_core_fwd`) are the same ones
  attn11's backward already consumes.

## NOT delegated (deliberate) — the plan's old "same signatures, drop-in" claim was wrong

- **`embed_fwd` / `head_fwd` / `head_fwd_row`** have **incompatible signatures**:
  attn11's forms are 0-arg / 2-arg and read globals (`A_tokens`, `P_tokemb`,
  `P_posemb`, `g_C`, `g_V`, …); rupantara's take explicit pointers (6-arg / 5-arg).
  Delegating them would need per-op adapter shims and buys little (they are thin
  composition, not leaf math, and carry training-time behavior). Left attn11-owned.
- **`attn_arena_size`** — attn11's arena includes the **backward** temporaries +
  gated-linear scratch and is LARGER than rupantara's forward-only
  `ru_attn_arena_size`. attn11 keeps its own; it must **never** adopt rupantara's
  (a smaller arena → backward writes past the allocation → heap corruption). This
  is why rupantara's copy is renamed `ru_attn_arena_size`, not shadowing.
- **Cyrius silently shadows duplicate `fn`s** (`warning: duplicate fn … last
  definition wins`, build still succeeds) — so the collision surface (36 symbols,
  8 public + 28 private) had to be renamed in full on rupantara's side *before*
  linking, or the build would have been green while running the wrong bodies.

## Result

The three CPU **leaf** ops (LayerNorm, GELU, causal attention core) are now a
**single source** (rupantara `ru_*`); the GPU forward is a single source
(`rosnet-gpu`). The composition/embed/head ops and the bidir/row/MLA/SSM/MoE/MTP
CPU paths remain attn11-local and are **not** de-duplicated by this re-fold — an
honest, bounded win, not "no divergence" across the whole forward.

## Gate (the live parity proof) — GREEN

`make check` (fmt + lint + `cyrius test`) = **1049 passed, 0 failed** in one
binary, CPU, with the three leaf ops delegated. No offline dump/compare hack. This
is the proof that rupantara's leaf forward feeds attn11's backward faithfully.
