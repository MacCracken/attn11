# attn11 — Experiments Ledger

> Running log of training experiments and improvement probes — settings,
> numbers, takeaways. The frontier track in [`roadmap.md`](roadmap.md)
> (E1–E6) graduates items *into* milestones; this file is where their
> evidence lives. Append-only, numbered chronologically — never renumber.
> Reproducibility: every entry records the tag/commit, corpus, and exact
> flags; checkpoints make runs bit-resumable (`--load`).

## X001 — vidya Cyrius corpus, first real-content training (2026-06-11)

**Setup**: 0.6.0 (default config: d_model 32, ctx 16, 4 heads, 3 layers).
Corpus = `find vidya/content -name '*.cyr' | sort | xargs cat` → 488,489
bytes, 74 files (one per topic: allocators, compression, state machines, …).
Byte-level vocab adapted to **117** symbols (vs 25 embedded-default);
params 39,488 → **42,432**. Run: `--corpus … --steps 8000 --save …`.

**Result**: loss `ln(117) ≈ 4.76` (random) → **~1.55** at step 8000;
curve still noisy-downward at the cosine floor (not converged, not
memorizing — the corpus is ~12× larger in entropy terms than anything
attn11 had seen). Wall time ≈ 8 min on x86_64 (~4.5k tok/s).

**Samples**: greedy output is syntactically valid Cyrius — guard-clause
idiom (`if (i < 0) { return 0; }`), brace/semicolon discipline, 4-space
indentation. Temp-0.8 reproduces section-divider comments (`# ====…`) and
call shapes. Words wobble; *texture* is right.

**Takeaways**:
1. `--corpus` + adaptive vocab worked unmodified on first real content —
   the M2 surface is adequate for the M6 example pipeline.
2. The binding constraint is **ctx 16** (fragments, not statements) →
   motivates roadmap E3 (ctx 64 / d_model 64 preset).
3. A real-entropy corpus makes loss curves *meaningful* — use vidya, not
   the embedded string, as the baseline corpus for E-track comparisons
   (E3 byte-vs-BPE, E4 hybrid-vs-transformer, E5 AR-vs-diffusion).

Checkpoint: `/tmp/vidya-cyr.ckpt` (ephemeral; regenerate with the flags
above — deterministic, so the run reproduces bit-for-bit).

## X002 — KV-cache + GQA gate measurements (E1/E2 → M6) (2026-06-11)

**Setup**: 0.7.0 (default config: d_model 32, ctx 16, 4 heads, 3 layers,
biases on; toolchain pin 6.1.33), embedded corpus, x86_64, greedy decode,
500 generated tokens (`tests/attn11.bcyr`).

**E1 result**: uncached generation (full window forward per token)
1 050 579 ns/token; KV-cached decode (one row per token + context-shift
re-prime every T/2 = 8 tokens) **170 392 ns/token — 6.16×**, i.e. 951 →
**5 868 tokens/sec**. Bit-identity gate green (logits identical at every
prefix and across context-shifts, greedy + temp, hd ∈ {4,6,8,10} ×
nkv ∈ {1,2,nh}).

**E2 result**: KV cache bytes 24 576 (`nkv=4`) → 12 288 (`nkv=2`) → **6 144
(`nkv=1`)** — linear in `nkv`, the 4× architectural cut at MQA. MQA also
trims compute: fwd+bwd step 3 680 703 → 3 456 801 ns (~6%, the K/V
projection shrink), cached generation 170 392 → 164 282 ns/token.

**Takeaways**:
1. At ctx 16 the cached path is re-prime-bound (a window recompute every 8
   tokens); the per-token win grows with T — E3's ctx-64 preset should
   roughly double the speedup ratio.
2. GQA's KV saving is real but only *matters* at scale (here the whole cache
   is 24 KB); its value in attn11 is the grad-checked reference backward.
3. Training quality at `nkv < nh` is untested on a real corpus — a vidya
   iso-param GQA-vs-MHA run is the natural X003 (pairs with E3).
