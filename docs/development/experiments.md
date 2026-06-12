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

## X003 — byte vs BPE at iso-compute (E3 → M7) (2026-06-11)

**Setup**: 0.7.1, `--preset` (d_model 64, ctx 64, 8 heads, 4 layers), the X001
vidya corpus (488,489 bytes, 74 files), x86_64, seed 1337, `--eval`
bits-per-byte. The comparison is held at **iso-compute**, not iso-step: each
config's total training MACs are matched, anchored on byte-level = 3000 steps.
Per-token MAC ≈ `12C² + 2TC + CV` (ADR 0006); the `CV` (tied-head) term grows
with the BPE vocab `V = 117 + K`, so a higher-`K` config does more work per
step and is given proportionally fewer steps. Derived horizons: byte 3000,
bpe-128 2664, bpe-256 2395, bpe-512 1993 — total MACs matched to <0.03%.
`bits/byte` (cross-entropy nats / ln 2, normalized by the **decoded byte**
count, BPE targets weighted by their span length) is the tokenizer-comparable
metric; eval byte counts are ~equal (488.3–488.5 K) across all four, so the
numbers compare directly.

**Result** (lower bits/byte = better):

| tokenizer | V | params | steps | corpus tokens | ce/token (nats) | **bits/byte** |
|-----------|-----|---------|-------|---------------|-----------------|---------------|
| byte      | 117 | 211 648 | 3000  | 488 489 (1.00×) | 1.323 | **1.909** |
| bpe-128   | 245 | 219 840 | 2664  | 269 779 (1.81×) | 2.130 | **1.697** (−11.1%) |
| bpe-256   | 373 | 228 032 | 2395  | 229 067 (2.13×) | 2.487 | **1.683** (−11.8%) |
| bpe-512   | 629 | 244 416 | 1993  | 192 434 (2.54×) | 2.907 | **1.652** (−13.4%) |

At matched compute, **BPE reaches lower bits-per-byte than byte-level, and the
advantage grows with merge count K** (−11% to −13%). `ce/token` *rises* with K
(each token now carries more bytes, so it is harder to predict) while
bits-per-*byte* *falls* — the metric correctly credits BPE for covering more
of the corpus per token. Byte-level itself reached train loss **1.17** here
(vs X001's 1.55 at the tiny config on the same corpus), so the preset's larger
model is independently the better predictor — X001 takeaway #2 (ctx-16 was the
binding constraint) confirmed.

**Takeaways**:
1. The frontier survey's **data-efficiency thesis holds on vidya**: at
   iso-compute, subword tokens spend the fixed MAC budget on more of the
   corpus's bytes, so BPE predicts the stream ~11–13% more cheaply (bits/byte).
   The gain is monotone in K across [128, 512] but with **diminishing returns**
   (−11.1 → −11.8 → −13.4%): most of it is captured by K=128.
2. **Caveat — iso-compute, not iso-parameter.** A larger BPE vocab inflates the
   embedding + tied head (211 K → 244 K params), so BPE also buys capacity.
   The MAC accounting prices the `CV` term in, so the comparison is fair on the
   *compute* axis the question posed; an iso-param sweep (hold params, vary K
   by shrinking C) is a separate future probe.
3. Methodology pin: `--eval` is RNG-neutral and the merge training is pure i64
   (bit-reproducible cross-arch), so every row above reproduces bit-for-bit
   from the flags — re-run with `--corpus <vidya> --preset [--bpe K] --steps S
   --eval`.

Checkpoints: ephemeral (no `--save`); regenerate deterministically with the
flags above.

## X004 — M9 perf-lever profiling (which levers actually pay) (2026-06-11)

**Setup**: 0.8.1, x86_64, isolated micro-benchmarks (`now_ns` deltas over
2000–4000-iteration loops). The M9 roadmap listed four candidate perf levers;
"numbers or it didn't happen" means measuring each before shipping. Lever 1
(SIMD LM head) already shipped as 0.8.1 (`head_fwd` V=768 9.7 → 3.59 ms, 2.7×).
This entry records the **profiling of levers 2–3, which did not pay** — recorded
so they are not re-attempted blindly.

**Lever 2 — packed `tanh` for GELU → marginal, NOT shipped.** `ganita_f64_tanh`
is `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)` — two `f64_exp`. The algebraically-exact one-exp form
`(e²ˣ−1)/(e²ˣ+1)` matches it to 2.2e-16 but benched **40 vs 47 ns/call**
(no-clamp; ~15%, and noisy): `f64_exp` on this toolchain is far cheaper than
assumed, so halving the exp count barely moves the needle. A cheap
integer-exponent clamp (to avoid `e²ˣ` overflow at |x|≥16) cost back the
saving (43 ns). GELU tanh is only ~8% of a step (fwd+bwd), so even a perfect
2× tanh is ~1–2% — inside the step-bench noise. Verdict: not worth a release.

**Lever 3 — matmul cache-blocking / register-tiling → SLOWER, NOT shipped.**
An m-blocked `linear_fwd` (4 output rows share each `W[k,n..n+3]` load, W
traffic ÷4) was **bit-identical (0 mismatches)** but **~15% slower** at the
preset MLP shape (M=64,K=64,N=256): **1.04 ms vs 0.91 ms**. At attn11's sizes
the weight matrices are L1/L2-resident, so there is no W-bandwidth problem to
solve — blocking only adds accumulator round-trips. Cache-blocking pays when
matrices exceed cache; attn11's don't.

**Lever 4 — batched prefill → measured, NO win, NOT shipped.** The
context-shift re-prime is currently `keep = T − GEN_SHIFT` single-row
`model_fwd_row` calls; the candidate was one batched window forward that fills
the K/V caches. Prototyped the *core* (the n-row batched forward vs n
single-row forwards) at the preset re-prime size n=32: **10.05 ms (batched) vs
10.12 ms (single-row) — ~1%, within noise.** The re-prime's ~70%-of-cached-cost
is irreducible *work* (the n-row causal forward — same MACs either way), not
per-call *overhead*; at attn11's scale the per-row compute dominates, so
batching saves nothing. (A bigger T might widen it, but the preset is the
target.) Not worth the inference-only complexity + bit-identity risk.

**Also confirmed already-SIMD (no lever):** the attention Q·K score and P·V
contraction (`attn_fwd`/`attn_fwd_row`) and all of `linear` were already
4-wide `f64v_fmadd` with scalar tails. The LM head was the lone scalar kernel.

**Takeaways**:
1. The perf levers are **exhausted** at attn11's scale — the LM head (0.8.1)
   was the one real win. All three remaining roadmap levers were measured and
   rejected: GELU-tanh marginal, matmul-blocking *slower*, batched prefill
   *no win*. The matmul/attention are already vectorized; transcendentals are
   exp-cheap; matrices are cache-resident (blocking hurts); the single-row gen
   path is already as efficient as a batch (work dominates overhead).
2. The residual matmul gap to SIMD peak (~10–15%) is **structural** — the
   memory-accumulator pattern forced by the "never reassign a SIMD var" rule
   (`docs/architecture/001`) and the 2-wide SSE lowering of `f64v_fmadd`.
   Closing it needs toolchain support (true AVX/FMA builtins), not an attn11
   code change. → M9 concludes; the next gains are a v2-track toolchain or
   algorithm question, not a v0.8.x lever.
3. Method note: micro-bench, not the step bench — a ~1% step change is below
   run-to-run noise (the step bench varies ±2–3%), so a lever must show clearly
   in isolation to be worth a release. Prototyping each candidate *before*
   committing (as here) is what kept three non-wins out of the release log.
