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

## X005 — MLA lands and learns (E7 → M12, v1.2.0) (2026-06-12)

**Setup**: 1.2.0, default config (d_model 32, ctx 16, 4 heads, 3 layers, biases
on), embedded corpus (vocab 25), seed 1337, 600 steps, x86_64. MHA baseline vs
MLA (`--attn-kind mla`) at d_c = 16 (= C/2). This entry records the
correctness/sanity landing, not an iso-compute tokenizer comparison.

**Result** (step-500 train loss):

| attn | d_c | params | step-500 loss |
|------|-----|--------|---------------|
| mha  | —   | 39 488 | **0.179** |
| mla  | 16  | 37 952 (−3.9%) | **0.211** |

MLA factors K/V through the low-rank latent (W_DKV: C→d_c, W_UK/W_UV: d_c→C),
trimming the per-block K/V weights from `2·C·Ckv` to `3·C·d_c` (−1 536 params at
d_c=16). It trains to a comparable loss with fewer parameters — the constrained
low-rank K/V is slightly less expressive at iso-config, as expected; MLA's payoff
is the **cached-KV** footprint (`d_c` per token vs `2·Ckv`), which the M12.2
latent-cache decode path will realize and measure.

**Takeaways**:
1. The MLA forward/backward is correct: per-op grad-check tight (≤1e-4), the full
   model composes, the checkpoint round-trips, and training converges — MLA is a
   real, trainable architecture in attn11, not just a parameterization.
2. The headline cache-compression number (KV-bytes vs GQA/MQA, cached-vs-uncached
   bit-identity) is **not** in 1.2.0 — generation uses the uncached reference
   path. The iso-param MLA-vs-GQA perplexity comparison on the vidya corpus
   (roadmap M12 gate) and the KV-bytes table are the M12.2 follow-on.
3. Method: deterministic (fixed seed), so both rows reproduce bit-for-bit —
   `./build/attn11 [--attn-kind mla --latent-dim 16] --steps 600`.

## X006 — MLA latent KV-cache decode: the compression number (M12.2, v1.2.1) (2026-06-12)

**Setup**: 1.2.1, default config (d_model 32, ctx 16, 4 heads, 3 layers), MLA at
d_c = 16 (= C/2), x86_64. The M12.2 follow-on X005 deferred: the headline
cache-bytes table + cached-vs-uncached bit-identity, now realized by the
`attn_mla_fwd_row` latent decode path. Bench harness numbers (`./build/bench`).

**Result** (persistent decode-cache footprint, NL=3, T=16, C=32, hd=8):

| cache kind          | per-token bytes/layer | total (NL·T) | vs MHA |
|---------------------|-----------------------|--------------|--------|
| MHA full K/V (nkv=4)| `2·Ckv·8` = 512       | **24 576**   | 1.0×   |
| GQA (nkv=2)         | 256                   | 12 288       | 2.0×   |
| MQA full K/V (nkv=1)| 128                   | 6 144        | 4.0×   |
| **MLA latent** (d_c=16) | `d_c·8` = 128     | **6 144**    | **4.0×** |

MLA at d_c = C/2 lands on MQA's footprint (4× under MHA) but keeps **full heads**
(nkv = nh) — MQA gets there by collapsing to a single shared K/V head; MLA gets
there by low-rank factorization while every query head keeps its own K/V on read.
The latent is also the single compression knob: d_c = 8 would halve it again to
3 072 (8× under MHA), d_c trades footprint for K/V rank directly.

**Generation speed** (default config, greedy, NGEN tokens): cached MLA decode
~4 852 tok/s vs ~1 050 tok/s for the uncached MLA reference (~4.6×). The win is
from not recomputing the whole window each token; it is **not** at MHA-cached
parity, because the reference re-up-projects the cached latents to K/V every step
(O(pos·d_c·C) per step) rather than absorbing W_UK into W_Q to attend the latents
directly. Absorption is the compute optimization (and a further memory win — no
materialized K/V working set), deferred as future work because it reorders the
accumulation and so needs its own bit-identity story.

**Takeaways**:
1. Bit-identity holds: cached MLA decode == uncached reference at every prefix and
   across context-shifts, greedy + temperature, on x86_64 AND aarch64/qemu
   (`test_kv_mla`, 25 asserts; 351→376 checks). The latent path is a drop-in.
2. The compression is real and tunable via d_c alone — 4× at the default d_c=C/2,
   matching MQA without sacrificing head count. This is the MLA thesis (the KV
   cache as the central inference object, E7/ADR 0007) made concrete at reference
   scale.
3. The iso-param MLA-vs-GQA perplexity comparison on the vidya corpus (the other
   half of the M12 gate) pairs naturally with the X003 byte-vs-BPE setup; left as
   a follow-on measurement since the bit-identity + bytes gates (the shippable
   M12.2 deliverables) are met.

## X007 — coupled RoPE: correctness, relative-position, and overhead (M12 incr. 4, v1.2.2) (2026-06-12)

**Setup**: 1.2.2, default config (d_model 32, ctx 16, 4 heads, 3 layers),
`--pos-kind rope` vs the learned-abs baseline, x86_64. This entry records the
correctness landing + the rotation overhead, not a perplexity bake-off.

**Correctness** (grad-checks, `cyrius test`, both arches):
- The rotation backward is **bit-exact** (`rope rotation dX maxrel = 0`) — RoPE
  is linear and parameter-free, so the transpose rotation IS the gradient.
- The **relative-position invariance** holds to rounding: `(R_m q)·(R_n k)` at
  offsets `(2,5)` and `(5,8)` agree to ~1e-15 (`rope rel-pos |s1-s2| ≈ 0`).
- Attention-with-RoPE: `dWq`/`dWk`/`dx` ~1e-7 (x86); the **K-bias gradient is now
  real** (a rotated bias is no longer softmax-shift-invariant) and checks at
  ~2e-7, unlike the learned-abs `|dbk| ≈ 0` no-op.
- The learned **posemb receives exactly zero gradient** under RoPE (off-path),
  pinned bit-for-bit.
- **376 → 470** checks green on x86_64 AND aarch64/qemu.

**Training** (default config, 400 steps, greedy sample): RoPE trains to a
comparable loss (~0.23 at step 250) and the cached generation reproduces real
corpus phrases — RoPE is a real, trainable, drop-in positional scheme, not just a
parameterization.

**Overhead** (default config, `./build/bench`):

| path                | learned-abs | rope     | overhead |
|---------------------|-------------|----------|----------|
| fwd+bwd step        | ~3.53 ms    | ~3.61 ms | **+2.3%** |
| gen cached (ns/tok) | ~163 µs     | ~179 µs  | **+10%**  |

The rotation adds a per-pair Maclaurin cos/sin of the base angle plus a binary
exponentiation to the position; in the cached decode that lands ~10%/token, in
the training step ~2% (attention's rotation is small against the matmuls).

**Takeaways**:
1. Coupled RoPE is correct and portable — the only new gradient (the rotation's
   transpose) is grad-checked bit-exact, and the relative-position property is
   pinned directly, independent of the gradient.
2. The trig is built without the x86-only `f64_sin`/`f64_cos` (Maclaurin on
   `θ_k ∈ (0,1]` + complex binary-exponentiation), so it runs identically on
   aarch64 and stays bit-identical cached-vs-uncached (`docs/architecture/005`).
3. RoPE on MLA is intentionally **rejected** here — the faithful form is decoupled
   RoPE on a separate `d_rope` channel (1.2.3, the last M12 rung). Coupled RoPE in
   MLA would forfeit the up-projection absorption (ADR 0007).

## X008 — decoupled RoPE: the faithful MLA + position combo (M12 incr 5, v1.2.3) (2026-06-12)

**Setup**: 1.2.3, default config (d_model 32, ctx 16, 4 heads, 3 layers), MLA at
d_c = 16 + decoupled RoPE at d_rope = 4, x86_64. Records the correctness landing +
the cache footprint; closes M12 (the `--pos-kind` switch).

**Correctness** (grad-checks, both arches):
- Per-op decoupled backward bit-tight: dWqr ~1e-6, dWkr ~1e-8 (the shared rope-key
  gradient, which accumulates across heads), content path ~1e-7, dbuk ≈ 0 (the
  up-K bias stays softmax-shift-invariant — the rope term doesn't touch it).
- Full-model wiring green (~1e-3, the composition bound); cached-vs-uncached
  **bit-identity** across context-shifts (greedy + temperature), including a
  non-even content head dim (the rotation lives on d_rope, not hd).
- v4 `pos_kind=2`/`rope_dim` round-trips; hostile rejections (decoupled on non-MLA,
  odd/out-of-range d_rope) gated before allocation. **470 → 572** checks, x86_64 +
  aarch64/qemu.

**Cache footprint** (`./build/bench`, NL=3 T=16 C=32, d_c=16, d_rope=4):

| cache kind                        | total bytes | vs MHA |
|-----------------------------------|-------------|--------|
| MHA full K/V (nkv=4)              | 24 576      | 1.0×   |
| MLA latent (learned/coupled)      | 6 144       | 4.0×   |
| **MLA + decoupled** (latent + K^R)| **7 680**   | **3.2×** |

Decoupled adds the shared rope key K^R (`NL·T·d_rope·8` = 1 536 B) on top of the
latent (6 144 B). It is the faithful DeepSeek-V2 form — position rides a separate
channel that bypasses the latent, so the latent stays absorbable (the compute
optimization, future work) and the cache stays far under full K/V while carrying
**relative** position (vs learned-absolute MLA).

**Takeaways**:
1. The decoupled core is the only new hand-derived math on the whole M12 ladder
   beyond the 1.2.2 rotation; it grad-checks bit-tight in isolation and composes.
2. The `--attn-kind` × `--pos-kind` matrix is now complete: {mha, mla} ×
   {learned, rope, rope-decoupled}, each opt-in, each with a cached bit-identity
   gate, the default run byte-identical. M12 closed.
3. A perplexity comparison (decoupled vs coupled vs learned MLA on the vidya
   corpus) pairs with the X003/X006 setups; left as a follow-on since the
   correctness + footprint gates (the shippable deliverables) are met.

## X009 — Mixture of Experts: the density sweep (M13, v1.3.0) (2026-06-12)

**Setup**: 1.3.0, default config (d_model 32, ctx 16, 4 heads, 3 layers), the
dense GELU MLP replaced by N experts + a top-2 router (`--experts N
--expert-topk 2`), Switch load-balance aux α = 0.01. Each N trained for a fixed
1200-step budget on the embedded reference corpus, then evaluated; bits/byte is
**pure cross-entropy** (the aux term is excluded from the eval metric).
Reproducible: `scripts/moe-sweep.sh`. x86_64.

**Correctness** (grad-checks, both arches): the router is the milestone. The
combine backward (`test_moe_op`, 1e-4) grad-checks the renormalized top-K softmax
+ per-expert MLP, incl. top-1 (renorm gate ≡ 1 → zero combine gradient) and K=N;
the load-balance aux (`test_moe_aux`, 1e-5) grad-checks `∂L_aux/∂logits` with the
dispatch counts held constant (straight-through). Full-model wiring green (1e-3);
cached-vs-uncached **bit-identity** across context-shifts (`test_kv_moe`); v5
round-trips + hostile rejections. **572 → 673** checks.

**Density sweep** (`scripts/moe-sweep.sh 1200`):

| N (experts) | total params | active/token | bits/byte | route-entropy |
|-------------|--------------|--------------|-----------|---------------|
| 1 (dense)   | 39 488       | 39 488       | 0.279     | —             |
| 4           | 115 040      | 64 928       | 0.291     | 0.9994        |
| 8           | 215 648      | 65 312       | 0.221     | 0.9992        |
| 16          | 416 864      | 66 080       | 0.215     | 0.9990        |
| 32          | 819 296      | 67 616       | 0.233     | 0.9958        |
| 64          | 1 624 160    | 70 688       | 0.261     | 0.9932        |

**Cost** (`./build/bench`, default config): the dense fwd+bwd step is ~3.6 ms; the
8-expert top-2 step is ~6.9 ms (top-2 = two active expert MLPs + the `C→N` router).
Cached generation ~273 µs/token. So per-token compute scales with **topk**, not N.

**Takeaways**:
1. **Total params decouple from active compute.** Total scales ~linearly with N
   (41× from N=1 to N=64) while per-token-active params stay ~65–71 K (top-2; the
   only growth is the tiny `C·N` gate). This is the whole point of sparse routing,
   and the grad-checked reference shows it learns.
2. **The aux loss prevents collapse.** Routing entropy stays 0.993–0.999 across the
   whole sweep — the experts carry near-uniform load even at N=64; without the aux
   term a top-K router collapses onto a few experts. The load-balance backward
   earns its grad-check in practice.
3. **Quality peaks at N=8–16, then over-parameterizes** (bits/byte 0.215 at N=16,
   rising to 0.261 at N=64). At attn11's reference scale (tiny corpus, fixed step
   budget) each of 64 experts sees too few tokens to train — exactly the honest
   caveat the roadmap called: the deliverable is the grad-checked reference + the
   density/utilization curve, not a quality win at this scale. N=256 exceeds the
   128 MB alloc cap at this config (rejected cleanly), bounding the sweep.
4. A vidya-corpus bake-off (MoE density vs dense vs the M12 attention variants, at
   matched compute) is the natural follow-on, pairing with X003/X006/X008.

## X010 — gated linear attention vs softmax (M14 rung a, v1.4.0) (2026-06-12)

**Setup**: 1.4.0, default config (d_model 32, ctx 16, 4 heads, 3 layers), 1200
steps on the embedded reference corpus, `--eval` (pure cross-entropy bits/byte).
Compares the three attention mixers at matched config/steps: MHA (softmax), MLA
(low-rank latent, d_c=16), and the new gated linear attention (`--attn-kind lin`,
RetNet retention, fixed per-head decay). x86_64.

**Correctness** (grad-checks, both arches): `test_lin_core` (per-op, **1e-9** — the
recurrence is pure multiply/add, no softmax/exp); `test_model_lin` (full-model
1e-3, incl. the now-real K-bias gradient — linear attention has no softmax
shift-invariance); `test_kv_lin` (cached-vs-uncached **bit-identity** across
context-shifts). **673 → 727** checks.

**Comparison**:

| mixer  | bits/byte | params | decode cache | cache scaling |
|--------|-----------|--------|--------------|---------------|
| MHA    | 0.279     | 39 488 | 24 576 B     | ∝ T           |
| MLA    | 0.273     | 37 952 | 6 144 B      | ∝ T           |
| **linear** | **0.239** | 39 488 | **6 144 B** | **constant in T** |

**Cost** (`./build/bench`, default config): linear train step ~3.8 ms (~6% over
the dense ~3.6 ms — the O(T·hd²) recurrence is comparable to softmax's O(T²·hd) at
small T); cached gen ~160 µs/token (the O(hd²) state update beats the O(T·hd)
cache scan). The state cache is 6 144 B at this config and **does not grow with
T** — at the preset (T=64) it is 16 384 B vs MHA's 262 144 B (**16×**).

**Takeaways**:
1. **Parameter-free, same projections.** Linear attention reuses the MHA Q/K/V/O
   layout with a fixed per-head decay, so it has the *same* parameter count as MHA
   (39 488) and rides the existing `attn_kind` checkpoint slot (no format bump).
2. **The cache is constant in T** — the structural win the survey points at
   (the KV cache as the central inference object). MHA/MLA caches grow with the
   window; the retention state is `nh·hd²` regardless of T.
3. **At reference scale it edges softmax on bits/byte** (0.239 vs 0.279). Read
   honestly: the tiny, repetitive corpus rewards the decay's strong recency bias,
   and there is no normalizer — this is a "the grad-checked mixer learns and is
   competitive here," not a general quality claim. The deliverable is the
   grad-checked retention reference + the constant-cache property.
4. M14's remaining rungs — (b) a minimal selective SSM (BPTT through the scan) and
   (c) per-layer mixer interleaving with a hybrid-ratio sweep — build on this core.
   A vidya-scale bake-off (linear vs softmax vs the hybrid) is the follow-on.

## X011 — selective SSM vs the other mixers (M14 rung b, v1.4.2) (2026-06-12)

**Setup**: 1.4.2, default config (d_model 32, ctx 16, 4 heads, 3 layers), 1200
steps on the embedded corpus, `--eval`. Extends X010 with the selective SSM
(`--attn-kind ssm`, state size N = 16): a Mamba-lite diagonal SSM with
input-dependent Δ/B/C (the selective scan) and a learned diagonal A. x86_64.

**Correctness** (grad-checks, both arches): the BPTT through the data-dependent
scan is the milestone. `test_ssm_core` (per-op, ~1e-7) grad-checks dWdt/dA/dWB/
dWC/dD/dWo/dbdt/dbo and the input grad through `exp(Δ·A)` + softplus + the reverse
`dh` accumulation. `test_model_ssm` (full-model 1e-3); `test_kv_ssm`
(cached-vs-uncached **bit-identity** across context-shifts — the constant C×N
state replayed on re-prime); v5 round-trip + hostile rejects. **727 → 801** checks.

**Comparison**:

| mixer  | bits/byte | params | decode cache | scaling |
|--------|-----------|--------|--------------|---------|
| MHA    | 0.279     | 39 488 | 24 576 B     | ∝ T     |
| MLA    | 0.273     | 37 952 | 6 144 B      | ∝ T     |
| linear | 0.239     | 39 488 | 6 144 B      | constant|
| **SSM**| **0.218** | 38 048 | 12 288 B     | **constant** |

**Cost** (`./build/bench`): SSM train step ~5.6 ms (the O(T·C·N) scan + the Δ/B/C
projections + exp/softplus; ~1.56× the dense ~3.6 ms); cached gen ~258 µs/token.
The C×N state cache (12 288 B at N=16) does NOT grow with T — at the preset
(T=64) it is 32 768 B vs MHA's 262 144 B (8×).

**Takeaways**:
1. **The selective scan grad-checks.** The hardest backward in the project — BPTT
   through a recurrence whose coefficients depend on the input — lands at ~1e-7,
   the deliverable for "the idea is expressible, hand-derived, in an i64 systems
   language." Δ, B, C, A, D and the input all receive gradient (the selectivity).
2. **Best bits/byte at reference scale** (0.218), edging linear (0.239) and the
   softmax mixers. Read honestly: the tiny repetitive corpus rewards the per-channel
   state + the input-dependent gating; this is a "competitive + grad-checked", not a
   scaling claim. SSMs are built for long contexts the reference scale can't show.
3. **Constant decode cache** (C×N, here 12 288 B; tunable by N) — the third
   constant-cache mixer (with linear attention), vs MHA/MLA's T-growing K/V.
4. The `--attn-kind {mha, mla, lin, ssm}` switch is now four mixers wide. Rung (c)
   — per-layer interleaving + the hybrid-ratio sweep — and a vidya-scale bake-off
   are the follow-ons.

## X012 — the per-layer hybrid: a mixer-ratio sweep (M14 rung c, v1.4.3) (2026-06-13)

**Setup**: 1.4.3, default config (d_model 32, ctx 16, 4 heads, 3 layers), 1200
steps on the embedded corpus, `--eval`. The new lever is `--attn-every K`: a
full-attention (MHA) block at every K-th layer, gated-linear (`--attn-kind lin`)
elsewhere — the survey's "a few attention layers among many cheap recurrent ones"
structural shift. NL=3 gives a four-point sweep on the attention fraction:
pure-lin (0/3) → every-3 (1/3) → every-2 (2/3) → pure-mha (3/3). x86_64.

**Correctness** (grad-checks, both arches): the per-layer dispatch is the wiring
proven here. `test_model_hybrid` (full-model 1e-3) grad-checks a [mha, lin, mha]
stack — the MIXED backward (`attn_bwd` for the attention blocks, `attn_lin_bwd`
for the linear block) composing through the residual stream and the tied head
(maxrel ~1e-5). `test_kv_hybrid` (cached-vs-uncached **bit-identity** across
context-shifts, two interleavings — each block replays its own kind's decode
path); `test_ckpt_hybrid` (the new **checkpoint v6** per-layer region round-trips,
and an image whose per-layer kind breaks the uniform-stride invariant is rejected
`-46`); `test_config_caps`/`test_alloc_accounting` hybrid pins. **801 → 857** checks.

**Comparison** (attention-fraction sweep; all configs are PARAMETER-identical —
the hybrid is free in parameters, it only redistributes the decode cache):

| attention | config            | bits/byte | params | decode cache | cache vs MHA |
|-----------|-------------------|-----------|--------|--------------|--------------|
| 0/3 (0%)  | pure lin          | 0.239     | 39 488 | 6 144 B      | 0.25×        |
| 1/3 (33%) | lin --attn-every 3| 0.244     | 39 488 | 12 288 B     | 0.50×        |
| 2/3 (67%) | lin --attn-every 2| **0.234** | 39 488 | 18 432 B     | 0.75×        |
| 3/3 (100%)| pure mha          | 0.279     | 39 488 | 24 576 B     | 1.00×        |

Decode cache is the SUM of per-layer caches: each attention layer keeps its
T-growing K/V (8 192 B at T=16), each linear layer the constant nh·hd² state
(2 048 B). So the cache scales with the attention fraction — exactly the lever.

**Cost** (`./build/bench`): the hybrid fwd+bwd step is the mix of its blocks'
steps — the 1/3-attention hybrid runs ~3.73 ms, essentially the linear step
(~3.80 ms) since two of three blocks are linear, and well under a pure-SSM step
(~5.65 ms). Cached decode advances each block's own state.

**Takeaways**:
1. **Per-layer interleaving trains and grad-checks.** A model whose blocks run
   DIFFERENT sequence mixers composes correctly forward and backward — the mixed
   stack's hand-derived gradients land at ~1e-5, and the cached decode is
   bit-identical to the uncached reference for every interleaving.
2. **The hybrid is parameter-free to switch on** (gated-linear reuses MHA's
   projections, so {mha, lin} share the block layout — no per-layer offset refactor,
   the per-block stride stays uniform). What it buys is a knob on the decode cache:
   the attention fraction sets how much of the cache is T-growing K/V vs constant
   state. At 1/3 attention the cache is half of pure-MHA's.
3. **At reference scale the ratio sweep is within noise.** The 2/3-attention hybrid
   (0.234) edges pure-lin (0.239), the 1/3 (0.244) trails it slightly, and all three
   beat pure-MHA (0.279) — but on a tiny repetitive corpus the spread is noise.
   Read honestly: this is "the hybrid is expressible and grad-checked", NOT a claim
   that any ratio wins; the survey's hybrid advantage is a long-context, scaled
   phenomenon the reference can't show. The infrastructure to run that sweep — at
   any ratio, persisted in the checkpoint — is the deliverable.
4. **Checkpoint v6** carries the per-layer pattern (the first model state that can't
   ride the scalar descriptor); uniform models still write v5, byte-identical.
5. The hybrid is restricted to LAYOUT-COMPATIBLE mixers {mha, gqa, lin}. Admitting
   MLA/SSM into a hybrid needs per-layer (or padded) parameter layouts — the rung-d
   follow-on. A vidya-scale bake-off across ratios is the standing M14 follow-on.

## X013 — the any-mixer hybrid: full attention ⊕ SSM (M14 rung d, v1.4.4) (2026-06-13)

**Setup**: 1.4.4, default config (d_model 32, ctx 16, 4 heads, 3 layers), 1200
steps on the embedded corpus, `--eval`. Rung d lifts rung c's layout restriction:
`--attn-kind ssm --attn-every K` is a hybrid of the SSM (attn11's best single
mixer, X011) with a full-attention block every K-th layer — the survey's strongest
pairing. The block K/V region is PADDED to the max `_kvw` over the present kinds
(here MHA's, since `2·C·Ckv` > the SSM's `3·C·N+C`), keeping a uniform per-block
stride. NL=3 → sweep 0/3 → 1/3 → 2/3 → 3/3 attention. x86_64.

**Correctness** (grad-checks, both arches): the deliverable is the MIXED backward
through a hybrid whose layers have DIFFERENT parameter layouts. `test_model_hybrid_ssm`
(an SSM ⊕ MHA stack, full-model 1e-3, maxrel ~1e-4) grad-checks `ssm_bwd` for the
SSM blocks composing with `attn_bwd` for the MHA block — the MHA `Wk` tiling the
padded region, the SSM `A/W_B` tiling theirs with a zeroed pad, and `Wo` (after the
padding) all correct. `test_model_hybrid_mla` (MLA ⊕ MHA). `test_kv_hybrid` adds
the mha/ssm + mha/mla **bit-identity** decode (each block replays its own kind's
cache — KV arena / C×N state / latent — within the padded layout). `test_ckpt_hybrid`
round-trips a padded mha/ssm v6 image. Alloc-accounting + config-cap pins for both.
**857 → 907** checks.

**Comparison** (attention-fraction sweep, base SSM, N=16):

| attention | config              | bits/byte | params | decode cache |
|-----------|---------------------|-----------|--------|--------------|
| 0/3 (0%)  | pure ssm            | **0.218** | 38 048 | 12 288 B     |
| 1/3 (33%) | ssm --attn-every 3  | 0.224     | 39 488 | 16 384 B     |
| 2/3 (67%) | ssm --attn-every 2  | 0.219     | 39 488 | 20 480 B     |
| 3/3 (100%)| pure mha            | 0.279     | 39 488 | 24 576 B     |

Decode cache sums the per-layer caches: each MHA layer's T-growing K/V (8 192 B at
T=16) + each SSM layer's constant C×N state (4 096 B at N=16). The padding lifts
the hybrid param count to MHA's 39 488 (vs pure SSM's 38 048 — +1 440, the SSM
layers' region padded up to MHA's by 480 each × 3 layers).

**Cost** (`./build/bench`): the mha/ssm hybrid (1/3 attention) fwd+bwd step
~5.0 ms — between pure SSM (~5.6 ms) and the dense step, since the one MHA block is
cheaper than an SSM block. Cached decode advances each block's own state.

**Takeaways**:
1. **Any-mixer hybrids train and grad-check.** A model whose layers have *different
   parameter layouts* (SSM's `3CN+C` vs MHA's `2C²` K/V region) composes correctly
   — the padded uniform stride (ADR 0012) keeps the addressing uniform while each
   layer runs its own kind's forward/backward. The mixed SSM/MHA and MLA/MHA
   backwards land at ~1e-4. This completes M14: the full `{mha, mla, lin, ssm}`
   mixer set is now interleavable.
2. **The padding cost is real but small** (+1 440 params here; the SSM layers' K/V
   region padded up to MHA's). It's the price of a uniform stride without a
   per-layer-offset refactor — exact for {mha,gqa,lin} hybrids (shared `_kvw`, no
   pad), a few % for SSM/MLA mixed with MHA at reference scale.
3. **At reference scale the ratio sweep is within noise** (again): the mha/ssm
   hybrids (0.219–0.224) sit between pure SSM (0.218, the best) and pure MHA (0.279),
   closer to SSM. Read honestly — NOT a claim that a hybrid beats pure SSM here; the
   tiny corpus and short context don't exercise where hybrids win (long-context
   recall the SSM-only model loses + the cache savings vs pure attention). The
   deliverable is the *mechanism*: any ratio of any mixers, persisted in v6,
   grad-checked — ready for a vidya-scale bake-off.
4. The decode cache is now a continuous knob from pure-SSM's 12 288 B (constant in
   T) to pure-MHA's 24 576 B (∝ T): each attention layer added trades constant
   state for T-growing K/V. The hybrid is where that trade is dialed.

## X014 — mixer perf consolidation: latency, cache, and the padding cost (1.4.6) (2026-06-13)

**Setup**: 1.4.6, the dedicated benchmarking release. One canonical `./build/bench`
run at the default config (V=25, d_model 32, ctx 16, 4 heads, 3 layers), x86_64,
stable to a few percent. Where X010–X013 measured bits/byte (quality), this
consolidates the LATENCY + CACHE + PARAM picture across the whole mixer family and
pins the rung-d padded-layout cost. No code change beyond two param-count prints
added to the hybrid bench entries.

**Latency + cache + params** (the money table; see docs/benchmarks.md):

| mixer            | step (ns) | step ×MHA | gen (ns/tok) | cache (B) | scaling | params |
|------------------|-----------|-----------|--------------|-----------|---------|--------|
| MHA (default)    | 3 572 260 | 1.00×     | 163 030      | 24 576    | ∝ T     | 39 488 |
| MLA (d_c=16)     | ~3.6e6    | ~1.0×     | ~206 000     | 6 144     | ∝ T     | 37 952 |
| linear           | 3 781 400 | 1.06×     | 161 310      | 6 144     | const   | 39 488 |
| SSM (N=16)       | 5 626 938 | 1.58×     | 260 161      | 12 288    | const   | 38 048 |
| MoE (8 / top-2)  | 6 969 545 | 1.95×     | 275 172      | 24 576    | ∝ T     | 215 648|
| mha/lin (1/3)    | 3 740 414 | 1.05×     | 164 000      | 12 288    | mixed   | 39 488 |
| mha/ssm (1/3)    | 4 900 562 | 1.37×     | 228 912      | 16 384    | mixed   | 39 488 |

**Takeaways**:
1. **The padded hybrid adds NO compute** — only memory. The mha/ssm 1/3 step
   (4 900 562 ns) matches the per-layer mix (1·MHA + 2·SSM)/3 = 4 942 045 ns to
   within noise: the zeroed pad is never read (each block's dispatch touches only
   its own kind's weights), so the rung-d padding costs params + Adam moments, not
   FLOPs. The mha/ssm hybrid is 39 488 params (vs pure SSM's 38 048, +1 440 / ~4%);
   the mha/lin hybrid is free (shared layout, no pad).
2. **linear ≈ MHA in compute, far under in cache.** The retention recurrence is the
   same order as the attention it replaces (+6% step, decode at parity) but its
   decode state is constant in T (6 144 B vs MHA's T-growing 24 576 B). SSM is
   ~1.58× the step (the O(T·C·N) selective scan + Δ/B/C projections) for a
   constant 12 288 B state.
3. **The decode cache is a continuous knob** in the hybrid's attention fraction
   (mha/ssm: 12 288 → 16 384 → 20 480 → 24 576 B from 0/3 to 3/3 attention) — each
   attention layer trades constant recurrent state for T-growing K/V, at no param
   cost beyond the pad. This is the survey's hybrid lever made measurable.
4. **Zero regression to the default path.** The no-flag MHA training step is flat
   from 0.4.0 through 1.4.6 (~3.6 ms, ~4 450 tok/s b=16; bench-history.csv) — the
   entire M12–M14 arc (MLA, RoPE, MoE, linear, SSM, hybrid) added five opt-in axes
   without touching the default run. MoE is the one big-param axis (5.5× params at
   N=8, ~2× step for top-2) — capacity at near-constant active compute.

## X015 — AR vs char-diffusion at matched compute (M15, v1.5.0) (2026-06-13)

**Setup**: 1.5.0, the first *training-objective* comparison. Matched trunk + matched
compute: the default config (V=25, d_model 32, ctx 16, 4 heads, 3 layers, dense,
learned-abs), seed 1337, the embedded 190-byte corpus, **2000 steps × batch 16** for
BOTH objectives — the only difference is `--objective`. The "super data learner"
regime: a tiny corpus seen for many epochs. x86_64. The two objectives are NOT
directly comparable head-to-head (see takeaway 3), so three clearly-scoped numbers:

| objective         | metric                                  | bits/byte |
|-------------------|-----------------------------------------|-----------|
| **AR**            | exact next-token NLL (`eval_corpus`)    | **0.254** |
| diffusion         | denoising, mask 10% (1 of 16 masked)    | 3.725     |
| diffusion         | denoising, mask 30%                      | 3.670     |
| diffusion         | denoising, mask 50%                      | 3.713     |
| diffusion         | denoising, mask 70%                      | 3.876     |
| diffusion         | denoising, mask 90%                      | 3.944     |
| **diffusion**     | **ELBO bound (mean over t)**            | **3.786** |

(uniform baseline = ln(25)/ln 2 ≈ 4.64 bits/byte.)

**Takeaways**:
1. **At this reference scale, AR wins decisively** — it memorizes the tiny corpus
   (0.254 bits/byte) while diffusion only edges below uniform (3.79 ELBO bound). The
   "super data learner" advantage does NOT appear here: it is a *scale* phenomenon
   (the survey's claim is for large models / large repeated corpora), and a 39 K-param
   model over 190 bytes is far below where dLLMs overtake AR. This is the honest
   negative result the milestone gate asked for — "a logged comparison, not a
   required win."
2. **Diffusion learns, but slowly.** Even at 10% masking (predict 1 char from 15
   bidirectional neighbours) it reaches only 3.73 bits/byte: training samples a
   *random* mask ratio t~U(0,1) per example, so the easy low-mask regime is seen
   rarely, and the high-variance objective converges far slower than AR's
   full-left-context next-token loss on a memorizable corpus. The grad-checks are
   tight (`test_model_diffusion` ~1e-5..1e-4), so the gradient is correct — this is
   training dynamics, not a bug.
3. **The comparison is intrinsically asymmetric — stated, not hidden.** AR's number
   is the *exact* NLL; diffusion's is an *ELBO upper bound* on its NLL (the MDLM 1/t
   weight cancels the t-scaling of the masked count, so the unweighted per-masked-
   token CE averaged over t IS the bound). A bound ≥ AR proves nothing on its own —
   BUT here the gap (3.5 bits) dwarfs any plausible bound-looseness, so AR genuinely
   wins at this scale. The shared, objective-neutral axis is the denoising grid.
4. **Greedy decode collapses to high-frequency tokens** at this scale (the demo
   samples skew toward spaces) — a documented small-dLLM limitation, not a
   correctness issue. A stochastic/temperature decode and larger scale are the
   fast-follows. Regeneration: `./build/attn11 --steps 2000 --eval` (AR) and
   `./build/attn11 --objective diffusion --steps 2000 --eval` (diffusion grid).

## X016 — teaching English from C4: a large external corpus (v1.5.1) (2026-06-13)

**Setup**: 1.5.1, the first run on a real **large external dataset** — a 4 MB slice
of **C4** (`c4/en`, the 305 GB Colossal Clean Crawled Corpus), streamed with
`scripts/c4_sample.py` (stdlib `gzip`+`json` over a public C4 shard — no
tensorflow/TFDS/pip; ~1 MB actually downloaded). Goal: does a tiny attn11 learn
English off raw web text and "speak" something? Default config (C=32, ctx 16, 3
layers) + **BPE 256** (subword tokens, so the model predicts word-pieces), 600 steps,
seed 1337, x86_64. The `data/` corpus is gitignored.

| metric | value |
|--------|-------|
| C4 sample | 4,002,896 bytes (1 stream, ~0.3 s) |
| vocab (byte base + 256 BPE merges) | 438 |
| params | 52,704 |
| corpus tokens (BPE) | 1,969,733 |
| train loss | 5.29 (250) → 4.90 (600) |
| eval CE/token · bits/byte | 4.836 · **3.433** |

Sample (temp 0.8): *"a transformer arduns famt m, resianit conin the loltinaitjae
bfre heelsaral be of poluring a you clonwetkes froms whisuly 20gam nurenten ... gent
ho the geyf eet tecudiis it pemired in the blench have leicing alle cidforesionw
pateat"*.

**Takeaways**:
1. **It speaks English-ish.** Real words and structure thread through the temp-0.8
   sample (`the`, `be of`, `a you`, `for`, `have`, `it`, `in the`) between wobbling
   subwords — a 53 K-param model producing English-shaped text off web prose it was
   never given the rules of. Greedy decoding collapses to high-frequency tokens
   ("and"/"see"/" "); temperature sampling is the readable view.
2. **Pipeline, not Beam.** The C4 source is 305 GB but attn11 only needs a few MB of
   bytes; streaming one shard and stopping early ([`scripts/c4_sample.py`](../../scripts/c4_sample.py))
   gets the *identical* corpus TFDS catalogs with zero heavy deps and a ~1 MB
   download. The "large dataset" is the *source*; the model-facing slice is small (a
   tiny model saturates on diversity long before 4 MB).
3. **Fluency is a capacity story, not a data one — and the preset proves it.** The
   default reaches 3.43 bits/byte (vs ~1–1.5 for a strong English model, ~6+ uniform
   over this vocab); the **`--preset`** (ctx 64 / C=64 / 4 layers, 232 K params) on
   the *same* 4 MB slice reaches **2.695 bits/byte** at 1500 steps — a **0.74
   bits/byte** gain purely from more context + capacity, with visibly more word-shaped
   output ("*…made … looking court and … will … make nights of … to posess … of your
   …*"). Same data, better model ⇒ better English: the lever is model scale (M16+)
   and budget, not corpus volume. Regeneration: see
   [`docs/examples/c4-english.md`](../examples/c4-english.md).

## X017 — curation: quality helps, diversity is a scale lever (v1.5.2) (2026-06-13)

**Setup**: 1.5.2, the first data-curation A/B (`scripts/c4_sample.py --curate`).
Iso-compute: default config (C=32, ctx 16, 3 layers) + BPE 256, **600 steps**, seed
1337, x86_64 — the *only* difference is the 4 MB corpus. Three corpora from C4-en:
**raw** (one shard, consecutive — the 1.5.1 baseline), **curated-s0** (the *same*
shard, quality-filtered — isolates the filter from diversity), and **curated-8sh**
(8 shards spread across the crawl + quality-filtered — the recommended diverse
output). C4 is already deduplicated at creation, so `dup=0`; the levers here are the
quality filter (~5–6% of docs dropped as tables/listings/url-hash spam) and shard
diversity.

| corpus | what differs vs raw | eval bits/byte | Δ |
|--------|---------------------|----------------|---|
| raw (1 shard) | — | 3.433 | — |
| **curated-s0** (1 shard, quality-filtered) | **quality only** | **3.232** | **−0.20 (−5.9%)** |
| curated-8sh (8 shards, quality + diversity) | + multi-shard diversity | 3.527 | +0.09 (+2.7%) |

**Takeaways**:
1. **The quality filter cleanly helps** — same shard, filter on vs off, **−0.20
   bits/byte (−5.9%)** at iso-compute. Dropping tables / listings / url-hash spam /
   low-prose docs leaves a more *learnable* corpus, so the fixed-capacity model
   captures more of it. This is the 1.5.2 gate, met.
2. **Multi-shard diversity *hurts* the tiny model** (+0.09 bits/byte): 8 spread
   shards raise the corpus's intrinsic entropy (more varied registers/topics), and a
   53 K-param model can't exploit the diversity it can't fit — bits/byte on its own
   corpus penalizes variety. **Diversity/volume is a *scale* lever**, not a tiny-model
   one — which is exactly why the roadmap sequences streaming + larger corpora with
   the model-scale work (M16+), not before it.
3. **So curate for QUALITY now, DIVERSITY later.** For the current reference models,
   the recommended sampler is the quality filter on a focused corpus; the
   `--shards N` diversity tooling is built and waiting for the capacity to use it.
   (Caveat: bits/byte-on-own-corpus rewards a less-varied corpus, so it understates
   diversity's value for generalization — a held-out cross-corpus eval is the honest
   way to score diversity, and lands with the scaled runs.) Regeneration:
   [`docs/examples/c4-english.md`](../examples/c4-english.md).

## X018 — token-packing: the corpus-ceiling unlock (v1.5.3) (2026-06-13)

**Setup**: 1.5.3, the data-ingestion arc's storage step. Not a training experiment —
a memory/representation measurement + a byte-identity proof. The corpus token stream
`g_data` moved from one **i64 per token (8 B)** to a **packed** byte store: u8 for
byte-level (vocab ≤ 256), u16 for BPE (vocab ≤ 768). The model/training math is
untouched (same ids feed the same forward), so the gate is "byte-identical loss
curve + a larger corpus loads/trains + grad-checks/fuzz green", not a number.

**Storage** (bytes per token, and the resulting single-allocation corpus ceiling
against the 256 MB `ALLOC_MAX`):

| tokenizer  | before (i64) | after (packed) | g_data ceiling | vs before |
|------------|--------------|----------------|----------------|-----------|
| byte-level | 8 B/token    | **1 B (u8)**   | ~256 MB        | **8×**    |
| BPE        | 8 B/token    | **2 B (u16)**  | ~128 MB        | **4×**    |

`MAX_CORPUS_BYTES` raised **4 MB → 64 MB**: the u16 `g_data` is then 128 MB (half
the per-alloc cap), leaving room for `g_text` (64 MB) + the model budget.

**Byte-identity** (the gate): default (byte-level/u8), `--bpe 64` (u16), and
`--preset` (byte-level, bigger config) training runs are **byte-for-byte identical**
to the 1.5.2 binary's output (full loss curve + eval + samples). Packing is invisible
to the math.

**Verified**: 977 grad-checks (was 966; +11 from `test_token_packing`) green on
x86_64 **and** aarch64/qemu; lint clean; fuzz green (100 random corpora + BPE
round-trip); `make smoke` green. A 6 MB corpus — over the old 4 MB cap — loads
(6 291 456 byte tokens / 1 138 445 BPE tokens) and trains (byte-level loss 0.239 at
step 250); a 65 MB corpus rejects cleanly (code −2, no crash).

**Takeaways**:
1. **The 8×/4× headroom is real and free** — token ids are ≤ 767, so they fit u8/u16
   trivially; the i64 store was 6–7 B of waste per token. The byte-level default
   (the common case) gets the full 8×.
2. **Diversity/volume is still a *scale* lever** (X017) — this lifts the *ceiling*
   that would otherwise block a larger curated corpus, but a tiny model saturates on
   data long before 64 MB. The payoff lands with 1.5.4 (curation at scale) and M16+.
3. **Streaming is still the RAM-independent path** (1.6.x) — packing buys ~4–8× in
   the *same* RAM; decoupling corpus size from RAM entirely waits for a model big
   enough to need it. Regeneration: `./build/attn11 [--bpe K] --steps N --eval`
   (deterministic; compare to a 1.5.2 binary for the byte-identity check).

## X019 — curation at scale: more clean data pays off with capacity (v1.5.4) (2026-06-13)

**Setup**: 1.5.4, the data-ingestion arc's scaled run — the first to use 1.5.3's
raised corpus ceiling. Two C4-en corpora, both `scripts/c4_sample.py --curate` (the
prose-quality filter), BPE 256, seed 1337, x86_64:
- **4 MB / 1-shard** (`--shards 1 --max-bytes 4000000`) — 4,000,061 B, the X017
  curated-s0 corpus, byte-for-byte reproduced (1812 docs, dup 0, lowq 110).
- **24 MB / 12-shard** (`--shards 12 --max-bytes 24000000`) — 24,004,524 B
  (11,143 docs across 12 shards spread over the crawl; dup 4, lowq 712). **6× the old
  4 MB cap** — impossible before 1.5.3's packed store; loads + trains under the 64 MB
  cap. The model-facing slice is real multi-source web English, not one block.

Two model sizes at **matched compute** (fixed steps per model class): **default**
(C=32, ctx 16, 3 layers, ~53 K params, 600 steps) and **preset** (C=64, ctx 64, 4
layers, ~232 K params, 1500 steps). Metric: `--eval` bits/byte over each model's
**own** corpus (the caveat below).

**Result** (eval bits/byte, lower = better):

| model | 4 MB curated (1-shard) | 24 MB curated (12-shard) | **capacity Δ (same corpus)** |
|-------|------------------------|--------------------------|------------------------------|
| default (≈53 K) | **3.232** | **3.405** | — |
| preset (≈232 K) | **2.666** | **2.741** | — |
| **data Δ (4 → 24 MB, same model)** | — | — | |

- capacity (default → preset): **−17.5%** on 4 MB, **−19.5%** on 24 MB.
- data/diversity (4 MB → 24 MB): **+5.4%** for default, **+2.8%** for preset.
- the 4 MB default cell **reproduces X017's curated-s0 (3.232) bit-for-bit** —
  confirming the curation pipeline is deterministic AND that 1.5.3's packed store
  trains identically to 1.5.2 (the transparency gate, re-verified end-to-end).
- preset on 4 MB *curated* (2.666) edges X016's 4 MB *raw* (2.695) by −1.1% — the
  quality filter helps the bigger model too (smaller effect than at default scale).

**Takeaways**:
1. **Capacity is the dominant, reliable lever.** default → preset cuts bits/byte
   ~17–20% on BOTH corpora, and the temp-0.8 samples jump from broken fragments to
   sentence-shaped English with real function words. No surprise, but it anchors the
   scale: a 4× bigger model beats any data move at this regime.
2. **The diversity/volume penalty HALVES with capacity** — the headline. On its own
   corpus a bigger, more diverse corpus reads as *higher* bits/byte (more entropy to
   fit): the tiny model pays **+5.4%** and its samples get visibly MORE garbled
   (it can't fit the variety — exactly X017's finding). The preset pays only **+2.8%**
   and stays fluent with **richer vocabulary** ("…sciences of… margizes… This a
   safter…" vs the 4 MB preset's flatter "…service and service…"), and its capacity
   benefit is actually **larger** on the diverse corpus (−19.5% vs −17.5%). So the
   bigger model extracts MORE from the richer data — the first attn11 evidence that
   **diversity/volume starts paying off as capacity grows**, validating the roadmap's
   sequencing of streaming + larger corpora with model scale (M16+).
3. **The metric understates data's value — stated, not hidden.** bits/byte-on-own-
   corpus penalizes the higher-entropy 24 MB corpus in ABSOLUTE terms, so it can't
   show a clean "more data → better generalization" win; the fair test is a **held-out
   cross-corpus eval** (train on A, eval on a disjoint B). attn11's `--eval` only
   scores the training corpus, and `--load`+`--corpus` enforces a tokenizer-vocab
   match, so held-out eval needs a small additive **`--eval-corpus FILE`** flag
   (re-encode a second file through the loaded tokenizer, no vocab-order check) — a
   clean 1.5.x follow-on, deferred (1.5.4 is binary-unchanged). With it, the +2.8%
   own-corpus penalty would very plausibly flip to a generalization win for the preset.
4. **Honest bottom line**: at attn11's reference + preset scales, **capacity binds**;
   curation-at-scale + 1.5.3's headroom have the data side ready, and the crossover
   where more clean data clearly wins lives at M16+ (bigger models) plus the held-out
   eval. Regeneration:
   `python3 scripts/c4_sample.py --curate --shards 12 --out data/c4-curated-24mb.txt --max-bytes 24000000`
   then `./build/attn11 --corpus data/c4-curated-24mb.txt [--preset] --bpe 256 --steps {600|1500} --eval`.

## X020 — held-out (cross-corpus) eval: the own-corpus metric is already honest (v1.5.6) (2026-06-13)

**Setup**: 1.5.6, the X019 deferred follow-on — the first run to use the new
**`--eval-corpus PATH`** flag (re-encode a disjoint corpus through the loaded tokenizer,
score bits/byte on it). Goal: quantify the **own → held-out generalization gap** X019
flagged as unmeasurable, i.e. *is* own-corpus bits/byte inflated by memorization?
Two **genuinely disjoint, same-distribution** slices of the X019 24 MB curated corpus
(verified no shared documents — only a blank line in common):
- **trainA** — first 4 000 000 B (`head -c`).
- **heldB** — bytes [12 MB, 16 MB) (`tail -c +12000001 | head -c 4000000`); an 8 MB gap
  from trainA guarantees disjoint docs, same C4-curated distribution.

Two model sizes at X019's matched compute, BPE 256, seed 1337, x86_64: **default**
(C=32, ctx 16, 3 layers, 52 768 params, 600 steps) and **preset** (C=64, ctx 64, 4
layers, 232 320 params, 1500 steps). Metric: bits/byte on trainA (`--eval`, own) vs
heldB (`--eval-corpus`, held-out).

**Result** (bits/byte, lower = better):

| model | own (trainA) | held-out (heldB) | **gen. gap** |
|-------|--------------|------------------|--------------|
| default (≈53 K) | **3.30599** | **3.33000** | +0.024 (**+0.73%**) |
| preset (≈232 K) | **2.73826** | **2.77305** | +0.035 (**+1.27%**) |

- capacity (default → preset): **−17.2%** own, **−16.7%** held-out — the X019 capacity
  lever persists unchanged on held-out text (it is not a memorization artifact).
- preset own-corpus **2.738** ≈ X019's 24 MB-curated preset **2.741** (trainA is a 4 MB
  slice of that same 24 MB pool) — an independent cross-run consistency check.

**Takeaways**:
1. **The own → held-out gap is tiny (< 1.3%) at both scales** — the headline, and a
   mild *surprise*. The hypothesis going in was that a bigger model would *memorize*
   its 4 MB slice and so show a *larger* own→held gap; instead BOTH gaps are within
   ~1% and the preset's is only marginally bigger (1.27% vs 0.73%). The reason is
   compute, not capacity: at 600/1500 steps × batch 16 over ~2 M tokens the models see
   well under one epoch (X016's "~8% of an epoch in 600 steps"), so there is almost
   nothing to memorize — own and held-out are both essentially *first-pass* estimates.
2. **This RETROACTIVELY VALIDATES the X016–X019 own-corpus numbers.** The worry X019
   logged — "bits/byte-on-own-corpus understates data's value / could be inflated by
   fitting the training set" — does **not** bite at this scale: own-corpus bits/byte is
   a trustworthy, near-unbiased generalization proxy here (it tracks held-out to within
   ~1%). The metric the whole 1.5.x data arc reported was honest all along; X020 is the
   evidence, not an assumption.
3. **The X019 "more clean data → held-out win" hypothesis is NOT yet testable — and X020
   says why.** That claim needs a *different* experiment: train one model on 4 MB and
   another on 24 MB, then eval BOTH on a third corpus disjoint from both. X020 instead
   measures the own→held gap for a *single* training set, so it can't show the data-
   volume win — but it shows the *precondition* for one (a memorization gap to convert)
   is absent at this scale. The data-volume generalization win, like the capacity
   crossover, lives at **M16+** where a bigger model can actually overfit a small corpus.
   That comparison is now unblocked by `--eval-corpus`; logged as the **X021** follow-on.
4. **Honest bottom line**: the new flag works (byte + BPE, AR + diffusion, RNG-neutral,
   bit-reproducible across the in-session and `--gen-only --load` paths), and its first
   result is a *reassuring null*: at attn11's scale own-corpus bits/byte already measures
   generalization. Regeneration:
   `head -c 4000000 data/c4-curated-24mb.txt > data/c4-24-trainA.txt`,
   `tail -c +12000001 data/c4-curated-24mb.txt | head -c 4000000 > data/c4-24-heldB.txt`,
   then `./build/attn11 --corpus data/c4-24-trainA.txt [--preset] --bpe 256 --steps {600|1500} --eval --eval-corpus data/c4-24-heldB.txt`.

## X021 — data-volume held-out win: more clean data generalizes better, with capacity (v1.6.5) (2026-06-14)

**Setup**: 1.6.5, the experiment X019 wanted and X020 unblocked but couldn't reach — the
own→held gap was <1.3% at sub-epoch compute, so there was no memorization to convert. The
data-volume win only shows once the model **overfits** the small corpus, so X021 forces
that regime: a SMALL train slice trained for many epochs vs a LARGER disjoint slice at
**matched compute**, both scored on a THIRD disjoint slice. Rides the shipped
`--eval-corpus` + `c4_sample.py` — **no new binary surface** (the runner is
[`scripts/x021-heldout.sh`](../../scripts/x021-heldout.sh); binary byte-identical to 1.6.4,
CFG_VERSION bump only). Three disjoint, same-distribution slices of one curated 12 MB C4-en
pool (byte-offset gaps guarantee disjoint documents, as in X020):
- **trainS** — `[0, 256 KB)` (the overfit target).
- **trainL** — `[1.25 MB, 5.25 MB)` (4 MB; 16× more, diverse data; a 1 MB gap from trainS).
- **heldB** — the last 512 KB `[11.49 MB, 12 MB)` (disjoint from both).

Both model sizes — **default** (≈51 K) and **preset** (≈229–233 K) — BPE 256, seed 1337,
**4000 steps each** (matched compute; at preset that is ~30 epochs over trainS vs ~4 over
trainL). The deviation from the roadmap's literal "4 MB vs 24 MB" is deliberate and
honest: attn11's largest model (preset, frozen surface) cannot overfit 4 MB of *diverse*
C4 even at many epochs (X020), so the hypothesis is untestable there; 256 KB *is*
overfittable at preset capacity, which is where the data-volume effect is decidable. Same
question, the scale at which attn11 can answer it. Metric: bits/byte (byte-normalized, so
comparable across the two tokenizers — X003) on own (`--eval`) vs heldB (`--eval-corpus`).

**Result** (bits/byte, lower = better):

| cell | params | own (train) | held-out (heldB) | own→held gap |
|------|--------|-------------|------------------|--------------|
| default · trainS (256 KB) | 51 200 | 2.70859 | 2.86080 | **+5.6%** |
| default · trainL (4 MB)   | 52 928 | 2.87577 | 2.85913 | −0.6% |
| **preset · trainS (256 KB)** | 229 184 | **1.97732** | **2.77612** | **+40.4%** |
| **preset · trainL (4 MB)**   | 232 640 | 2.36582 | **2.38319** | +0.7% |

**Held-out (the headline), trainL vs trainS:** preset **2.38319 vs 2.77612 = −14.2%**
(more data generalizes much better); default **2.85913 vs 2.86080 = −0.06%** (a tie).

**Takeaways**:
1. **More clean data → a decisive held-out win, AT CAPACITY.** The preset trained on 4 MB
   beats the one trained on 256 KB by **−14.2% bits/byte on held-out text** — the
   data-volume generalization win X019 hypothesized and X020 said needed the overfit
   regime. The default (tiny) model is a **tie** (−0.06%): it lacks the capacity to exploit
   the extra data. So the data-volume win is a **capacity lever** — the exact shape of the
   X017 ("diversity is a scale lever") and X019 ("the penalty halves with capacity")
   findings, now confirmed on *held-out* text.
2. **The overfit regime — X020's missing precondition — is reached, and own-corpus
   bits/byte INVERTS the truth there.** preset·trainS has a **+40.4%** own→held gap (own
   1.977 ≪ held 2.776): it memorized the 256 KB slice. On its OWN corpus trainS (1.977)
   looks far *better* than trainL (2.366) — but that is memorization, not skill: on
   held-out, trainL (2.383) crushes trainS (2.776). This is precisely the failure mode
   X019 flagged ("own-corpus understates data's value") and X020 couldn't exhibit (no
   memorization at sub-epoch). The gap scales with capacity (default·trainS only +5.6%:
   too small to memorize much), so own-corpus bits/byte is a trustworthy proxy ONLY below
   the overfit threshold (validating X020's read of the 1.5.x numbers) and **misleading
   above it** — exactly when a held-out metric earns its keep.
3. **Closes the data story.** X016 (data ingestion) → X017 (quality > diversity at tiny
   scale) → X018 (the packing ceiling) → X019 (capacity uses more data) → X020 (own is
   honest sub-epoch) → **X021 (more data wins on held-out, at capacity, once overfitting
   makes own dishonest).** The `--eval-corpus` flag (built in 1.5.6 for exactly this) is
   the instrument. Reproduce: `scripts/x021-heldout.sh` (deterministic curation + seed;
   RNG-neutral evals; cross-arch reproducible).

## X022 — ternary (BitNet) vs f64 at reference scale (M16, v1.6.0) (2026-06-14)

**Setup**: 1.6.0, the first **precision-axis** comparison — same trunk, same compute,
the only difference is `--ternary`. Default config (V=25, d_model 32, ctx 16, 4 heads,
3 layers, dense, MHA, learned-abs), seed 1337, the embedded 190-byte corpus, **2000
steps × batch 16**, `--eval` (exact next-token NLL → bits/byte). Ternary quantizes the
MHA Q/K/V/O + dense-MLP weights to `W_eff = γ·clamp(round(W/γ),−1,+1)` (γ = absmean) in
the forward, with an STE backward; the master weights stay f64. The "super data learner"
regime: a tiny corpus seen for many epochs (matching X015's setup, so the f64 row is a
direct cross-check).

| weights | params | bits/byte |
|---------|--------|-----------|
| **f64** (baseline) | 39 488 | **0.254** |
| **ternary** {−1,0,+1} | 39 488 (master f64) | **0.228** |

(The f64 row reproduces X015's 0.254 **bit-for-bit** — the `--ternary` flag is additive,
the default path byte-identical.)

**Takeaways**:
1. **Ternary trains, grad-checks, and is competitive at reference scale** — it even edges
   f64 here (0.228 vs 0.254). Read honestly: on a tiny, memorizable corpus the ~1.58-bit
   constraint acts as a mild **regularizer**, and at 39 K params both objectives converge
   low; this is a "the grad-checked STE learns and is competitive," **NOT** a general
   claim that ternary beats f64. BitNet's real advantage is the **memory + i64-add
   compute** at large scale, which the reference scale can't show (and which increment 1
   does not yet realize — see takeaway 3).
2. **The STE is correct where it is defined.** The op-level grad-check (`test_ternary_quant`)
   confirms the `dx` path through the fixed `W_eff` matches finite differences (1e-5) and
   that the STE `dW` is the bit-exact pass-through (`dW = xᵀ·dy`, independent of the
   quantizer — a naive FD of `dW` through the piecewise-constant quantizer is meaningless,
   which is *why* an STE is used); the full-model check FDs the smooth params (embeddings,
   LN, biases) through the quantized stack. 986 → 1010 checks, green x86_64 + aarch64/qemu.
3. **This is correctness, not yet speed.** Increment 1 reuses the f64 `linear_fwd` (it
   re-quantizes `W` into a scratch each matmul — redundant f64 work), so it shows the
   *learning* is sound but realizes **no** throughput win. The **i64-add ternary matmul**
   (`x·W_eff = γ·(x·t)`, the multiply collapsing to add/subtract/skip) **benched against
   the SIMD-f64 path** is the M16 increment-2 follow-on (the remaining gate). Regeneration:
   `./build/attn11 --steps 2000 --eval` (f64) and `./build/attn11 --ternary --steps 2000 --eval`.

## X023 — the i64-add ternary matmul vs the SIMD-f64 path (M16 increment 2, v1.6.1) (2026-06-14)

**Setup**: 1.6.1, the M16 increment-2 gate (the X022 takeaway-3 follow-on). The BitNet
realization: `x·W_eff = γ·(x·t)`, `t ∈ {−1, 0, +1}`, so the contracted dim collapses to
**add / subtract / skip** with **one** γ-scale per output (M·N multiplies vs the dense
M·K·N). Two reference kernels (`ternary_matmul_fwd`, `ternary_matmul_dx` in `ops.cyr`)
implement the collapse and are grad-checked (`test_ternary_matmul`: the forward and `dx`
pinned against the SIMD-f64 `W_eff` path at maxrel **0**, `dx` FD'd at 1e-5; 1010 → 1014
checks). They are benched **head-to-head** against the 1.6.0 SIMD-f64 path
(`ternary_quant` → rosnet `linear_fwd`, 4-wide `f64v_fmadd`) at the MLP shape
**T×C×F = 16×32×128**, x86_64, 2000 iters/cell.

| surface | SIMD-f64 (W_eff) | i64-add collapse | i64-add ÷ SIMD-f64 |
|---------|------------------|------------------|--------------------|
| matmul only | **60.5 µs** | 181.5 µs | **2.95–3.00×** (slower) |
| end-to-end (quant + matmul) | **94.2 µs** | 225.0 µs | **2.38–2.39×** (slower) |

**Takeaways**:
1. **The collapse is exact and grad-checked, and at this scale/hardware it LOSES — an
   honest negative.** `γ·(x·t)` reproduces the `W_eff` matmul bit-for-bit-to-rounding
   (maxrel 0 on the forward + `dx` pins), so it is a *correct* ternary matmul. But on
   x86_64 it is **~3× slower** (matmul) / **~2.4× slower** (end-to-end) than the SIMD-f64
   path — `f64v_fmadd` does **4** fused multiply-adds per instruction, while the collapse
   is **scalar** add/sub/skip (one element per op, plus a branch). The SIMD multiply is
   already cheaper per-element than a scalar add; the ~31% zero-skip on Gaussian `W`
   doesn't recover the gap (branch cost).
2. **Why the BitNet win is absent here, and where it lives.** The integer-add advantage
   needs one of: (a) **activation quantization** (int8 absmax acts → a literal *integer*
   matmul, not f64 add — the heavier follow-on this reference deliberately scopes out,
   ADR 0014), and/or (b) **hardware where mul ≫ add or without wide FMA** (the collapse
   trades one expensive mul for one cheap add). Modern x86 with 4-wide f64 FMA + f64
   activations is exactly the regime where the collapse can't win. The **memory** win
   (1.58 bits/weight vs 64) is real and orthogonal — it is not what this kernel measures.
3. **Decision: the default ternary forward keeps the SIMD-f64 `W_eff` path** (the faster
   one). The collapse ships as the **grad-checked + benched reference kernel** + this
   honest finding — so the gate is met ("build it + bench it vs SIMD-f64") and **ternary
   runs stay byte-identical to 1.6.0** (the kernel is additive, wired into no run). This
   mirrors the project's standing pattern of honest negatives (X004 rejected M9 levers,
   X015 diffusion-below-uniform at tiny scale). Regeneration: `cyrius build
   tests/attn11.bcyr build/bench && ./build/bench` (the `ternary i64-add` / `SIMD-f64`
   rows, `(x100)` ratios).

## X024 — REINFORCE vs SFT: the policy moves toward the reward (M17, v1.7.0) (2026-06-14)

**Setup**: 1.7.0, the first **reinforcement-learning** run — `--objective rl` (REINFORCE,
E9, ADR 0015). The M17 gate: *does the policy move toward a reward?* Train the SAME default
model (C=32, ctx 16, 3 layers, 39 488 params, byte-level, seed 1337) two ways on one
corpus and compare per reward-target char: the target's frequency in the policy's samples
(measured identically for both via the in-binary rollout eval over 64 rollouts) and the
corpus bits/byte (LM quality). **SFT** = the usual AR next-token CE. **RL** = on-policy
REINFORCE: each step samples `batch` rollouts at temperature 1, scores each by the count of
the target char, and weights its log-prob gradient by the advantage `(R − b)` (b = an EMA
of past mean rewards). The policy-gradient *is* the softmax-CE backward over the sampled
rollout scaled by `(R − b)` — no new backward math (grad-checked in `test_rl_op`). 400 steps
each. Runner: [`scripts/m17-rl.sh`](../../scripts/m17-rl.sh) (no new binary surface beyond
the M17 flags). SFT baseline corpus bits/byte **0.241**.

**Result** (target frequency in policy samples; corpus bits/byte):

| reward target | SFT freq | RL freq | SFT b/b | RL b/b |
|---------------|----------|---------|---------|--------|
| `'e'` (common)  | 9.76%  | **99.70%** | 0.241 | 13.749 |
| `' '` (space)   | 18.75% | **99.70%** | 0.241 | 12.682 |
| `'z'` (rare)    | 0.87%  | **99.80%** | 0.241 | 15.292 |

**Takeaways**:
1. **The policy moves decisively toward the reward — the M17 gate is met.** From any
   starting frequency (rare `z` 0.9%, common `e` 9.8%, space 18.8%) REINFORCE drives the
   target to **~99.7%** of sampled tokens. The reward-weighted softmax-CE gradient does
   exactly what policy gradient promises: raise the probability of high-reward actions. The
   SFT frequencies independently track the corpus's natural letter statistics (a sanity
   check on the measurement: `e` ≈ 10%, space ≈ 19%, `z` ≈ 0.9%).
2. **The SFT→RL alignment tax is real and visible.** Maximizing the naive count reward is
   reward hacking — the policy collapses to emitting the target, so corpus bits/byte blows
   up **0.241 → 12.7–15.3** (the LM is destroyed). The tax scales **inversely with target
   rarity**: over-emitting rare `z` costs most (15.3) and the already-frequent space least
   (12.7). This is the canonical RL-vs-SFT trade-off at char scale — RL optimizes the
   reward it is given, not language modeling — and an honest caveat: a useful reward must
   encode what you actually want (valid text, a length/format target), not a degenerate
   count. PPO/GRPO + richer rewards are the documented heavier follow-on (ADR 0015).
3. **RL is a tiny, grad-checkable delta — attn11's wheelhouse.** No new forward, no new
   backward op, no checkpoint bump: the model stays a plain AR transformer and the RL image
   is a normal v5 checkpoint. The only new gradient surface is one scalar `(R − b)` scale on
   the seeded logit gradient, grad-checked three ways (`test_rl_op`: RL grad == advantage ×
   AR grad to rounding; FD vs the numeric gradient of `advantage × CE`; sign-flip + zero
   advantage). Reproduce: `scripts/m17-rl.sh [steps]` (deterministic; seed 1337; cross-arch).

## X025 — f64 on the GPU: mabda's native AMD path, proven bit-exact (M18 recon, 2026-06-19)

**Setup**: an M18 GPU-readiness probe (not a training run), done while cutting v1.7.3. The
question: now that mabda has moved 3.0.1 → **3.3.0**, does it actually provide what attn11's
GPU compute backend needs — and specifically, can the **f64 oracle run on the GPU**, which
the 2026-06-14 recon had assumed was blocked on an unshipped mabda v3.2 capability? Verified
two ways: (1) an adversarial audit of the 3.3.0 dist API surface + CHANGELOG, then (2) an
**on-hardware run** on this dev box — **AMD Cezanne (gfx90c)**, `amdgpu`/RADV,
`/dev/dri/renderD128` (mode 0666, no root), kernel 7.0.11, cyrius 6.2.27. mabda cloned at tag
**3.3.0** (HEAD `c09bc84`); each `programs/native_spirv_f64_*_e2e.cyr` built with `cyrius
build` and run. These are mabda's own conformance programs, but they exercise *exactly* the
public consumer path attn11 would use.

**The path under test** (launcher-free, pure-Cyrius — no wgpu, no C launcher, no libc):
`gpu_context_new_native()` opens the DRM render node → an f64 SPIR-V kernel is compiled
**in-tree** by mabda's `gfx9_compile` (SPIR-V → GFX9 ISA, scalar `V_*_F64`) → published to a
BO → dispatched via PM4/DRM → results read back and checked **bit-exact** against an
in-process CPU reference. (This is mabda's *own* SPIR-V→GFX9 emitter, landed mabda **3.2.12**
— **not** wgpu: portable wgpu compute is `MABDA_WGPU_COMPUTE=0` with a `GPU_ERR_OTHER`
dispatch stub, so portable f64 does not run. f64 on the GPU exists *only* on this native-AMD
route.)

**Result** — all 13 native f64 op-programs **exit 0, bit-exact on Cezanne**:

| program | f64 op surface | verdict |
|---------|----------------|---------|
| native_f64_fma | hand-authored `V_FMA_F64` (the F.4–F.6 baseline) | ✅ 8/8 bit-exact |
| native_spirv_f64_fma | **compiled** fma (`OpExtInst Fma` → `V_FMA_F64`) | ✅ 8/8, matches *fused* (≠ unfused CPU) |
| native_spirv_f64_arith | FADD / FMUL / FMIN / FMAX | ✅ 8/8 |
| native_spirv_f64_subabs | FSub + FAbs (src modifiers) | ✅ 8/8 |
| native_spirv_f64_div | OpFDiv (reciprocal-Newton, correctly rounded) | ✅ 8/8 |
| native_spirv_f64_sqrt | Sqrt (Newton-Goldschmidt, incl. 3 subnormals) | ✅ 8/8 |
| native_spirv_f64_const | f64 OpConstant operands (Horner poly) | ✅ |
| native_spirv_f64_select | ordered compare + OpSelect | ✅ |
| native_spirv_f64_cvt | f32 ↔ f64 round-trip | ✅ 8/8 |
| native_spirv_f64_i32_cvt | i32 ↔ f64 (round-toward-zero) | ✅ |
| native_spirv_f64_ldexp | Ldexp | ✅ |
| native_spirv_f64_vec | `array<dvec2>` add (vectorized) | ✅ 8/8 |
| **native_spirv_f64_layernorm** | **`(x-mean)/sqrt(var+eps)` via the PUBLIC API** | ✅ 8/8 bit-exact |

The `native_spirv_f64_fma` lane is the decisive one: lanes 0/1/6 the GPU result matches the
**fused** single-rounding FMA and *differs* from the unfused CPU `f64_add(f64_mul(a,b),c)` —
a CPU fallback could not produce that, so the f64 is genuinely executing on the silicon. The
**layernorm** is the attn11-shaped proof: a real transformer normalization (`FSub`+`FAdd`+
`OpConstant`+`Sqrt`+`FDiv`), bit-exact, through `gpu_shader_module_create_spirv` →
`gpu_compute_dispatch`.

**Takeaways**:
1. **The f64 oracle can live on the GPU — on AMD, proven here.** Every scalar f64 primitive
   attn11's tensor kernels need (add/mul/fma/div/sqrt/sub/abs/select/convert/ldexp/vec) runs
   bit-exact on the dev Cezanne, plus a full layernorm. The 2026-06-14 "f64 gated on an
   unshipped mabda capability" premise is **obsolete** — it shipped in mabda 3.2.12. ADR 0016
   + roadmap M18 amended accordingly.
2. **This path fits the charter *better* than the portable f32 path.** Native-AMD f64 is
   launcher-free + pure-Cyrius + bit-exact f64 — it preserves all three attn11 invariants
   (f64 oracle, dependency-free single-static-ELF, sovereign). The portable wgpu path is the
   opposite: f32-only **and** needs a C launcher (wgpu-native + libc), breaking the
   dependency-free charter. So on AMD, native f64 is a viable *first-class* M18 target, not a
   deferred follow-on; portable f32 is the route for non-AMD GPUs.
3. **Honest limits (none are blockers, all are known).** (a) **Scalar VALU** f64 — no matrix
   core (`V_MFMA_F64` absent; Cezanne is FP64-throttled anyway), so this is a
   **correctness/oracle** path, **not a speed win** at attn11's scale (same honest-negative
   footing as ternary X023). (b) **No f64 transcendentals** — `exp`/`log` are rejected by
   spirv-val on `double`; layernorm needs only sqrt/div (proven), but **GELU + softmax `exp`
   must be hand-rolled in-shader** as polynomials from the proven primitives — the one real
   f64-track work item mabda doesn't cover, squarely attn11's "transformer math is our
   domain." (c) **AMD GFX9 only today** — a *first-substrate* limit, not a ceiling: AMD is
   mabda's first GPU substrate; further variants (NVIDIA, Intel) are on mabda's roadmap, so
   attn11's native-f64 GPU coverage broadens as mabda's does, with no change to attn11's
   kernel layer.
4. **Pin `mabda = 3.3.0`** when the dep is added at 1.8.0 (clean superset: f32 WGSL pipeline
   from 2.4.0 + the f64 SPIR-V→GFX9 path from 3.2.12). Reproduce: clone mabda at tag 3.3.0,
   then `cyrius build programs/native_spirv_f64_layernorm_e2e.cyr build/ln && ./build/ln` on
   any AMD GFX9 box with a readable `/dev/dri/renderD128`. (Audit material + the 13-program
   sweep script: ephemeral, under `/tmp`.)

## X026 — M18 integration: mabda 3.4.1 clears the collision, the GPU path is buildable (M18, v1.7.4, 2026-06-19)

**Question.** X025 proved mabda's native-AMD f64 SPIR-V compute is bit-exact on the dev
Cezanne. The open follow-on: can attn11 actually *consume* mabda end-to-end — `include
"lib/mabda.cyr"` + bring the device up behind `--gpu` — now that mabda has cut **3.4.1** (the
release that fixes the `F64_HALF`/`F64_TWO` symbol collision filed during the 1.7.x staging)
and the installed cyrius is **6.2.29**? The staging note listed two blockers; this entry
resolves both empirically.

**Method.** An isolated-worktree integration probe (the realign tree stays clean): pin
`[deps.mabda] tag 3.4.1`, append mabda's stdlib needs
(`hashmap`/`tagged`/`fnptr`/`mmap`/`dynlib`/`sakshi`/`thread`/`sankoch`), `cyrius deps`,
`include "lib/mabda.cyr"` in `src/main.cyr`, wire `--gpu → gpu_probe()`, then `CYRIUS_DCE=1`
build and run — measuring binary size and the default-run loss with and without mabda.

**Result.**
- **mabda 3.4.1 resolves** — `cyrius deps` checks out tag 3.4.1; `lib/mabda.cyr` = 1,109,827 B
  (~1.06 MB dist); the upstream issue doc ships marked *RESOLVED — mabda 3.4.1*.
- **Gate 1 (the real one) — CLEARED.** mabda 3.4.1 renamed its constants to
  `MABDA_F64_HALF`/`MABDA_F64_TWO`, leaving the `math` stdlib sole owner of
  `F64_HALF`/`F64_TWO`. `include "lib/mabda.cyr"` now **builds clean** (first time ever — no
  "conflicting fn"), and the default no-`--gpu` run is **byte-identical** to the base binary
  — it reproduces the base run's finite loss trajectory bit-for-bit (0.31234 at step 250 down
  to 0.13807 at step 2000), no NaN. The collision that statically zeroed those
  constants → NaN in ganita tanh/GELU is permanently retired.
- **GPU online.** `--gpu → gpu_probe()` builds + runs on the Cezanne (gfx90c): *"native AMD
  compute device online (launcher-free, pure-Cyrius)"* + *"f64 SPIR-V compute supported"* —
  the X025 device path is reachable from attn11's own host context, not just mabda's tests.
- **Gate 2 — a transient binary-size cost, NOT a gate.** cyrius 6.2.29 has no
  `dep-module-call` (absent from `cyrius build --help` + a full grep of
  `~/.cyrius/versions/6.2.29`), so DCE NOPs mabda's unused code but still **amalgamates** the
  ~1 MB dist: base **373,504 B** → with-mabda **1,338,344 B** (+964,840, ×3.6) even when
  `--gpu` is unused. `dep-module-call` (call-without-amalgamate) is slated for the 6.2.x line
  and removes the bloat with **zero attn11 change** — it gates binary SIZE, not the milestone.

**Takeaway.** M18 is **buildable + functional end-to-end today** — the mabda gate is cleared
and the native AMD f64 device comes up. The only open item is binary leanness, which a later
6.2.x hands back for free, so it does not gate the GPU kernel work (the 1.8.x increments:
matmul → forward → backward + Adam). attn11 holds mabda **staged out of the default build**
purely to keep the lean single-static-ELF story (the B3 benchmark) intact until the landing
approach is chosen — un-stage now and accept the interim 1.3 MB binary, or wait for
`dep-module-call`. This supersedes X025's "pin `mabda = 3.3.0`": the correct pin is **3.4.1**
(the collision-fixed release). Reproduce: isolated worktree, mabda tag 3.4.1, cyrius 6.2.29,
`CYRIUS_DCE=1 cyrius build src/main.cyr` with/without the mabda include; sizes above are exact.

## X027 — M18 1.8.0: the `--gpu` forward matmul, bit-exact on-device (M18, v1.8.0, 2026-06-20)

**Question.** X026 proved the device comes up from attn11's host context. The milestone
question: can attn11 run a real **forward matmul on the GPU**, and does it match the CPU
oracle closely enough to ship? The roadmap framed 1.8.0 as the *portable f32* path validated
at f32 tolerance; this entry reports what actually shipped — the **native-AMD f64** path, and
at a far stronger tolerance.

**Method.** Land the GPU backend on `main` (un-stage mabda 3.4.1, `include "lib/mabda.cyr"`),
write `src/gpu.cyr` (`gpu_matmul_fwd`) and hook it at `qlinear_fwd` behind `--gpu`. Two
properties had to be discovered empirically on the dev Cezanne (gfx90c) before the kernel
shape was fixed, each proven with a throwaway spike then the committed harness:
(1) `gfx9_compile` rejects `OpLoopMerge`/`OpPhi` (straight-line only), and (2) mabda's public
native-compile path caps a module at **256 ids** (`NATIVE_SHADER_CAP_IDS`). Both force the
final design: an **unrolled** dot product, **host-tiled** over `k` (a bounded `GPU_TK=16`-term
read-modify-write kernel; host pre-fills `y` with the bias and loops `k0` in `ceil(K/16)`
serialized dispatches). Validate `gpu_matmul_fwd` vs rosnet `linear_fwd` across attn11's real
shapes (`tests/gpu_matmul.cyr`), and run a full `--gpu` training step end-to-end vs CPU.

**Result.**
- **Bit-exact, not merely f32-tolerant.** Across every attn11 shape — default
  16×32×{32,128}, MLP-down 16×128×32, preset 64×{64,256}, single-row decode 1×32×32 —
  `gpu_matmul_fwd` matches `linear_fwd` **byte-for-byte** (the GPU's scalar `V_MUL_F64`+
  `V_ADD_F64` in the same k-order as `linear_fwd`'s `f64v_fmadd` rounds identically). The
  roadmap's f32-tolerance gate is met at the *strongest* possible tolerance.
- **Engagement proven, not assumed.** `gpu_matmul_fwd` returns 0 only after a real
  dispatch+readback; the test fails on a non-zero (silent-fallback) return. The K=24
  (not a `GPU_TK` multiple) case correctly falls back to CPU. A default 30-step `--gpu` run
  dispatched **17,514** forward matmuls on-device.
- **A `--gpu` run reproduces the CPU run.** A 30- and 40-step `--gpu` training run produced a
  checkpoint **byte-identical** to the no-flag CPU run (forward on GPU is bit-exact →
  identical loss → identical Adam trajectory). The CPU stays the oracle; the GPU is an
  execution target with no numerical effect.
- **No regression.** **1056** grad-checks green on x86_64 **and** aarch64/qemu (mabda
  cross-compiles; the device is absent under qemu → the CPU fallback keeps the suite
  byte-identical); lint clean; fuzz + `make smoke` green; the no-flag run is byte-identical to
  the pre-M18 HEAD binary (proven by diffing checkpoints from a clean HEAD worktree build).

**Takeaway.** M18 1.8.0 ships the **sovereign GPU forward matmul** — generated f64 SPIR-V,
host-tiled to fit mabda's compile cap, bit-exact vs the CPU oracle, on the launcher-free
native-AMD path. It is "the GPU path works + is validated," exactly the honest-scale
deliverable the roadmap set (no speed claim at attn11's tiny scale — scalar VALU f64, no
matrix core; per-call host↔device transfer + `ceil(K/16)` dispatches dominate). The same
generated-SPIR-V + host-tiling pattern extends to the rest of the forward (1.8.1) and
backward + Adam (1.8.2). Reproduce: `make gpu-test` on an AMD GFX9 box (skips cleanly
elsewhere); `./build/attn11 --gpu --steps 40 --save g.ckpt` vs the same without `--gpu`,
then `cmp` the checkpoints. Invariants: [`../architecture/008-gpu-matmul-spirv.md`](../architecture/008-gpu-matmul-spirv.md); ADR 0016.

## X028 — M18 1.8.1: layernorm on GPU bit-exact, and the transcendental wall (M18, v1.8.1, 2026-06-20)

**Question.** 1.8.0 put the matmul on the GPU bit-exact. The 1.8.1 question: what is the
*rest of the forward*, and how much of it can stay bit-exact (preserving the byte-identical
`--gpu` invariant) on the native-AMD f64 path?

**Finding (the gating one): the transcendental wall.** Auditing the forward ops by their CPU
reduction order + math dependency:
- **`ln_fwd`** — **sequential** reductions (`mu += x`, `v += (x-mu)²`), only sqrt/div. A tiled
  sequential-sum GPU kernel matches it op-for-op → **bit-exact**. ✓
- **softmax** (attention scores, masked-CE) + **GELU** (`tanh` = `(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)`) — need
  **`f64_exp`**, which is an **x86 hardware builtin** (`lib/math.cyr` ships only an *aarch64
  polyfill*). Reproducing it bit-for-bit in SPIR-V is infeasible, and mabda's native path has
  no f64 transcendentals (spirv-val rejects `exp`/`log` on `double`). An in-shader polynomial
  exp matches only within a **tolerance** → would break byte-identity.
- **`head_fwd`** + the attention QK dots — bit-exact in principle, but use **SIMD 4-lane
  partials + a tree horizontal sum** (`f64v_fmadd` into `acc[4]`), a *different rounding
  order* than a sequential kernel.

So `ln_fwd` is the one forward op (besides matmul) that is cleanly bit-exact, and 1.8.1 ships
**only** it — keeping the invariant. softmax/GELU (in-shader exp, tolerance-validated) and
head/QK (tree-reduction kernel) are the next, separately-gated increment (1.8.2).

**Method.** A throwaway HW spike proved the 3-pass tiled layernorm bit-exact (M=4 C=32, then
M=8 C=64) before integration. Then `gpu_ln_fwd` landed in `src/gpu.cyr`: the 256-id
native-compile cap forbids a per-row unroll, so the contraction is **host-tiled** — pass1
tiled Σx → per-row `S` (RMW), host `mean = S/C`; pass2 tiled Σ(x−mu)² → `V` (reads mean), host
`rstd = 1/√(V/C+eps)`; pass3 elementwise normalize (1 thread per (m,c)). Per-row scalars are
computed host-side with `ln_fwd`'s exact ops; `mean`/`rstd` are written for the CPU backward.
Three generated SPIR-V kernels share a new `_gpu_pre` preamble; the GTT buffer pool grew 3→7
via a clean alloc/teardown pair. Hooked at the `ln_fwd` seam, g_gpu-gated + AGNOS-guarded.

**Result.**
- **Bit-exact** across attn11's real shapes — default 16×32, preset 64×64, single-row decode
  1×32, wide 8×128 — on y **and** the `mean`/`rstd` outputs (`tests/gpu_ln.cyr`,
  `make gpu-test`); engagement proven (a non-zero return = silent fallback = test failure);
  `C=24` (not a `GPU_TK` multiple) correctly falls back to CPU.
- **A `--gpu` run reproduces the CPU run** with **both** matmul and layernorm on-device: a
  30-step run dispatched **17,514 matmuls + 6,811 layernorms** and produced a checkpoint
  **byte-identical** to the no-flag run.
- **No regression:** the 1.8.0 matmul test re-passes (the 3→7 buffer-pool refactor is clean);
  **1056** grad-checks green on x86_64 **and** aarch64/qemu; AGNOS static-ELF builds clean
  (the GPU path guarded out, the `SYS_IOCTL` stub keeps the auto-prepended mabda compiling);
  lint + fuzz + `make smoke` green.

**Takeaway.** The bit-exact GPU forward now covers the two highest-frequency ops (matmul +
layernorm) and the byte-identical invariant holds. The transcendental wall is the real
boundary: making the *full* forward GPU-resident requires an in-shader exp and accepting
tolerance (not bit-exact) for softmax/GELU — the explicit subject of 1.8.2. Reproduce:
`make gpu-test`; `./build/attn11 --gpu --steps 40 --save g.ckpt` vs without `--gpu`, `cmp`.
Invariants: [`../architecture/009-gpu-layernorm-and-the-transcendental-wall.md`](../architecture/009-gpu-layernorm-and-the-transcendental-wall.md); ADR 0016.

## X029 — M18 1.8.2: crossing the transcendental wall — in-shader f64 exp + GELU (M18, v1.8.2, 2026-06-20)

**Question.** X028 named the wall: softmax/GELU need f64 `exp`, an x86 hardware builtin with
no bit-exact SPIR-V equivalent. Can an in-shader f64 `exp` be made accurate enough to put GELU
on the GPU at a *tolerance*, and how is that done without losing the byte-identical `--gpu`
invariant matmul (X027) + layernorm (X028) hold?

**Method.** Two throwaway HW spikes first: (1) an in-shader exp transliterated from
`lib/math.cyr`'s aarch64 `_f64_exp_polyfill` — Cody-Waite range reduction (`n = round(x/ln2)`
via the `1.5·2⁵²` magic-rounding trick, since mabda lowers no `Round` ext-inst), an 11-term
Taylor Horner for `exp(r)`, then `ldexp(p, n)` (GLSL Ldexp 53) — built only from the X025-proven
f64 primitives (FMul/FAdd/FSub/ConvertFToS/Ldexp), no device transcendental; (2) GELU composing
it twice (`tanh(z) = (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)`, FDiv proven). Then `gpu_gelu_fwd` landed in
`src/gpu.cyr` (one elementwise kernel; `_gpu_emit_exp` emitted inline twice — gfx9_compile is
single-function), hooked at `gelu_fwd` behind a **separate** gate `g_gpu_tc`/`--gpu-tc`.

**Result.**
- **Accuracy.** In-shader exp: **~2.3e-13** max relative error vs the CPU `f64_exp` over
  `[−16, 0]`. GELU: **~3e-14** absolute error vs `gelu_fwd` across default (2048) / preset
  (16384) / decode (128) widths (`tests/gpu_gelu.cyr`, `make gpu-test`). The gate is
  **allclose** (`|gpu−cpu| ≤ atol+rtol·|cpu|`, `atol=rtol=1e-10`) — pure relative error is the
  wrong metric near GELU's zero-crossings (`|ref|→0` blows it up while abs error stays ~1e-14).
- **Invariant preserved.** Plain `--gpu` does NOT engage GELU → its run stays **byte-identical**
  to the no-flag run (verified). GELU rides `--gpu-tc` only; engagement proven (a non-zero
  `gpu_gelu_fwd` return = silent fallback = test failure).
- **Statistical gate.** A `--gpu-tc` 500-step training run's loss (`0.30858 → 0.17790`) and eval
  bits/byte (`0.29942`) match the CPU run to 5-decimal print precision, never NaN — the ~1e-13
  per-op divergence is invisible at that precision. So `--gpu-tc` is numerically a faithful
  training run, just not byte-identical.
- **No regression:** matmul (X027) + layernorm (X028) tests re-pass; **1056** grad-checks green
  x86_64 + aarch64/qemu; AGNOS static-ELF clean (GPU path incl. GELU guarded out); lint + fuzz +
  smoke green.

**Takeaway.** The transcendental wall is crossed for elementwise GELU: an in-shader f64 exp from
proven primitives is ~2e-13-accurate, plenty for a tolerance-validated op, and the separate
`--gpu-tc` gate keeps the bit-exact `--gpu` invariant intact while offering the fuller forward.
`_gpu_emit_exp` is now the reusable foundation for **softmax** (1.8.3, which adds the max/sum
reduction tiling). Reproduce: `make gpu-test`; `./build/attn11 --gpu-tc --steps 500 --eval` vs
`--gpu`/CPU. Invariants: [`../architecture/010-gpu-transcendentals.md`](../architecture/010-gpu-transcendentals.md); ADR 0016.

## X030 — M18 1.8.3: the LM head on GPU (transposed-weight matmul, tolerance) (M18, v1.8.3, 2026-06-20)

**Question.** With matmul (X027) + layernorm (X028) bit-exact and GELU (X029) tolerance on
`--gpu-tc`, the next forward op is the LM head (`logits = f · tokembᵀ`). Can it reuse the matmul
machinery, and is it bit-exact or tolerance?

**Finding.** The head is a matmul but with two twists: (1) the tied embedding is contracted on
its **last** dim, so the weight index is `n·K+k` (transposed) not matmul's `k·N+n`; and (2) the
CPU `head_fwd_row` uses a **4-lane-partial + tree** reduction (the M9 SIMD head), not the
sequential per-output accumulation that made the QKV/O/MLP matmuls bit-exact (X027). So a
sequential GPU reduction matches the head only to ~1e-15 → **tolerance**, not bit-exact.
Replicating the tree order would need a 2-pass tiled-tree kernel (the 256-id cap blocks a
per-output unroll even at C=32) — deferred as not worth it; the sequential transposed-matmul on
the existing `--gpu-tc` tolerance gate is the low-cost path.

**Method.** `_gpu_build_tile_t` — a transposed-weight variant of the 1.8.0 `GPU_TK`-tile kernel
(weight index `n·K+k`, advance +1; x = f, RMW into logits, no bias). `gpu_head_fwd(f, emb,
logits, T, V, C)` reuses the matmul host path (upload, tiled dispatch, readback), hooked at
`head_fwd_n` behind `g_gpu_tc`.

**Result.**
- **Within tolerance** across attn11's real head shapes — default (T16·V25·C32), preset
  (T64·V256·C64), max-BPE (T64·V768·C64) — worst absolute error **~4.5e-13** vs `head_fwd` via
  allclose (`atol=rtol=1e-10`); `tests/gpu_head.cyr` (`make gpu-test`). Engagement proven
  (non-zero return = silent fallback = failure).
- **Statistical:** a `--gpu-tc` 500-step run (now matmul + ln bit-exact + GELU + head tolerance)
  matches the CPU run's loss (`0.17790`) and eval bits/byte (`0.29942`) to print precision,
  never NaN — 153090 matmuls + 59535 layernorms + 25515 gelus + 8012 heads dispatched.
- **Invariant intact:** plain `--gpu` does NOT engage the head → byte-identical to the no-flag
  run (verified). No regression: matmul/ln/gelu tests re-pass; **1056** grad-checks x86_64 +
  aarch64/qemu; AGNOS clean; lint + fuzz + smoke green.

**Takeaway.** The LM head joins the GPU forward cheaply by reusing the matmul tiling with a
transposed weight — tolerance (the head's SIMD-tree reduction order), so it rides `--gpu-tc`.
The only remaining forward op is **softmax** (attention scores + masked-CE) — the fine-grained,
fused-attention piece (1.8.4), reusing `_gpu_emit_exp` + a max/sum reduction. Reproduce:
`make gpu-test`; `./build/attn11 --gpu-tc --steps 500 --eval` vs `--gpu`/CPU. Invariants:
[`../architecture/010-gpu-transcendentals.md`](../architecture/010-gpu-transcendentals.md); ADR 0016.
