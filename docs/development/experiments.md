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
