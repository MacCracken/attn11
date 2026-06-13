# 0013 — Char-diffusion training objective (masked absorbing-state)

**Status**: Accepted
**Date**: 2026-06-13

## Context

Through M14 the 1.x arc varied *architecture* (attention/KV, FFN density, the
sequence-mixer family) while keeping the autoregressive next-token objective. M15
(E5) is the first **training-objective** departure: a masked absorbing-state
**diffusion** model on the same trunk, to test the survey's "diffusion LMs are
super data learners" thesis at reference scale. The constraints: it must be opt-in
and **additive** (the no-flag AR run byte-identical, every older checkpoint still
loads), every new hand-derived backward must pass a finite-difference grad-check,
and the AR-vs-diffusion comparison must be *honest* (the two objectives are not
trivially comparable).

A masked-diffusion model differs from the AR trunk in three ways that force design
choices: it needs a representation for the `[MASK]` corruption token; it attends
**bidirectionally** (no causal mask); and its loss is over the *masked* positions
only. Each interacts with attn11's invariants (the weight-tied LM head, the packed
`g_params` layout, the byte-identical default run, the checkpoint format).

## Decision

Add `--objective diffusion` (default `ar`): a masked absorbing-state diffusion
objective (D3PM-absorbing / MDLM lineage) on the existing MHA trunk. Scope for
v1.5.0 is deliberately narrow — **MHA only (GQA allowed), learned-absolute
positions, dense MLP, uniform (no hybrid)** — so M15 lands one idea behind a tight
grad-check. The load-bearing choices:

1. **`[MASK]` = a learned embedding vector** (`mask_emb`, width `C`), appended to
   the param block *after* `lnf_b`, present only when diffusion. The weight-tied
   head stays exactly `V`-wide and untouched; `g_NP` grows by `C` only under
   diffusion, so an AR model's layout / RNG draw order / Adam state / v5–v6
   checkpoints are byte-identical. `mask_emb` rides `g_params`, so Adam, grad-clip,
   zero-grads, and serialization (all of which iterate `g_NP`) handle it for free.

2. **Bidirectional attention via a `g_bidir` global** (default 0 = causal). The
   causal mask is purely the `j <= i` loop bound in `attn_core_fwd`/`_bwd`; under
   `g_bidir == 1` it becomes `j <= T-1` (the `Pc` buffer is already `nh·T·T`). The
   softmax-backward `dotPdP` sum uses the same range as the forward. `g_bidir` is 1
   only for MHA-diffusion, so the shared core stays causal for MLA/AR.

3. **Masked cross-entropy: unweighted mean over the masked positions.**
   `softmax_xent_masked_fwd/bwd` reuse the softmax-CE math with a per-position mask
   and a `1/nmask` normalizer. The principled **1/t ELBO weight is NOT applied to
   training** — it has unbounded variance as t→0 on a tiny corpus and is unfair at
   matched compute; it is reported at eval (and the MDLM 1/t weight cancels the
   t-scaling of the masked count, so the *unweighted* per-masked-token CE averaged
   over t IS the NELBO bound).

4. **Three explicit per-position arrays** — `A_tokens` (originals), `A_mask`
   (1=masked), `A_targets` (originals) — so the input never diverges from the
   target via a sentinel, and `mask_emb` (masked positions) vs `tokemb` (given
   positions) gradients route correctly through `embed_bwd`. A ≥1-mask floor keeps
   `1/nmask` finite (the train-loop NaN guard would otherwise abort).

5. **Confidence-ordered parallel decode** (MaskGIT-style): an uncached
   *bidirectional* full-window forward each round (the causal KV cache is AR-only),
   greedy with a frozen lowest-index confidence tie-break — fully deterministic and
   cross-arch reproducible.

6. **Checkpoint v7**: one `objective` field at slot [22] + the `+C` mask_emb in NP.
   A diffusion image writes v7; uniform AR still writes v5 and a hybrid AR v6 —
   both byte-identical to before M15. v≤6 synthesize `objective = 0`. Code `-47`
   rejects a hostile v7 (objective ∉ {0,1}, or diffusion paired with mla/lin/ssm/
   rope/MoE).

## Consequences

- **Positive**: a grad-checked diffusion objective on the shared trunk, zero
  regression to the AR path, full checkpoint back-compat, and an *honest*
  AR-vs-diffusion comparison axis (X015). The mask_emb-in-`g_params` choice means
  the optimizer/clip/persist machinery needed no diffusion special-casing.
- **Negative**: diffusion mode is O(T²) attention (full square vs causal's
  half-triangle); a small param bump (`+C`) under diffusion; greedy decode at tiny
  scale collapses toward high-frequency tokens (a known dLLM-at-small-scale
  limitation, not a correctness issue — the grad-checks are tight).
- **Neutral**: MLA/MoE/RoPE-diffusion and a stochastic/temperature decode are clean
  additive fast-follows. lin/ssm are inherently causal recurrences and have no
  bidirectional form without a redesign (rejected for diffusion).

## Alternatives considered

- **`[MASK]` as a `V+1` tied vocab class** — rejected: it changes `g_V`, the head
  output width, and `g_NP`, and forces the softmax to suppress a mask logit at
  every position; breaks AR byte-identity.
- **A fixed-zero mask vector (no learned param)** — rejected: marginally more
  minimal but less faithful (BERT/MDLM/MaskGIT use a learned mask token), and the
  v7 machinery is needed for the `objective` field regardless.
- **The 1/t ELBO weight as the training loss** — rejected for training (unbounded
  variance as t→0 on a tiny corpus, unfair at matched compute); reported at eval.
- **A separate `mask_emb` allocation outside `g_params`** — rejected: it would
  force diffusion special-cases in Adam, grad-clip, zero-grads, and the checkpoint
  (four silent-failure surfaces) vs the one threaded `objective` param.
- **Bidirectionalizing the cached single-row decode** — rejected: the KV cache is
  causal/AR-only; diffusion decode uses the uncached full-window path instead.
