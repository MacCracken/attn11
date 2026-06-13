# Sources

> Every algorithm, formula, and constant in attn11 traces to a published
> source. A reviewer should be able to follow any piece of the model back to its
> origin and verify the implementation against it. Inline citations live on the
> declaring functions; this file is the index.

## Model architecture

### Transformer — scaled dot-product & multi-head self-attention, positional embeddings
**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N.,
Kaiser, Ł., Polosukhin, I. (2017).** "Attention Is All You Need." *NeurIPS 2017.*
arXiv:[1706.03762](https://arxiv.org/abs/1706.03762).
Used in: `src/attn.cyr` (scaled dot-product attention `score = (Q·K)/√d`,
causal mask, multi-head split), `src/model.cyr` (learned positional embeddings).

### Decoder-only LM, pre-norm block placement, residual-projection init scaling
**Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019).**
"Language Models are Unsupervised Multitask Learners." *OpenAI technical report
(GPT-2).*
[pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
Used in: `src/model.cyr` — pre-norm decoder block, and the §2.3 residual-layer
weight scaling by `1/√(2·n_layers)` applied to the attention-output and MLP
projection weights at init.

### Pre-norm Transformer (LayerNorm before each sub-layer)
**Xiong, R., Yang, Y., He, D., et al. (2020).** "On Layer Normalization in the
Transformer Architecture." *ICML 2020.* arXiv:[2002.04745](https://arxiv.org/abs/2002.04745).
Used in: `src/model.cyr` — justifies the `LayerNorm → sublayer → residual`
ordering (more stable than the original post-norm).

### Weight tying (input embedding == output projection)
**Press, O., Wolf, L. (2017).** "Using the Output Embedding to Improve Language
Models." *EACL 2017.* arXiv:[1608.05859](https://arxiv.org/abs/1608.05859).
**Inan, H., Khosravi, K., Socher, R. (2017).** "Tying Word Vectors and Word
Classifiers." *ICLR 2017.* arXiv:[1611.01462](https://arxiv.org/abs/1611.01462).
Used in: `src/model.cyr` (`head_fwd`/`head_bwd`) — the token embedding doubles
as the LM head; its gradient accumulates from both paths.

### Multi-query attention (shared K/V heads)
**Shazeer, N. (2019).** "Fast Transformer Decoding: One Write-Head is All You
Need." arXiv:[1911.02150](https://arxiv.org/abs/1911.02150).
Used in: `src/attn.cyr` — the `nkv = 1` end of the `n_kv_heads` config: all
query heads share a single K/V head, cutting KV-cache memory `nh`-fold.

### Grouped-query attention (GQA)
**Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F.,
Sanghai, S. (2023).** "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints." *EMNLP 2023.*
arXiv:[2305.13245](https://arxiv.org/abs/2305.13245).
Used in: `src/attn.cyr`, `src/model.cyr` — `1 < nkv < nh`: each group of
`nh/nkv` query heads shares one K/V head (`Ckv = nkv·hd`-wide K/V
projections, grouped `dK`/`dV` accumulation in the hand-derived backward).

### Multi-head latent attention (MLA — low-rank K/V latent)
**DeepSeek-AI (2024).** "DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model." arXiv:[2405.04434](https://arxiv.org/abs/2405.04434).
Used in: `src/attn.cyr` (`attn_mla_fwd`/`attn_mla_bwd`), `src/model.cyr`
(`--attn-kind mla`) — K/V are factored through a shared low-rank latent
(down-projection `C → d_c`, up-projections `d_c → C`) instead of projected from
`x` directly, with full heads (`nkv = nh`). The reference MLA defaults to
learned-absolute positions; coupled (1.2.2) and the faithful decoupled (1.2.3)
RoPE variants ship too (`--pos-kind`, see the RoPE entry above and ADR 0007). The
latent down/up projections reuse the grad-checked `linear` backward, so the MLA
gradient adds no novel hand-derived math.

### Rotary positional embeddings (RoPE — coupled, relative positions)
**Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021).** "RoFormer:
Enhanced Transformer with Rotary Position Embedding." arXiv:[2104.09864](https://arxiv.org/abs/2104.09864).
Used in: `src/attn.cyr` (`rope_apply_fwd`/`rope_apply_bwd`), `src/model.cyr`
(`--pos-kind rope`) — a position-dependent rotation of Q and K (interleaved
dimension pairs `(2k, 2k+1)` rotated by `m·θ_k`, `θ_k = 10000^(-2k/hd)`) so the
score depends only on the relative offset `m-n`. Mutually exclusive with the
learned absolute embeddings (you pick one; ADR 0007). Parameter-free, so the
only new gradient is the rotation's transpose; the cos/sin are computed without
the x86-only trig builtins (Maclaurin on `θ_k ∈ (0,1]` + complex binary
exponentiation — see `docs/architecture/005`). Coupled RoPE is the dense-MHA/GQA
rung; the **decoupled** variant for MLA (`--pos-kind rope-decoupled`, 1.2.3,
`attn_dec_core_*`/`attn_mla_dec_*`) carries position on a separate `d_rope` channel
per DeepSeek-V2 (arXiv:2405.04434, above) — the score splits into content + a
shared-key rope term scaled by `1/sqrt(hd + d_rope)`.

### Mixture of Experts (sparse FFN — top-K router + load-balance aux)
**Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.
(2017).** "Outrageously Large Neural Networks: The Sparsely-Gated
Mixture-of-Experts Layer." *ICLR 2017.* arXiv:[1701.06538](https://arxiv.org/abs/1701.06538).
**Fedus, W., Zoph, B., Shazeer, N. (2021).** "Switch Transformers: Scaling to
Trillion Parameter Models with Simple and Efficient Sparsity." *JMLR 2022.*
arXiv:[2101.03961](https://arxiv.org/abs/2101.03961) — the load-balance auxiliary
loss `α·N·Σ fᵢ·Pᵢ` attn11 uses (`moe_aux_*`).
**Jiang, A.Q., Sablayrolles, A., Roux, A., et al. (2024).** "Mixtral of Experts."
arXiv:[2401.04088](https://arxiv.org/abs/2401.04088) — the renormalized top-K
softmax combine (softmax over the selected logits) attn11 uses for the gate
weights (`moe_fwd`/`moe_bwd`).
Used in: `src/ops.cyr` (`moe_fwd`/`moe_bwd` router + combine, `moe_aux_*`),
`src/model.cyr` (`_mlp_weight_size`, `_o_expert`/`_o_Wgate`, the MLP branch on
`g_num_experts > 1`). `--experts N --expert-topk K`; the discrete top-K pick is a
frozen lower-index tie-break (bit-reproducible cross-arch), differentiated
straight-through; see ADR 0008 for the combine/balance/dense-invariant decisions.

### KV-cache inference (cache K/V per position, one row per decoded token)
**Pope, R., Douglas, S., Chowdhery, A., et al. (2022).** "Efficiently Scaling
Transformer Inference." *MLSys 2023.* arXiv:[2211.05102](https://arxiv.org/abs/2211.05102).
Used in: `src/attn.cyr` (`attn_fwd_row`), `src/model.cyr` (`model_fwd_row`,
per-layer `KV_K`/`KV_V`) — generation appends each position's K/V to a cache
and attends over it instead of recomputing the window. (The cache idea is
folklore-old — it falls out of the incremental decoder in Vaswani et al. —
but this paper is the standard reference for KV-cache memory accounting,
which the GQA config and the bench's `kv cache bytes` line implement.)

### Context-shift for a full window under absolute positional embeddings
**Radford et al. (2019)** (GPT-2, above) — stride-style re-evaluation when a
fixed window slides; **Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M.
(2024).** "Efficient Streaming Language Models with Attention Sinks." *ICLR
2024.* arXiv:[2309.17453](https://arxiv.org/abs/2309.17453) — the modern
treatment of why cached rows cannot simply shift positions.
Used in: `src/train.cyr` (`gen_decode`) — when the window fills, drop the
oldest `T/2` tokens and re-prime the kept context at its new positions (one
window recompute amortized over `T/2` tokens); see ADR 0005 for why shifting
the cache in place is unsound with learned absolute positions.

## Tokenization

### Byte-pair encoding (BPE) for subword vocabularies
**Sennrich, R., Haddow, B., Birch, A. (2016).** "Neural Machine Translation of
Rare Words with Subword Units." *ACL 2016.*
arXiv:[1508.07909](https://arxiv.org/abs/1508.07909).
Used in: `src/train.cyr` (`bpe_learn`/`tok_encode`/`bpe_build_spans`) — the
iterative most-frequent-adjacent-pair merge algorithm, applied over the
byte-level base vocab (the byte-level layering follows GPT-2's byte-level BPE,
Radford et al. 2019, above). attn11 freezes its own deterministic tie-break
(row-major ascending argmax) and greedy left-to-right encoding — see ADR 0006.

## Layers & activations

### Layer normalization
**Ba, J.L., Kiros, J.R., Hinton, G.E. (2016).** "Layer Normalization."
arXiv:[1607.06450](https://arxiv.org/abs/1607.06450).
Used in: `src/ops.cyr` (`ln_fwd`/`ln_bwd`) — normalize over the feature dim, then
scale/shift by learned γ/β; backward per the paper's derivation.

### GELU activation (tanh approximation)
**Hendrycks, D., Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)."
arXiv:[1606.08415](https://arxiv.org/abs/1606.08415).
Used in: `src/ops.cyr` (`gelu_fwd`/`gelu_bwd`). The tanh approximation
`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` and the constant `0.044715` are from
that paper (also the form used by GPT-2).

### Dropout (inverted residual dropout)
**Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.
(2014).** "Dropout: A Simple Way to Prevent Neural Networks from Overfitting."
*JMLR* 15(56):1929–1958. <https://jmlr.org/papers/v15/srivastava14a.html>.
Used in: `src/ops.cyr` (`dropout_gen_mask`/`dropout_apply`/`dropout_bwd`) —
inverted dropout (kept units scaled by `1/(1−p)` so the forward expectation is
unchanged and inference needs no rescaling); applied to the residual stream,
config-gated, and disabled in eval/generation so grad checks stay deterministic.

### Softmax cross-entropy (max-subtraction for numerical stability)
**Goodfellow, I., Bengio, Y., Courville, A. (2016).** *Deep Learning*, MIT Press,
§4.1 (softmax stability) and §6.2.2 (cross-entropy). <https://www.deeplearningbook.org/>.
Used in: `src/ops.cyr` (`softmax_xent_fwd`/`_bwd`) — subtract the row max before
`exp`; the combined softmax+cross-entropy backward is `softmax − onehot`.

## Optimization

### Adam optimizer
**Kingma, D.P., Ba, J. (2015).** "Adam: A Method for Stochastic Optimization."
*ICLR 2015.* arXiv:[1412.6980](https://arxiv.org/abs/1412.6980).
Used in: `src/model.cyr` (`model_adam_step`). Bias-corrected first/second moment
estimates; defaults β₁=0.9, β₂=0.999, ε=1e-8 are the paper's.

### Cosine learning-rate decay (with linear warmup)
**Loshchilov, I., Hutter, F. (2017).** "SGDR: Stochastic Gradient Descent with
Warm Restarts." *ICLR 2017.* arXiv:[1608.03983](https://arxiv.org/abs/1608.03983).
Linear warmup follows the Transformer schedule of Vaswani et al. (2017) above.
Used in: `src/train.cyr` (`lr_at`).

### Gradient clipping by global norm
**Pascanu, R., Mikolov, T., Bengio, Y. (2013).** "On the difficulty of training
recurrent neural networks." *ICML 2013.* arXiv:[1211.5063](https://arxiv.org/abs/1211.5063).
Used in: `src/model.cyr` (`model_clip_grads`) — rescale all gradients when the
global L2 norm exceeds the threshold.

## Random number generation (weight init & sampling)

### xorshift64 PRNG
**Marsaglia, G. (2003).** "Xorshift RNGs." *Journal of Statistical Software*,
8(14). doi:[10.18637/jss.v008.i14](https://doi.org/10.18637/jss.v008.i14).
Used in: `src/tensor.cyr` (`rng_u64`) — the (13, 7, 17) shift triple.

### splitmix64 seed mixing
**Steele, G.L., Lea, D., Flood, C.H. (2014).** "Fast Splittable Pseudorandom
Number Generators." *OOPSLA 2014.*
doi:[10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195).
Used in: `src/tensor.cyr` (`rng_seed`) — the finalizer constants
`0x9E3779B97F4A7C15`, `0xBF58476D1CE4E5B9`, `0x94D049BB133111EB` scramble the
seed so small/sequential seeds don't produce correlated streams.

### Normal sampling — Marsaglia polar method
**Marsaglia, G., Bray, T.A. (1964).** "A Convenient Method for Generating Normal
Variables." *SIAM Review*, 6(3), 260–264.
doi:[10.1137/1006063](https://doi.org/10.1137/1006063).
Used in: `src/tensor.cyr` (`rng_normal`) — rejection sampling in the unit disc;
chosen over Box–Muller because it needs no `sin`/`cos` (which this toolchain
lacks).

## Verification

### Finite-difference gradient checking (central differences)
**Nocedal, J., Wright, S.J. (2006).** *Numerical Optimization*, 2nd ed.,
Springer, §8.1 (finite-difference derivative approximation). Standard practice
for verifying hand-derived backprop.
Used in: `tests/attn11.tcyr` — `(L(x+ε) − L(x−ε)) / 2ε` compared against the
analytic gradient (O(ε²) accurate), the correctness gate for every backward op.
