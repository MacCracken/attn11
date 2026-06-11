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
