# Tensors and floats in attn11

Non-obvious truths about how the numerics work. (For *why* decisions were made,
see `docs/adr/`; this file is *what is true about the code*.)

## Everything is i64; an f64 is a bit pattern

Cyrius has no float type. An `f64` value is its 64-bit IEEE-754 pattern carried
in an ordinary i64 register. Arithmetic never uses `+`/`*` — those are integer
ops. Instead it goes through builtins:

```
f64_add  f64_sub  f64_mul  f64_div  f64_neg
f64_sqrt f64_abs  f64_floor f64_round
f64_exp  f64_ln   f64_tanh  f64_pow      # from lib/math.cyr
f64_lt   f64_gt   f64_eq   f64_le  f64_ge # comparisons return i64 0/1
f64_from(int) -> f64bits                  # integer -> double
f64_to(f64bits) -> int                    # double -> integer (truncates)
```

Consequence: `var s = a + b;` on two f64 values is a **silent bug** — it adds
the bit patterns as integers. Always use `f64_add(a, b)`.

## A tensor is a flat f64 array

A "tensor" is just a heap pointer (`alloc`) to a contiguous, **row-major** run
of f64 values. Element `i` lives at byte offset `i*8`. Shapes are not stored;
they're passed as `i64` args (`T`, `C`, …). Helpers in `src/tensor.cyr`:
`tget`/`tset`, `t_alloc`, `t_zero`, `t_copy`, `t_axpy`, `t_scale`, `t_randn`.

2-D indexing is explicit: element `(r, c)` of an `R×C` tensor is at
`base + (r*C + c)*8`.

## Forward writes, backward accumulates

Convention across `ops.cyr` / `attn.cyr` / `model.cyr`:

- forward functions **write** their outputs (and cache what backward needs);
- backward functions **accumulate** into parameter gradients (`dW`, `db`,
  `dgamma`, …) and **write** input gradients (`dx`).

So a training step zeroes all parameter grads once (`model_zero_grads`), runs
forward+backward over the batch accumulating grads, then steps Adam. A `0`
(null) pointer passed where a bias is expected means "no bias".

## No per-step allocation

All parameters, optimizer state, activation caches, and backward temporaries
are allocated **once** in `model_init` (shapes are fixed for a run). The
allocator is a bump allocator with no free, so allocating inside the training
loop would grow memory without bound — the forward/backward paths therefore
touch only pre-allocated buffers. Attention packs all its caches and temporaries
into a single arena (`attn_arena_size`, carved by the `_at_*` offset helpers).

## Parameters are one flat vector

Every trainable weight is packed into one vector `g_params`; `g_grads`,
`g_adam_m`, `g_adam_v` share the identical layout. Each logical weight
(`P_Wq`, `P_tokemb`, …) is a sub-pointer into `g_params`, its gradient the
same offset into `g_grads`. Adam then updates the whole vector in one loop.
The token embedding is **weight-tied**: it serves as both the input embedding
and the output LM head, so its gradient accumulates from both code paths.

## Gradient checking is the correctness gate

Hand-derived backprop is easy to get subtly wrong. Every backward function is
verified against **central finite differences** in `tests/attn11.tcyr`:
perturb each input by ±ε, recompute a scalar loss, compare `(L₊−L₋)/2ε` to the
analytic gradient. Pass threshold is max relative error `< 1e-5` per op
(`< 1e-4` for the full model, where finite-difference noise is larger).

## Gotchas observed in the toolchain

- **Long float literals mis-parse.** `0.5` and `0.044715` are fine, but a
  many-digit literal like `0.7978845608` parsed to a wrong value. Build precise
  constants from integer ratios or runtime math instead (e.g.
  `sqrt(2/pi) = f64_sqrt(f64_div(f64_from(2), F64_PI))`).
- **`print(s, len)` counts bytes, not characters.** Multi-byte UTF-8 (e.g. the
  em-dash `—` = 3 bytes) makes a hand-counted length wrong; use `puts(s)` which
  calls `strlen`.
- **`>>` is a logical shift**; the PRNG relies on that to keep values
  non-negative.
