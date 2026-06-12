# Getting started with attn11

attn11 trains a small char-level GPT-style transformer from scratch in Cyrius —
forward pass, hand-derived backprop, and Adam, all on raw `f64` arrays.

## Build & run

```sh
cyrius deps                                # resolve stdlib deps
cyrius build src/main.cyr build/attn11     # compile (static ELF)
./build/attn11                             # train on the embedded corpus, then sample
cyrius test                                # gradient-check suite + smoke tests
```

## CLI

```
attn11                       train on the embedded corpus, then sample
attn11 --corpus PATH         train on a corpus file (O_NOFOLLOW, 4 MB cap)
attn11 --stdin               train on a corpus read from stdin
attn11 --steps N             total training steps (also the LR-schedule horizon)
attn11 --save PATH           write a crash-atomic checkpoint after training
attn11 --load PATH           resume from a checkpoint (+ --corpus to continue training)
attn11 --gen-only            skip training, just sample
attn11 --preset              ctx 64 / d_model 64 / 8 heads / 4 layers (default: ctx 16)
attn11 --heads N             override attention heads (must divide d_model)
attn11 --kv-heads N          override K/V heads (< heads = GQA, 1 = MQA)
attn11 --layers N            override transformer blocks
attn11 --bpe K               learn K BPE merges first (1..512; default is byte-level)
attn11 --eval                print CE/token + bits-per-byte after training/save
```

Config flags (`--preset`/`--heads`/`--kv-heads`/`--layers`/`--bpe`) shape a
**fresh** model; under `--load` the checkpoint's config and tokenizer win.
Magnitude caps (enforced in `model_config_ok`, else a clean abort): d_model
≤ 4096, ctx ≤ 8192, `--layers` 1..128, vocab ≤ 768; `--heads` must divide
d_model and `--kv-heads` must divide `--heads`. See
[`../STABILITY.md`](../STABILITY.md) for the full frozen surface.

Example — train on your own text, checkpoint, then resume:

```sh
attn11 --corpus mytext.txt --steps 2000 --save run.ckpt
attn11 --load run.ckpt --corpus mytext.txt --steps 4000   # resume to 4000
attn11 --load run.ckpt --gen-only                         # sample from it
```

Example — the scale preset with BPE, compared on bits-per-byte:

```sh
attn11 --preset --corpus mytext.txt --steps 4000 --eval            # byte-level
attn11 --preset --bpe 256 --corpus mytext.txt --steps 4000 --eval  # BPE
```

Sampling is **KV-cached** (0.7.0): the prompt prefills per-layer K/V caches
and each generated token costs one cached row instead of a window recompute
(~6× faster at ctx 16, ~23× at the ctx-64 preset; bit-identical to the
uncached reference — see ADR 0005). Checkpoints from earlier formats (v1
≤ 0.6.0, v2 = 0.7.0) still load; saves write v3 (which records the
tokenizer — see ADR 0006).

## Layout

- `src/main.cyr` — CLI + orchestration; `src/{tensor,fileio,ops,attn,model,train,persist}.cyr` — the model.
- `tests/attn11.tcyr` — gradient-check + persistence + robustness suite (`cyrius test`).
- `tests/attn11.fcyr` — fuzz harness for the loaders.
- `tests/attn11.bcyr` — benchmarks (`./scripts/bench-history.sh`).
- A minimal embedding-of-the-API example: [`../examples/minimal_train.cyr`](../examples/minimal_train.cyr).

## aarch64

The model cross-builds and runs under qemu (grad checks pass on both arches):

```sh
cyrius build --aarch64 tests/attn11.tcyr build/test_a64 && qemu-aarch64 build/test_a64
```

## Adding a feature

1. Edit/add a module under `src/` and `include` it from `src/main.cyr`.
2. If it adds a backward op, **grad-check it** in `tests/attn11.tcyr` (finite
   differences) — a backward without a passing grad check is incomplete.
3. `cyrius test`; benchmark with `./scripts/bench-history.sh` if it touches a hot path.
4. Bump `VERSION`, update `CHANGELOG.md` and `docs/development/state.md`; add an
   ADR ([`../adr/template.md`](../adr/template.md)) for any non-trivial choice.

See [`../sources.md`](../sources.md) for the citation map and
[`../architecture/001-tensors-and-floats.md`](../architecture/001-tensors-and-floats.md)
for the f64/SIMD conventions.
