# Example — the vidya corpus pipeline (train → checkpoint → sample)

The end-to-end "curated small corpus" workflow against a tagged build: train
the scale `--preset` on real Cyrius source (the [vidya](https://github.com/MacCracken/vidya)
content corpus), checkpoint it, then sample. This is the worked example behind
experiments [X001](../development/experiments.md) (first real-content run) and
[X003](../development/experiments.md) (byte-vs-BPE at iso-compute).

## Recipe

```sh
# 1. Build the corpus: every vidya/content/**/*.cyr, concatenated (488,489 bytes).
find ../vidya/content -name '*.cyr' | sort | xargs cat > vidya.txt

# 2. Train the preset (ctx 64 / d_model 64 / 8 heads / 4 layers) for 4000 steps,
#    checkpoint, and report the tokenizer-comparable metric.
./build/attn11 --corpus vidya.txt --preset --steps 4000 --save vidya.ckpt --eval

# 3. Sample from the checkpoint (no training).
./build/attn11 --load vidya.ckpt --gen-only
```

It is fully deterministic (seed 1337, no external RNG): the run reproduces
bit-for-bit, and `--load` resumes bit-for-bit.

## Measured (0.9.0, x86_64, preset, 4000 steps)

| metric | value |
|--------|-------|
| vocab (byte-level, adaptive) | 117 |
| params | 211 648 |
| train loss | 2.12 (step 250) → **1.089** (step 4000) |
| eval CE/token · **bits/byte** | 1.220 · **1.760** |
| checkpoint size | ~5.0 MB |
| wall time | a few minutes on CPU |

The preset reaches a lower loss than X001's tiny config (1.55 at 8000 steps on
the same corpus) in half the steps — the ctx-64 window covers whole statements,
not 16-char fragments (the lever X001 identified).

## Sample (temperature 0.8, from the checkpoint)

```
a transformer Chand in slable `address as astart
# dection accumulator, carithm ====
# Cand -1 replayer bytes if allocation al (0x10)
```

Greedy output is syntactically Cyrius-shaped (section-divider comments `# ===`,
hex literals `0x10`, the allocator/bytes/accumulator idiom); words wobble but
the *texture* is right — expected for a 0.2 M-param char model on a small
real-entropy corpus.

## Variant — BPE (shorter sequences, lower bits/byte)

```sh
./build/attn11 --corpus vidya.txt --preset --bpe 256 --steps 4000 --save vidya-bpe.ckpt --eval
```

Adds a 256-merge BPE tokenizer over the byte base vocab (ADR 0006); the
checkpoint carries the tokenizer (format v3). At iso-compute, BPE reaches
~11–13 % lower bits-per-byte than byte-level on this corpus — see
[X003](../development/experiments.md) for the controlled comparison.
