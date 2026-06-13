# Example — teaching attn11 English from C4 (stream → train → speak)

The "large external dataset" workflow: sample real English prose out of
**[C4](https://www.tensorflow.org/datasets/catalog/c4#c4en_default_config)** (the
Colossal Clean Crawled Corpus — the `c4/en` TFDS config), train a tiny transformer
on it, and watch it produce English-flavored text. This is the worked example behind
experiment [X016](../development/experiments.md).

## Why not the TFDS pipeline directly

The TFDS `c4/en` config is **305 GB** and its preparation runs an Apache-Beam job
over the raw crawl — the wrong tool for materializing a few MB of text for a tiny
model, and it needs `tensorflow` + `tensorflow_datasets` installed. attn11 instead
reads a plain UTF-8 corpus file (`--corpus`), so all we need is *some* C4 text in a
file. [`scripts/c4_sample.py`](../../scripts/c4_sample.py) gets it with **zero
third-party deps** (stdlib `gzip` + `json` only): C4's records are public gzipped
JSON-lines shards (`allenai/c4` is the raw mirror of the exact corpus TFDS
catalogs), so the script streams ONE shard and stops once it has enough text —
downloading only ~the compressed bytes backing the sample (a few MB), not the
319 MB shard, let alone the full dataset. Same corpus, none of the heavyweight
machinery.

## Recipe

```sh
# 1. Sample ~4 MB of English from C4 (en) into a gitignored data file.
#    (attn11 caps a corpus at 4 MB; the source — C4 — is 305 GB, we slice from it.)
python3 scripts/c4_sample.py --out data/c4-en-sample.txt --max-bytes 4000000

# 2. Train: BPE subword tokenizer (so the model predicts word-pieces, not bytes)
#    over the C4 sample, checkpoint, and report the tokenizer-comparable metric.
./build/attn11 --corpus data/c4-en-sample.txt --bpe 256 --steps 2000 --save data/c4.ckpt --eval

# 3. Sample from the checkpoint (no training).
./build/attn11 --load data/c4.ckpt --gen-only

# Bigger/better English: the --preset (ctx 64 / d_model 64 / 4 layers) covers ~40
# words of context instead of ~10, at ~17x the per-step cost.
./build/attn11 --corpus data/c4-en-sample.txt --preset --bpe 256 --steps 1500 --save data/c4-preset.ckpt --eval
```

`data/` is **gitignored** (the sample + any shards are large and regenerable) — the
corpus never enters the source tree. The run is deterministic (seed 1337, no
external RNG beyond the C4 download): retraining on the same sample reproduces
bit-for-bit.

## What to expect

attn11's reference models are *tiny* (tens of thousands to a few hundred thousand
parameters). On 4 MB of diverse web English they learn English **statistics** —
real words, common subwords, plausible local structure — but NOT fluent sentences.
The honest result is **English-flavored word-salad**: recognizable words and
fragments stitched with subword noise. Greedy decoding tends to collapse to the
most frequent tokens ("and", "the", " "); temperature sampling is far more
interesting. "Speaking English" at this scale means *English-shaped*, not coherent —
fluency is a model-capacity (and training-budget) story, not a pipeline one.

## Measured (1.5.1, x86_64, default config + BPE 256, 600 steps)

| metric | value |
|--------|-------|
| C4 sample | 4,002,896 bytes (one stream, ~0.3 s, ~1 MB downloaded) |
| vocab (byte base + 256 BPE merges) | 438 |
| params | 52,704 |
| corpus tokens (BPE) | 1,969,733 |
| train loss | 5.29 (step 250) → 4.90 (step 500) |
| eval CE/token · **bits/byte** | 4.836 · **3.433** |

## Sample (temperature 0.8)

```
a transformer arduns famt m, resianit conin the loltinaitjae bfre heelsaral be of
poluring a you clonwetkes froms whisuly 20gam nurenten ... gent ho the geyf eet
tecudiis it pemired in the blench have leicing alle cidforesionw pateat
```

Recognizable English threads through it — `the`, `be of`, `a you`, `for`, `have`,
`it`, `in the`, `20...` — between the wobbling subwords. The model is *speaking
English-ish* off raw web text it has never been told the rules of. Scaling the
model (`--preset`), the data, and the step budget tightens the wobble; that crossover
is the open follow-on (and the M16+ capacity story).
