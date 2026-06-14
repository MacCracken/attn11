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
#    (attn11 caps a corpus at 64 MB (1.5.3); the source — C4 — is 305 GB. A few MB
#    already saturates a tiny model, so this example slices a small 4 MB sample.)
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

## Curating the sample (1.5.2)

A tiny model is data-rich and capacity-poor, so **data quality > volume**:
`--curate` adds (all stdlib, deterministic) de-duplication (exact + a normalized
prefix), **multi-shard sampling** (`--shards N` spreads across the crawl for
diversity, not one consecutive block), and prose/register filters (letter & digit
ratios, terminal punctuation, average word length, repetition, long-token spam —
dropping tables, listings, and url/hash spam). Defaults (no `--curate`, one shard)
reproduce the raw slice byte-for-byte, so it stays a clean A/B baseline.

```sh
# curated 4 MB — quality filter on a focused (few-shard) corpus: the win for a TINY
# model (X017: same shard, filter on vs off, eval bits/byte 3.43 -> 3.23, -5.9%).
python3 scripts/c4_sample.py --curate --shards 1 --out data/c4-en-curated.txt --max-bytes 4000000
./build/attn11 --corpus data/c4-en-curated.txt --bpe 256 --steps 2000 --save data/c4.ckpt --eval

# diversity (more shards) is a SCALE lever, not a tiny-model one — it raised
# bits/byte here (+2.7%): a 53 K-param model can't exploit registers it can't fit.
# Reach for --shards N once the model is big enough to absorb the variety (M16+).
python3 scripts/c4_sample.py --curate --shards 8 --out data/c4-en-diverse.txt --max-bytes 4000000
```

**X017's lesson: curate for quality now, diversity later.** C4 is already
deduplicated at creation (so the dedup is a no-op here — it earns its keep on noisier
sources); the measured win is the **prose-quality filter** (dropping tables /
listings / url-hash spam). Multi-shard *diversity* hurts the tiny model's bits/byte
because it can't absorb the extra variety — diversity/volume pays off with model
scale, which is why the roadmap sequences streaming + larger corpora with M16+.

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
English-ish* off raw web text it has never been told the rules of.

## Scaling up: the `--preset` (1.5.1, x86_64, BPE 256, 1500 steps)

The preset (ctx 64 / d_model 64 / 8 heads / 4 layers — ~40 words of context, 232 K
params) is a clear step toward fluency over the default config, on the *same* 4 MB
C4 slice:

| metric | default (ctx 16, 53 K) | **preset (ctx 64, 232 K)** |
|--------|------------------------|----------------------------|
| train loss | 4.90 (600 steps) | **3.82 (1500 steps)** |
| eval CE/token · **bits/byte** | 4.836 · 3.433 | 3.796 · **2.695** |
| checkpoint | — | 5.58 MB |

Preset sample (temperature 0.8):

```
a transformer feneffacting and memgram wits made untustory, and usix dateverse and
vist looking court and loge will meast make nights of Jologian fure these moil sehe
in createds, itord to posess datcharvially cosing gover tract of your hombwrum and
```

More function words and word-shaped chunks survive (`and ... made ... looking court
and ... will ... make nights of ... in ... to posess ... of your ...`) — the longer
context and bigger model roughly **0.74 bits/byte** lower than the default. Greedy
still collapses to a frequent phrase (`... service and service and ...`); temperature
is the readable view. Fluency keeps coming with more model, more context, and a
bigger step/data budget — the **token-packing (1.5.3) + curation-at-scale (1.5.4)**
data arc and the **M16+** capacity work are where that crossover lives.

## Curation at scale (1.5.4, X019)

1.5.3's packed token store raised the corpus cap 4 MB → 64 MB, so we can finally
curate a corpus bigger than a tiny model has seen and ask: **does more clean data pay
off, and does a bigger model use it better?** Curate a **24 MB / 12-shard** corpus
(6× the old cap) and compare both model sizes against the 4 MB curated baseline, all
BPE 256, matched compute:

```sh
# the scaled corpus (multi-shard quality-curated; ~7 MB downloaded, ~11 s)
python3 scripts/c4_sample.py --curate --shards 12 --out data/c4-curated-24mb.txt --max-bytes 24000000
python3 scripts/c4_sample.py --curate --shards 1  --out data/c4-curated-4mb.txt  --max-bytes 4000000
# four cells: {default, --preset} x {4 MB, 24 MB}
./build/attn11 --corpus data/c4-curated-4mb.txt          --bpe 256 --steps 600  --eval
./build/attn11 --corpus data/c4-curated-24mb.txt         --bpe 256 --steps 600  --eval
./build/attn11 --corpus data/c4-curated-4mb.txt  --preset --bpe 256 --steps 1500 --eval
./build/attn11 --corpus data/c4-curated-24mb.txt --preset --bpe 256 --steps 1500 --eval
```

Eval bits/byte (lower = better):

| model | 4 MB curated | 24 MB curated | data Δ (4→24 MB) |
|-------|--------------|---------------|------------------|
| default (≈53 K) | 3.232 | 3.405 | **+5.4%** |
| **preset (≈232 K)** | 2.666 | 2.741 | **+2.8%** |
| capacity Δ | −17.5% | **−19.5%** | |

**The diversity/volume penalty halves with capacity.** On its OWN corpus a bigger,
more diverse corpus reads as *higher* bits/byte (more entropy to fit) — but the tiny
model pays +5.4% (and its samples get more garbled), while the preset pays only +2.8%
and stays fluent with **richer vocabulary**, and its capacity gain is actually larger
on the diverse corpus (−19.5%). So the bigger model extracts more from richer data —
the first sign that diversity/volume starts paying off with scale. The 4 MB default
cell reproduces X017's 3.232 exactly (the curation + the 1.5.3 packed store are both
deterministic/transparent). preset-24mb temp-0.8 sample:

```
a transformer commoss. Nonal can see fintermence pleous we post your impect of the
weeken misies of a charchery of the compage. Yous free new to dother like take suppa
and inally sease the percing. Got weatime anirey, you were you were to juse of need
```

**Caveat (the honest part):** bits/byte here is measured on each model's *own*
corpus, which penalizes the higher-entropy 24 MB set in absolute terms — so it can't
show a clean "more data → better generalization" win. The fair test is a **held-out
cross-corpus eval** (train on A, eval on a disjoint B); that needs a small additive
`--eval-corpus` flag (a deferred 1.5.x follow-on — 1.5.4 is binary-unchanged). The
crossover where more clean data clearly wins lives there + at **M16+** capacity. Full
write-up: [X019](../development/experiments.md).

## Streaming a large corpus without loading it (1.6.2)

The in-memory `--corpus` path caps the corpus at 64 MB (the packed store vs the 256 MB
single-allocation cap). For a bigger corpus, **pre-encode it to a token-shard once** and
**stream** it — RAM then stays O(model + one 64 K-token chunk), independent of corpus
size (a GB shard trains in ~13 MB). Two ways to make a shard:

```sh
# (a) from a curated text corpus already on disk (byte-level; ≤ 64 MB), via attn11:
./build/attn11 --corpus data/c4-curated-24mb.txt --encode-shard data/c4-24mb.tsh
# (b) GB-scale, straight from C4 in bounded RAM (byte-level), via the sampler:
python3 scripts/c4_sample.py --curate --shards 64 --max-bytes 1000000000 \
        --out data/c4-1gb.txt --emit-shard data/c4-1gb.tsh
# then train/eval from the shard — the tokens never load into RAM:
./build/attn11 --stream-corpus data/c4-24mb.tsh --preset --steps 2000 --eval
```

The shard is **self-describing** (it carries its own tokenizer), so `--stream-corpus`
needs no other flags; it is mutually exclusive with `--corpus`/`--stdin`/`--bpe`/`--load`.
A streamed run is **byte-identical** to the equivalent in-memory run (verified on x86_64
and aarch64), and `c4_sample.py --emit-shard` produces a shard byte-identical to
`attn11 --encode-shard` on the same text. BPE shards: use `attn11 --bpe K --encode-shard`
(≤ 64 MB; whole-corpus merge statistics don't stream, so the sampler's GB path is
byte-level). Format + invariants:
[`../architecture/007-streaming-token-shards.md`](../architecture/007-streaming-token-shards.md).
