# attn11 — Stable surface (frozen at v0.9.0, for v1.0.0)

> This is the **frozen surface**: the user-facing contract attn11 commits to at
> v1.0.0. Past v1.0 it is **additive-only** — no flag is removed or repurposed,
> no checkpoint format stops loading, and defaults do not change in a way that
> alters a no-flag run. New flags / formats / config knobs may be *added* with
> backward compatibility. Anything not listed here (internal function names,
> arena layout, the exact sample text of a given checkpoint) is **not** part of
> the contract and may change.

## CLI flags (frozen, runtime)

All 14 behavioral flags below are stable. A run with **no flags** trains on the
embedded corpus and samples; that behavior is frozen.

| flag | meaning |
|------|---------|
| `--corpus PATH` | train on a corpus file (`O_NOFOLLOW`, 4 MB cap) |
| `--stdin` | train on a corpus read from stdin |
| `--steps N` | total training steps (also the LR-schedule horizon) |
| `--save PATH` | write a crash-atomic checkpoint after training |
| `--load PATH` | resume from a checkpoint (+ `--corpus` to continue training) |
| `--gen-only` | skip training, just sample |
| `--preset` | ctx 64 / d_model 64 / 8 heads / 4 layers (the scale config) |
| `--heads N` | attention heads (must divide d_model) |
| `--kv-heads N` | K/V heads (must divide heads; `1` = MQA; default = heads) |
| `--layers N` | transformer blocks (1..128) |
| `--attn-kind K` | attention variant: `mha` (default) or `mla` (latent KV); `mla` forces full heads (1.2.0) |
| `--latent-dim N` | MLA latent width `d_c` (`1..d_model`; default `d_model/2`); only with `--attn-kind mla` (1.2.0) |
| `--bpe K` | learn K BPE merges first (1..512; byte-level is the default) |
| `--eval` | print CE/token + bits-per-byte after training/save |

Plus `--help`/`-h` and `--version` (informational). The parser **rejects
unknown arguments** and a value-flag given without a value — both exit non-zero
with usage. Config flags (`--preset`/`--heads`/`--kv-heads`/`--layers`/`--bpe`)
shape a **fresh** model; under `--load` the checkpoint's config and tokenizer
win. Magnitude caps (frozen, enforced in `model_config_ok`): d_model ≤ 4096,
ctx ≤ 8192, layers ≤ 128, vocab ≤ 768; `heads | d_model`, `kv-heads | heads`.

## Config knobs (compile-time constants — frozen defaults)

These live in `src/main.cyr` (`CFG_*`) and are **not** flag-exposed; changing
them requires a rebuild. Their *values* are the frozen defaults:

| knob | value | flag override |
|------|-------|---------------|
| d_model `C` | 32 (preset 64) | via `--preset` |
| ctx `T` | 16 (preset 64) | via `--preset` |
| heads | 4 (preset 8) | `--heads`, `--preset` |
| kv-heads | = heads | `--kv-heads`, `--preset` |
| layers | 3 (preset 4) | `--layers`, `--preset` |
| steps | 2000 | `--steps` |
| seed | 1337 | — (rebuild) |
| batch | 16 | — (rebuild) |
| log-every | 250 | — (rebuild) |
| LR warmup | 100 | — (rebuild) |
| LR (base → min) | 3e-3 → 3e-4, cosine | — (rebuild) |
| grad-clip (global-norm) | 1.0 | — (rebuild) |
| attention bias | on | — (rebuild) |
| residual dropout | 0.0 | — (rebuild) |

## Checkpoint format (frozen contract: load-compatibility)

Native-endian i64 blob; records the tokenizer (since v3) and the architecture
descriptor (since v4). **v1 (≤0.6.0), v2 (0.7.0), and v3 (1.0/1.1) all still
load** and always will — backward load-compatibility is part of the contract.
The save format advances additively (the post-1.0 additive-only rule): saves
currently write **v4** (adds the reserved `attn_kind`/`pos_kind`/`latent_dim`/
`rope_dim` descriptor, ADR 0007, defaulting to the current MHA / learned-absolute
model — a default-descriptor v4 is a byte-identical resume of a v3). The *exact
save version* is not itself frozen (it advances additively); what is frozen is
that older images keep loading. Checkpoints are **not** portable across
architectures (raw native-endian `f64`); that is by design (ADR 0004) and frozen.
The hostile-input validation surface (codes documented in `src/persist.cyr`) is
frozen in behavior, additive in new codes (`-40..-43` are the v4 descriptor).

## Not part of the contract

Internal symbol names, the activation-arena layout, the exact bytes/sample text
a given checkpoint produces (it can shift if a kernel's SIMD reduction order
changes within tolerance), benchmark numbers, and the toolchain pin
(`cyrius.cyml`) — these are implementation details, free to change.
