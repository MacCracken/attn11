# attn11 — Stable surface (frozen at v0.9.0, for v1.0.0)

> This is the **frozen surface**: the user-facing contract attn11 commits to at
> v1.0.0. Past v1.0 it is **additive-only** — no flag is removed or repurposed,
> no checkpoint format stops loading, and defaults do not change in a way that
> alters a no-flag run. New flags / formats / config knobs may be *added* with
> backward compatibility. Anything not listed here (internal function names,
> arena layout, the exact sample text of a given checkpoint) is **not** part of
> the contract and may change.

## CLI flags (frozen, runtime)

All 19 behavioral flags below are stable. A run with **no flags** trains on the
embedded corpus and samples; that behavior is frozen. (Flags added past 1.0 are
additive — the post-1.0 rule: a no-flag run stays byte-identical regardless.)

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
| `--attn-kind K` | sequence mixer: `mha` (default), `mla` (latent KV, 1.2.0), `lin` (gated linear, 1.4.0), `ssm` (selective SSM, 1.4.2); `mla`/`lin`/`ssm` force full heads |
| `--latent-dim N` | MLA latent width `d_c` / SSM state size `N` (`1..d_model`; MLA default `d_model/2`, SSM default 16); only with `--attn-kind mla`/`ssm` (1.2.0/1.4.2) |
| `--attn-every K` | per-layer mixer hybrid: a full-attention (MHA) block every K-th layer, the `--attn-kind` base elsewhere (any mix of mha/mla/lin/ssm; 1.4.3/1.4.4) |
| `--pos-kind K` | positions: `learned` (default), `rope` (coupled, mha/gqa, even head dim) (1.2.2), or `rope-decoupled` (mla) (1.2.3) |
| `--rope-dim N` | decoupled-RoPE channel width `d_rope` (even, `2..d_model`; default ~`hd/2`); only with `--pos-kind rope-decoupled` (1.2.3) |
| `--experts N` | MoE: N experts per block (`1` = dense baseline, byte-identical; `1..256`) (1.3.0) |
| `--expert-topk K` | active experts per token (default 2; `1..N`); needs `--experts > 1` (1.3.0) |
| `--bpe K` | learn K BPE merges first (1..512; byte-level is the default) |
| `--eval` | print CE/token + bits-per-byte after training/save |

Plus `--help`/`-h` and `--version` (informational). The parser **rejects
unknown arguments** and a value-flag given without a value — both exit non-zero
with usage. Config flags (`--preset`/`--heads`/`--kv-heads`/`--layers`/
`--attn-kind`/`--latent-dim`/`--attn-every`/`--pos-kind`/`--rope-dim`/`--experts`/
`--expert-topk`/`--bpe`) shape a **fresh** model; under `--load` the checkpoint's
config and tokenizer win. Magnitude caps (frozen, enforced in `model_config_ok`):
d_model ≤ 4096, ctx ≤ 8192, layers ≤ 128, vocab ≤ 768; `heads | d_model`,
`kv-heads | heads`; MLA/SSM/lin force `kv-heads = heads` and (MLA/SSM) `1 ≤ d_c ≤
d_model`; coupled RoPE requires an even head dim on `mha`/`gqa`; decoupled RoPE
requires `mla` and an even `2 ≤ d_rope ≤ d_model`; MoE requires `1 ≤ experts ≤ 256`
and `1 ≤ expert-topk ≤ experts`; a hybrid (`--attn-every`) requires learned-abs
positions and full heads.

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

Native-endian i64 blob; records the tokenizer (since v3), the architecture
descriptor (since v4), the MoE descriptor (since v5), and — for a per-layer hybrid
— the per-layer mixer kinds (v6). **v1 (≤0.6.0), v2 (0.7.0), v3 (1.0/1.1), v4
(1.2.x), and v5 (1.3.0–1.4.2) all still load** and always will — backward
load-compatibility is part of the contract. The save format advances additively
(the post-1.0 additive-only rule): a **uniform** model writes **v5** (the
`attn_kind`/`pos_kind`/`latent_dim`/`rope_dim` descriptor, ADR 0007, +
`num_experts`/`expert_topk`, ADR 0008); a **per-layer hybrid** writes **v6** (the
NL per-layer mixer kinds, ADR 0011/0012). `attn_kind` ∈ {mha, mla (1.2.0), lin
(1.4.0), ssm (1.4.2)}, the `pos_kind` RoPE variants (1.2.2/1.2.3), MoE
(`num_experts`/`topk`, 1.3.0), and the v6 hybrid pattern are all accepted. A
default-descriptor v5 (`mha`/`learned`/dense) is a byte-identical resume of a v4.
The *exact save version* is not itself frozen (it advances additively); what is
frozen is that older images keep loading. Checkpoints are **not** portable across
architectures (raw native-endian `f64`); that is by design (ADR 0004) and frozen.
The hostile-input validation surface (codes documented in `src/persist.cyr`) is
frozen in behavior, additive in new codes (`-40..-43` the v4 descriptor, `-44`/
`-45` the v5 MoE descriptor, `-46` an invalid v6 per-layer kind).

## Not part of the contract

Internal symbol names, the activation-arena layout, the exact bytes/sample text
a given checkpoint produces (it can shift if a kernel's SIMD reduction order
changes within tolerance), benchmark numbers, and the toolchain pin
(`cyrius.cyml`) — these are implementation details, free to change.
