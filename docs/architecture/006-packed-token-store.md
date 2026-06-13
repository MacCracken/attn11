# 006 — The packed corpus token store (`g_data`)

> What's true about the code, not why we chose it (that's the CHANGELOG / the
> 1.5.x roadmap arc). Read this before touching `g_data`, `corpus_set`,
> `bpe_learn`, `tok_encode`, or the window samplers.

## The one packed buffer

`g_data` is the encoded corpus — one token id per position, `g_datalen` tokens.
It is the **only** buffer whose size scales with the corpus, so it is the only
one that is packed. Before 1.5.3 it was one **i64 per token** (8 B); now it is a
byte buffer of `g_datalen * g_data_w` bytes:

- `g_data_w == 1` — **u8**, byte-level tokenizer (vocab ≤ 256, max id 255).
- `g_data_w == 2` — **u16**, BPE (vocab ≤ 768, max id 767).

Everything else stays i64: the per-window batch buffers `A_tokens` / `A_targets`
/ `A_mask` / `G_win` and the prompt-encode scratch `g_enc_buf` are all `g_T`-sized
(≤ 8192), so packing them buys nothing. The BPE tables (`g_merges`, `g_tok_len`,
`g_tok_bytes`, `g_paircnt`) are vocab-sized, not corpus-sized. **Do not pack them**
— the asymmetry is deliberate.

## Accessors — never index `g_data` raw

Token-id access goes through the width-generic helpers in `train.cyr`:

- `_tw_ld(buf, i, w)` / `_tw_st(buf, i, w, v)` — width `w ∈ {1, 2, 8}`. Loads
  **zero-extend** (token ids are non-negative); stores write only the low `w`
  bytes. Built on `load8`/`load16`/`load64` + `store8`/`store16`/`store64` (the
  same idioms `lib/cffi`, `lib/slice`, `lib/net`, `lib/dynlib` use).
- `gd_ld(i)` / `gd_st(i, v)` — `g_data` at the active `g_data_w`. **`i` is a token
  index, not a byte offset** — identical semantics to the old
  `load64(g_data + i*8)`, so every call site's index math is unchanged. This is
  why the loss curve stays **byte-identical**: the same token ids reach the same
  forward; only the storage layout changed.

A grep for `load64(g_data` / `store64(g_data` / `g_data + .* * 8` should return
**nothing** in `src/` (one comment aside). New code reads `g_data` via `gd_ld`.

## Who sets the width, and when

`g_data_w` must be set **before** `corpus_set` (which sizes + fills the packed
buffer from it). `corpus_set` does **not** modify it.

- **`main`** sets it before the corpus load: `1` byte-level, `2` if `--bpe`, `2` if
  `--load` (a checkpoint may carry a BPE tokenizer whose corpus rebuild re-encodes
  ids up to 766 into `g_data` — see below).
- The **grad-check tests** call `build_corpus(); bpe_learn(...)` directly and leave
  `g_data_w` at its default `1`; `bpe_learn` self-widens (next section). The default
  global value `1` means the no-flag run and the early byte-level tests exercise the
  **u8** path.

## `bpe_learn` self-widens u8 → u16

BPE mints ids `base .. base+K-1` (≤ 255 + 511 = **766**), which overflow a u8 store.
`bpe_learn` therefore requires u16 and, if it finds `g_data_w == 1`, **widens in
place once**: allocate `g_datalen * 2`, copy each u8 id to u16, set `g_data_w = 2`.

- In production `main` pre-sizes u16 for `--bpe`, so the widen never runs (no leak).
- The widen path exists for the direct `build_corpus(); bpe_learn()` test/bench
  callers; the abandoned u8 buffer is small and one-time (the bump allocator doesn't
  reclaim it).

## `tok_encode` carries an explicit output width

`tok_encode(text, len, out, ow)` writes ids into `out` at width `ow`. Two callers,
two widths — keep them straight:

- **prompt encoding** (`generate` / `gen_diffusion`) writes the i64 scratch
  `g_enc_buf` → `ow = 8` (byte-identical to the pre-1.5.3 behaviour).
- **checkpoint resume** (`persist.cyr`, the corpus rebuild from retained raw bytes)
  writes the packed `g_data` → `ow = g_data_w`. Because `--load` forces
  `g_data_w = 2`, the rebuild buffer is u16 and holds any BPE vocab.

## Capacity math (why `MAX_CORPUS_BYTES` is 64 MB)

The binding limit is the **256 MB single-allocation cap** (`ALLOC_MAX`), per buffer,
not total heap. With the 64 MB corpus cap:

- `g_text` (raw bytes, retained for BPE re-encode) — one ≤ 64 MB alloc.
- `g_data` — u8 ≤ 64 MB, **u16 ≤ 128 MB** (half the cap; the margin is deliberate).

So a byte-level corpus could in principle go to ~256 MB and a BPE one to ~128 MB
before a single allocation hits the cap; `MAX_CORPUS_BYTES` is held at 64 MB for
headroom (corpus + model both fit comfortably in RAM). An over-cap corpus rejects
cleanly through the existing size-cap path (`secure_read_file` / the stdin cap+1
probe), code −2 — never a crash.

## Invariants to preserve

1. **Byte-identical default run** — packing must not change any token id the model
   sees. `gd_ld` returns the same value `load64` did; the samplers' RNG draws and
   index math are unchanged.
2. **Width is the token store's, for its lifetime** — set once per corpus before
   `corpus_set`; it persists until the next corpus load (re-decided then).
3. **u8 only ever holds ids ≤ 255; u16 holds ≤ 767.** byte-level vocab is ≤ 256 by
   construction; BPE forces u16.
4. **Only `g_data` is packed.** Mixing a packed batch buffer with the i64 forward
   would silently corrupt embedding lookups.
