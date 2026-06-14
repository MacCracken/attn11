# 007 — Streaming token-shard ingestion (`--stream-corpus`)

> What's true about the code, not why we chose it (that's the CHANGELOG / the
> 1.6.x roadmap group). Read this before touching `gd_ld`, `stream_tok`, the
> `g_stream*` globals, `src/stream.cyr`, or `file_seek`. Builds on
> [006 — the packed corpus token store](006-packed-token-store.md).

## The shape of streaming

A **token-shard** is a pre-encoded, self-describing corpus on disk: a fixed
header, the tokenizer (byte vocab + BPE merges, the same bytes a checkpoint
carries), then the packed token ids (`width` bytes/token, exactly the in-memory
`g_data` layout). `attn11 --encode-shard PATH` writes one from a loaded corpus;
`attn11 --stream-corpus PATH` trains/evals from one **without loading the tokens
into RAM**. `scripts/c4_sample.py --emit-shard` is the GB-scale (byte-level)
producer. The format + writer + reader live in `src/stream.cyr`.

RAM is then `O(model + one chunk)`, independent of corpus size: a GB shard trains
in ~13 MB where the in-memory path (capped at the 256 MB single-allocation cap)
cannot even hold it.

## The one invariant: `gd_ld` is the only token reader

Every corpus-token read — `sample_window`, `sample_window_diffusion`,
`eval_corpus`, `eval_diffusion` — goes through **`gd_ld(i)`** (006). The corpus
WRITE paths (`corpus_set`, `bpe_learn`) do not run in streaming mode (the shard is
already encoded). So streaming is a **read-through cache inside `gd_ld`** and
nothing else changes:

```
fn gd_ld(i): i64 {
    if (g_stream != 0) { return stream_tok(i); }   # streamed: chunk cache
    return _tw_ld(g_data, i, g_data_w);            # in-memory: unchanged
}
```

`stream_tok(i)` returns **exactly** what `_tw_ld(g_data, i, g_data_w)` would for an
in-memory corpus of the same tokens, so `sample_window` / `eval_corpus` / the model
are untouched and **byte-identity is structural** — not something re-verified at
each call site. The branch is never taken on the default path (`g_stream == 0`), so
the no-flag run is byte-identical (verified default / preset / BPE / ternary /
diffusion / ssm-hybrid).

## The chunk cache

`stream_tok` holds ONE cached token range `[g_stream_lo, g_stream_lo + g_stream_clen)`
in `g_stream_buf` (`g_stream_cap` = `STREAM_CHUNK()` = 65536 tokens, sized at the max
width, allocated once and reused across opens). A miss (re)loads `[i, i+cap)` —
clamped to `g_datalen` — in one read:

- **file source** (`g_stream_fd >= 0`): `file_seek(fd, g_stream_doff + lo*width)`
  then `file_read` the chunk. `file_seek` (`src/fileio.cyr`) is a raw `lseek`(SEEK_SET)
  arch-dispatched like `_fdatasync` (x86_64 `8` / aarch64 `62`); the stdlib ships no
  wrapper.
- **memory source** (`g_stream_fd < 0`): copy the chunk from `g_stream_membuf` — the
  hermetic test path, exercising the *same* cache logic with no syscalls.

`cap > 8192` (the max ctx) guarantees a whole window `[s, s+T]` never straddles two
chunks: the first access `gd_ld(s)` loads `[s, s+cap)`, and since `s ≤ g_datalen−T−1`
the chunk holds ≥ `T+1` tokens, so the rest of the window hits the cache. One read
per window (fewer for `eval_corpus`'s sequential stride). This is why the cache is a
single range, not an LRU.

## The shard format (`src/stream.cyr`)

All i64 native-endian (the supported targets — x86_64, aarch64 — are little-endian;
a shard, like a checkpoint, is a local artifact consumed on the arch that wrote it):

```
[0] magic = SHARD_MAGIC ("ATTNSH01")  [1] version = 1   [2] tok_kind (0 byte/1 BPE)
[3] Vb (base vocab 1..256)  [4] K (merges 0..512)  [5] V (= Vb + K)
[6] width (1 byte-level / 2 BPE)       [7] ntokens
[8 .. 8+Vb)            base vocab (id -> byte 0..255)
[8+Vb .. 8+Vb+2K)      merge pairs (left, right), learned order
then ntokens * width raw bytes of packed token ids   (offset g_stream_doff)
```

`shard_open` reads + validates the **metadata region** (header + tokenizer blob)
only; the token data stays on disk. The token data offset `g_stream_doff` =
`(8 + Vb + 2K) * 8`.

## Hostile-input safety

`shard_validate_buf` is the single validation authority (shard_open and the fuzz
harness both route through it). It bounds every read against the buffer length, then
checks — BEFORE the tokenizer is installed or any token data is touched — the
`(kind, Vb, K, V)` triple, the width/tokenizer agreement, and the merge table as a
**well-founded DAG with bounded expansion** (`shard_validate_tok`, mirroring the
audited checkpoint loader's `-37`/`-38` block; the BPE format is frozen). `shard_open`
additionally validates `ntokens` against the **real file size** (`fstat`), so a
lying/truncated header is rejected (`-68`) and a window seek can never read past EOF.
Error codes are a distinct **`-60..-73`** range (the checkpoint loader uses `..-49`):

| code | meaning | code | meaning |
|------|---------|------|---------|
| -60 | bad magic / too short / short read | -67 | ntokens < 1 |
| -61 | bad version | -68 | file too small for ntokens (truncated/lying) |
| -62 | bad tok_kind | -69 | base-vocab byte out of 0..255 |
| -63 | Vb out of 1..256 | -70 | merge id out of range / not a DAG |
| -64 | K out of 0..512 | -71 | merge expansion exceeds `BPE_MAX_TOKLEN` |
| -65 | V != Vb + K | -72 | allocation failure |
| -66 | width / merge–kind mismatch | -73 | agnos (streaming is Linux-only) |

## Scope and fast-follows

- **Composes with everything** (transparently, via `gd_ld`): byte + BPE, all
  mixers, MoE, hybrids, the diffusion objective, `--eval`, `--eval-corpus`.
  `eval_corpus_buf` disables streaming for the held-out pass (the held-out file is
  in RAM) and restores it.
- **Mutually exclusive** with `--corpus`/`--stdin`/`--bpe` (the shard *is* the
  corpus + tokenizer) and, for now, with `--load`.
- **Linux-only** — `lseek`; the agnos target rejects `shard_open` (`-73`).
- **Fast-follows** (documented, additive): resume-from-stream (`--load` +
  `--stream-corpus`), and BPE GB-scale shards (whole-corpus merge statistics do not
  stream, so the c4 emitter is byte-level; `attn11 --bpe --encode-shard` handles the
  ≤ 64 MB BPE case).
