#!/usr/bin/env python3
"""Stream a small English sample out of the C4 (en) corpus into a plain UTF-8
text file for attn11's `--corpus` path — with optional QUALITY CURATION (1.5.2).

NO third-party deps — stdlib `gzip` + `json` + `hashlib` + `urllib` only (the box
this targets has no pip / no tensorflow / no tensorflow_datasets). C4's TFDS `c4/en`
config is 305 GB and needs an Apache-Beam preparation; that is the wrong tool for a
few MB of text. Instead we stream C4's public gzipped JSON-lines shards (allenai/c4
is the raw mirror of the exact corpus TFDS catalogs) and STOP once --max-bytes of
text is collected — so the download is only ~the compressed bytes backing the sample
(a few MB), not the 319 MB shard, let alone the full dataset.

CURATION (--curate, 1.5.2). A tiny model is data-rich and capacity-poor, so for it
"data quality > volume" (the frontier survey). --curate adds, all deterministic
(stable hashing, no RNG — a given (shards, flags) reproduces byte-for-byte):
  * multi-shard sampling (--shards N spreads sampling ACROSS the crawl, not one
    consecutive run, for diversity),
  * de-duplication (exact full-text + a 120-char normalized prefix, to drop
    near-duplicate templated pages), and
  * prose / register quality filters (letter & digit ratios, terminal punctuation,
    average word length, repetition, long-token spam) — keeping clean English prose.
The DEFAULTS (no --curate, --shards 1) reproduce the 1.5.1 raw behavior byte-for-byte
so the raw slice stays a clean A/B baseline.

Usage:
  # raw 4 MB (1.5.1 behavior, the comparison baseline):
  python3 scripts/c4_sample.py --out data/c4-en-sample.txt --max-bytes 4000000
  # curated 4 MB sampled across 8 shards:
  python3 scripts/c4_sample.py --curate --shards 8 --out data/c4-en-curated.txt --max-bytes 4000000
  # pipe a shard in (e.g. from curl), with --url -:
  curl -sL "<C4 shard url>" | python3 scripts/c4_sample.py --url - --out data/c4-en-sample.txt
"""
import sys
import time
import gzip
import json
import argparse
import hashlib
import urllib.request

SHARD_FMT = ("https://huggingface.co/datasets/allenai/c4/resolve/main/"
             "en/c4-train.%05d-of-01024.json.gz")
N_SHARDS_TOTAL = 1024


def shard_urls(n):
    """n shard URLs spread across the crawl (index i*1024//n) for diversity."""
    if n <= 1:
        return [SHARD_FMT % 0]
    return [SHARD_FMT % ((i * N_SHARDS_TOTAL) // n) for i in range(n)]


def quality_ok(text, min_doc_len):
    """Cheap, deterministic prose/register heuristics (C4-paper-style) — keep clean
    English prose, drop tables / listings / url-hash spam / boilerplate repetition."""
    n = len(text)
    if n < min_doc_len:
        return False
    letters = digits = 0
    for c in text:
        if c.isalpha():
            letters += 1
        elif c.isdigit():
            digits += 1
    if letters < n * 0.6:                       # mostly alphabetic prose
        return False
    if digits > n * 0.12:                        # not a table / numeric listing
        return False
    if (text.count(".") + text.count("!") + text.count("?")) < 3:   # has sentences
        return False
    words = text.split()
    nw = len(words)
    if nw < 20:
        return False
    tot = 0
    longw = 0
    for w in words:
        lw = len(w)
        tot += lw
        if lw > 30:
            longw += 1
    avg = tot / nw
    if avg < 3 or avg > 12:                       # filters url/hash/code spam
        return False
    if longw > nw * 0.02:
        return False
    if len(set(words)) < nw * 0.30:              # not boilerplate repetition
        return False
    return True


def open_src(url):
    if url == "-":
        return sys.stdin.buffer, False
    req = urllib.request.Request(url, headers={"User-Agent": "attn11-c4-sample"})
    return urllib.request.urlopen(req), True     # follows the HF -> CDN redirect


def iter_lines(gz, oversized, max_line=8 * 1024 * 1024, chunk_size=1 << 20):
    """Yield newline-delimited decompressed lines, BOUNDING any single line to
    max_line bytes. The C4 shards are untrusted external data; a plain
    `for line in gz` materializes a whole line before yielding, so a malformed or
    MITM'd gzip member with a giant no-newline run (highly compressible, ~1000:1)
    would buffer hundreds of MB into one object — an OOM / gzip-bomb (1.5.5 audit).
    We instead read fixed `chunk_size` blocks and split locally; a runaway
    no-newline line is dropped (oversized[0] += 1) and resynced at the next
    newline, capping memory at ~chunk_size + max_line. Real C4 docs are a few KB,
    so a >8 MB single line is never a legitimate record. For well-formed shards the
    yielded text (and thus every keep/dedup decision) is byte-identical to the old
    iteration, so sampled corpora reproduce bit-for-bit."""
    buf = b""
    dropping = False
    while True:
        chunk = gz.read(chunk_size)
        if not chunk:
            break
        buf += chunk
        if b"\n" in buf:
            parts = buf.split(b"\n")
            buf = parts.pop()            # last element is the incomplete tail
            for ln in parts:
                if dropping:             # tail end of a dropped runaway line
                    dropping = False     # resync: skip it, resume normal yielding
                    continue
                yield ln
        if len(buf) > max_line:          # runaway line with no newline -> drop it
            if not dropping:
                oversized[0] += 1
            buf = b""
            dropping = True
    if buf and not dropping and len(buf) <= max_line:
        yield buf


def main():
    ap = argparse.ArgumentParser(description="Sample (optionally curate) English text from C4 (en).")
    ap.add_argument("--out", default="data/c4-en-sample.txt")
    ap.add_argument("--max-bytes", type=int, default=4000000)
    ap.add_argument("--min-doc-len", type=int, default=200)
    ap.add_argument("--curate", action="store_true", help="de-dup + prose-quality filters (1.5.2)")
    ap.add_argument("--shards", type=int, default=1, help="number of C4 shards to sample across (diversity)")
    ap.add_argument("--url", default=None, help="single shard URL, or '-' for stdin (overrides --shards)")
    args = ap.parse_args()

    urls = [args.url] if args.url is not None else shard_urls(args.shards)
    nshards = len(urls)
    retries = 3

    written = scanned = kept = dup = lowq = 0
    failed = 0
    ov = [0]                       # oversized (no-newline runaway) lines dropped
    seen_full = set()
    seen_pre = set()
    out = open(args.out, "wb")
    try:
        for ui, url in enumerate(urls):
            if written >= args.max_bytes:
                break
            # Dynamic budget: split the REMAINING bytes over the shards still to come,
            # so a skipped/short shard is made up by the rest (the last one mops up).
            shards_left = nshards - ui
            budget = (args.max_bytes - written) // shards_left
            if budget < 1:
                budget = args.max_bytes - written
            # Open the shard with retries — the HF CDN can 408/throttle a burst of
            # connections; a transient failure must not kill the whole sample.
            opened = None
            attempt = 0
            while attempt < retries:
                try:
                    opened = open_src(url)
                    break
                except Exception as e:
                    attempt += 1
                    sys.stderr.write("c4_sample: shard open failed (%s) %d/%d\n" % (e, attempt, retries))
                    if attempt < retries and url != "-":
                        time.sleep(2.0 * attempt)
            if opened is None:
                failed += 1
                continue
            src, close = opened
            shard_written = 0
            try:
                gz = gzip.GzipFile(fileobj=src)
                for line in iter_lines(gz, ov):
                    if shard_written >= budget or written >= args.max_bytes:
                        break
                    try:
                        doc = json.loads(line)
                    except Exception:
                        continue
                    text = doc.get("text", "")
                    if not isinstance(text, str):
                        continue
                    scanned += 1
                    if args.curate:
                        if not quality_ok(text, args.min_doc_len):
                            lowq += 1
                            continue
                        enc = text.encode("utf-8", "replace")
                        fh = hashlib.blake2b(enc, digest_size=8).digest()
                        ph = hashlib.blake2b(text[:120].lower().encode("utf-8", "replace"), digest_size=8).digest()
                        if fh in seen_full or ph in seen_pre:
                            dup += 1
                            continue
                        seen_full.add(fh)
                        seen_pre.add(ph)
                    elif len(text) < args.min_doc_len:
                        continue
                    chunk = (text.strip() + "\n\n").encode("utf-8")
                    out.write(chunk)
                    written += len(chunk)
                    shard_written += len(chunk)
                    kept += 1
            except Exception as e:
                # a mid-stream network error: keep what this shard gave, move on
                sys.stderr.write("c4_sample: shard stream error (%s); kept %d bytes, continuing\n"
                                 % (e, shard_written))
            finally:
                if close:
                    try:
                        src.close()
                    except Exception:
                        pass
    finally:
        out.close()
    sys.stderr.write(
        "c4_sample: wrote %d bytes -> %s (%s; shards=%d failed=%d scanned=%d kept=%d dup=%d lowq=%d oversized=%d)\n"
        % (written, args.out, "curated" if args.curate else "raw",
           nshards, failed, scanned, kept, dup, lowq, ov[0]))


if __name__ == "__main__":
    main()
