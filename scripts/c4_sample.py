#!/usr/bin/env python3
"""Stream a small English sample out of the C4 (en) corpus into a plain UTF-8
text file for attn11's `--corpus` path.

NO third-party deps — stdlib `gzip` + `json` only (the box this targets has no pip
/ no tensorflow / no tensorflow_datasets). C4's TFDS `c4/en` config is 305 GB and
needs an Apache-Beam preparation; that is the wrong tool for materializing a few MB
of text for a tiny model. Instead we stream ONE of C4's public gzipped JSON-lines
shards (allenai/c4 is the raw mirror of the exact corpus TFDS catalogs) and STOP
once --max-bytes of text is collected — so the actual download is only ~the
compressed bytes backing the sample (a few MB), not the 319 MB shard, let alone the
full dataset.

Usage:
  # fetch directly (urllib follows the HF redirect):
  python3 scripts/c4_sample.py --out data/c4-en-sample.txt --max-bytes 4000000
  # or pipe a shard in (e.g. from curl), with --url -:
  curl -sL "<C4 shard url>" | python3 scripts/c4_sample.py --url - --out data/c4-en-sample.txt

Each C4 record is a JSON object {"text": ..., "timestamp": ..., "url": ...}; we
keep the `text` field of documents at least --min-doc-len chars, paragraph-separated.
"""
import sys
import gzip
import json
import argparse

DEFAULT_URL = ("https://huggingface.co/datasets/allenai/c4/resolve/main/"
               "en/c4-train.00000-of-01024.json.gz")


def stream(fileobj, out, max_bytes, min_doc_len):
    written = 0
    gz = gzip.GzipFile(fileobj=fileobj)
    for line in gz:
        if written >= max_bytes:
            break
        try:
            doc = json.loads(line)
        except Exception:
            continue
        text = doc.get("text", "")
        if not isinstance(text, str) or len(text) < min_doc_len:
            continue
        chunk = (text.strip() + "\n\n").encode("utf-8")
        out.write(chunk)
        written += len(chunk)
    return written


def main():
    ap = argparse.ArgumentParser(description="Sample English text from C4 (en).")
    ap.add_argument("--out", default="data/c4-en-sample.txt")
    ap.add_argument("--max-bytes", type=int, default=4000000,
                    help="stop after ~this many bytes of text (attn11 caps a corpus at 4 MB)")
    ap.add_argument("--min-doc-len", type=int, default=200,
                    help="skip documents shorter than this many chars")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="C4 shard URL, or '-' to read a gzipped shard from stdin")
    args = ap.parse_args()

    close = False
    if args.url == "-":
        src = sys.stdin.buffer
    else:
        import urllib.request
        req = urllib.request.Request(args.url, headers={"User-Agent": "attn11-c4-sample"})
        src = urllib.request.urlopen(req)   # follows the HF -> CDN redirect
        close = True

    with open(args.out, "wb") as out:
        try:
            n = stream(src, out, args.max_bytes, args.min_doc_len)
        finally:
            if close:
                try:
                    src.close()
                except Exception:
                    pass
    sys.stderr.write("c4_sample: wrote %d bytes of C4-en text -> %s\n" % (n, args.out))


if __name__ == "__main__":
    main()
