#!/bin/sh
# attn11 — X021: the data-volume held-out generalization experiment (the X019/X020
# follow-on, rides the shipped --eval-corpus + c4_sample.py; no new binary surface).
#
# Question: does training on MORE clean data generalize better to a THIRD disjoint
# held-out set? X020 found own≈held-out (gap <1.3%) because sub-epoch training barely
# memorizes — so the data-volume win can only show once the model OVERFITS the small
# corpus. This run forces that regime: a SMALL train slice (many epochs → overfit) vs
# a LARGER disjoint slice (matched compute), both scored on a third disjoint slice.
#
# Three disjoint, same-distribution slices of one curated C4 pool (byte-offset gaps
# guarantee disjoint documents, as in X020):
#   trainS  = [0, SMALL)                         (overfit target)
#   trainL  = [SMALL+GAP, SMALL+GAP+LARGE)       (more, diverse data; disjoint)
#   heldB   = [POOL-HELD, POOL)                  (the held-out set; disjoint from both)
# Matched compute: the SAME step budget for every cell. Metric: bits/byte (byte-
# normalized, so comparable across the two BPE tokenizers — X003) on own vs heldB.
# Deterministic (curation + seed 1337); RNG-neutral evals; cross-arch reproducible.
#
# Usage:  scripts/x021-heldout.sh [steps]      (default 4000)
set -e
cd "$(dirname "$0")/.."

POOL_BYTES="${POOL_BYTES:-12000000}"      # curated C4 pool to slice from
SMALL="${SMALL:-262144}"                  # 256 KB  (overfittable)
LARGE="${LARGE:-4194304}"                 # 4 MB    (more, diverse data)
HELD="${HELD:-524288}"                    # 512 KB  (held-out)
GAP="${GAP:-1048576}"                     # 1 MB gap between trainS and trainL
STEPS="${STEPS:-${1:-4000}}"
BPE="${BPE:-256}"
DATA=data/x021
POOL="$DATA/c4-pool.txt"

mkdir -p "$DATA"
CYRIUS_DCE=1 cyrius build src/main.cyr build/attn11 >/dev/null 2>&1

# 1) curated C4 pool (deterministic; skip if already downloaded)
if [ ! -s "$POOL" ]; then
  echo "x021: downloading + curating a ${POOL_BYTES}-byte C4 pool (one-time)..." >&2
  python3 scripts/c4_sample.py --curate --shards 12 --max-bytes "$POOL_BYTES" --out "$POOL"
fi
actual=$(wc -c < "$POOL")
echo "x021: pool = $actual bytes" >&2
if [ "$actual" -lt $((SMALL + GAP + LARGE)) ] || [ "$actual" -lt $((HELD + 1)) ]; then
  echo "x021: pool too small for the requested slices; raise POOL_BYTES" >&2; exit 1
fi

# 2) disjoint slices (tail -c +K skips K-1 bytes; head -c N takes N)
slice () { # off len out
  tail -c +"$(( $1 + 1 ))" "$POOL" | head -c "$2" > "$3"
}
slice 0 "$SMALL" "$DATA/trainS.txt"
slice $((SMALL + GAP)) "$LARGE" "$DATA/trainL.txt"
slice $((actual - HELD)) "$HELD" "$DATA/heldB.txt"

bpb () { printf '%s\n' "$1" | grep -aoE 'bits/byte [0-9.]+' | sed -n "$2p" | awk '{print $2}'; }

run_cell () { # label preset_flag corpus
  out=$(./build/attn11 $2 --corpus "$3" --bpe "$BPE" --steps "$STEPS" \
        --eval --eval-corpus "$DATA/heldB.txt" 2>/dev/null)
  own=$(bpb "$out" 1)        # first bits/byte = --eval (own corpus)
  held=$(bpb "$out" 2)       # second bits/byte = --eval-corpus (heldB)
  params=$(printf '%s\n' "$out" | grep -aoE 'params=[0-9]+' | head -1 | cut -d= -f2)
  gap=$(awk -v o="$own" -v h="$held" 'BEGIN{ if(o>0) printf "%.1f%%", 100*(h-o)/o; else print "-" }')
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$params" "$own" "$held" "$gap"
}

echo
echo "X021 — data-volume held-out generalization (BPE ${BPE}, ${STEPS} steps, seed 1337)"
echo "  trainS=${SMALL}B  trainL=${LARGE}B  heldB=${HELD}B  (disjoint slices of a ${actual}B C4 pool)"
echo
printf 'cell\tparams\town b/b\theld b/b\town→held gap\n'
run_cell "default trainS" ""        "$DATA/trainS.txt"
run_cell "default trainL" ""        "$DATA/trainL.txt"
run_cell "preset  trainS" "--preset" "$DATA/trainS.txt"
run_cell "preset  trainL" "--preset" "$DATA/trainL.txt"
echo
echo "Headline: compare 'held b/b' of trainL vs trainS per model size."
echo "  trainL < trainS on held-out  => more clean data generalizes better (the X019 win)."
echo "  a large own→held gap on trainS confirms the overfit regime X020 lacked."
