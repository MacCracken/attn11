#!/bin/sh
# Run the attn11 benchmark harness, append a commit-stamped row to
# bench-history.csv, and print a Markdown table of the recent trail.
#
# Per first-party-standards § Benchmarking: dual output (machine-readable CSV +
# human-readable Markdown), normalized to nanoseconds, stamped with date /
# commit / branch.

set -e
cd "$(dirname "$0")/.."

CSV=bench-history.csv

CYRIUS_DCE=1 cyrius build tests/attn11.bcyr build/bench >/dev/null 2>&1
OUT=$(./build/bench)

# pull "<label> ... : <N> ns/op" -> N (first match)
metric() { echo "$OUT" | grep "$1" | grep -oE '[0-9]+ ns/op' | grep -oE '[0-9]+' | head -1; }
linear=$(metric 'linear_fwd')
fwd=$(metric 'model_forward')
bwd=$(metric 'model_backward')
step=$(metric 'fwd+bwd step')
adam=$(metric 'adam_step')
toks=$(echo "$OUT" | grep 'tokens/sec' | grep -oE '[0-9]+$' | head -1)

date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
commit=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit)

if [ ! -f "$CSV" ]; then
    echo "date,commit,branch,tokens_per_sec,step_ns,forward_ns,backward_ns,linear_fwd_ns,adam_ns" > "$CSV"
fi
echo "$date,$commit,$branch,$toks,$step,$fwd,$bwd,$linear,$adam" >> "$CSV"

echo ""
echo "| date | commit | tok/s | step ns | fwd ns | bwd ns | linear ns |"
echo "|------|--------|-------|---------|--------|--------|-----------|"
tail -n +2 "$CSV" | tail -6 | awk -F, \
  '{printf "| %s | %s | %s | %s | %s | %s | %s |\n", $1, $2, $4, $5, $6, $7, $8}'
