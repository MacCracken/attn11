#!/bin/sh
# attn11 — MoE expert-density sweep (M13, E8). For each expert count N, train the
# default config (C=32, ctx 16, 3 layers) for a fixed budget on the embedded
# reference corpus, then eval. Reports, per N: total params, per-token-active
# params, bits/byte (pure cross-entropy), and routing entropy (1 = balanced
# expert load, 0 = collapse). N=1 is the dense baseline. Deterministic (fixed
# seed) and reproducible. A full vidya-corpus bake-off is the follow-on X-entry.
#
# Usage:  scripts/moe-sweep.sh [steps]   (default 1200)
set -e
cd "$(dirname "$0")/.."
CYRIUS_DCE=1 cyrius build src/main.cyr build/attn11 >/dev/null 2>&1
STEPS="${1:-1200}"
echo "MoE density sweep — default config, ${STEPS} steps, embedded corpus, top-2"
echo "N	params	active/token	bits/byte	route-entropy"
for N in 1 4 8 16 32 64; do
  if [ "$N" = "1" ]; then
    out=$(./build/attn11 --steps "$STEPS" --eval --gen-only 2>/dev/null) || out=""
    out=$(./build/attn11 --steps "$STEPS" --eval 2>/dev/null)
  else
    out=$(./build/attn11 --experts "$N" --expert-topk 2 --steps "$STEPS" --eval 2>/dev/null)
  fi
  params=$(printf '%s\n' "$out" | grep -oE 'params=[0-9]+' | head -1 | cut -d= -f2)
  bpb=$(printf '%s\n' "$out" | grep -oE 'bits/byte [0-9.]+' | head -1 | awk '{print $2}')
  active=$(printf '%s\n' "$out" | grep -oE 'active/token [0-9]+' | head -1 | awk '{print $2}')
  ent=$(printf '%s\n' "$out" | grep -oE 'route-entropy [0-9.]+' | head -1 | awk '{print $2}')
  [ "$N" = "1" ] && active="$params" && ent="-"
  printf '%s\t%s\t%s\t%s\t%s\n' "$N" "$params" "$active" "$bpb" "$ent"
done
