#!/bin/sh
# attn11 — X024: REINFORCE vs SFT. The M17 gate: does the policy move toward the
# reward? Trains the SAME default model two ways on one corpus — supervised (AR
# next-token CE) and REINFORCE (--objective rl, reward = count of a target char per
# rollout) — and compares, per target char: the target's frequency in the policy's
# samples (measured the same way for both, via the in-binary RL rollout eval) and
# the corpus bits/byte (LM quality). Rides only shipped flags; deterministic
# (seed 1337); RNG-neutral evals. The SFT baseline's target frequency is measured
# by loading the AR checkpoint and running the RL rollout eval on it (--gen-only).
#
# Usage:  scripts/m17-rl.sh [steps]      (default 400)
set -e
cd "$(dirname "$0")/.."
CYRIUS_DCE=1 cyrius build src/main.cyr build/attn11 >/dev/null 2>&1
STEPS="${1:-400}"
DATA=data/x024
CORP="$DATA/corpus.txt"
mkdir -p "$DATA"
if [ ! -s "$CORP" ]; then
  printf 'the quick brown fox jumps over the lazy dog. she sells sea shells by the sea shore. a transformer learns from text. ' > "$DATA/base.txt"
  : > "$CORP"
  i=0; while [ $i -lt 200 ]; do cat "$DATA/base.txt" >> "$CORP"; i=$((i+1)); done
  rm -f "$DATA/base.txt"
fi

freq () { printf '%s\n' "$1" | grep -aoE 'freq [0-9.]+%' | head -1 | grep -aoE '[0-9.]+' | head -1; }
bpb  () { printf '%s\n' "$1" | grep -aoE 'bits/byte [0-9.]+' | head -1 | awk '{print $2}'; }

# SFT/AR baseline (shared across targets): train once, record corpus bits/byte.
ar=$(./build/attn11 --corpus "$CORP" --steps "$STEPS" --save "$DATA/ar.ckpt" --eval 2>/dev/null)
ar_bpb=$(bpb "$ar")

echo
echo "X024 — REINFORCE vs SFT (default config, byte-level, ${STEPS} steps, seed 1337)"
echo "SFT/AR baseline corpus bits/byte: ${ar_bpb}"
echo
printf 'target\tSFT freq\tRL freq\tSFT b/b\tRL b/b\n'
for t in 'e' ' ' 'z'; do
  # SFT target frequency: the AR policy's rollout target-count, measured the RL way
  sftm=$(./build/attn11 --load "$DATA/ar.ckpt" --corpus "$CORP" --objective rl --rl-target "$t" --gen-only 2>/dev/null)
  sftf=$(freq "$sftm")
  # RL: train toward the target, then its frequency + corpus bits/byte
  rlm=$(./build/attn11 --corpus "$CORP" --objective rl --rl-target "$t" --steps "$STEPS" --eval 2>/dev/null)
  rlf=$(freq "$rlm"); rlb=$(bpb "$rlm")
  printf "'%s'\t%s%%\t%s%%\t%s\t%s\n" "$t" "$sftf" "$rlf" "$ar_bpb" "$rlb"
done
echo
echo "Gate: RL freq >> SFT freq  => the policy moves toward the reward (M17 gate met)."
echo "      RL b/b  >> SFT b/b   => the SFT->RL alignment tax (reward hacking; naive reward)."
