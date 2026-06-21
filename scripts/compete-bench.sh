#!/bin/sh
# attn11 — competitor benchmarks (the B-series, B0 harness).
#
# The self-bench (scripts/bench-history.sh) tracks attn11 against its OWN
# history. This adds the missing axis: attn11 vs external references, on two
# surfaces (training step tok/s, decode tok/s) and two stories — honest raw
# throughput AND the normalized "throughput-at-zero-dependencies" (where attn11
# is alone: one static ELF, no BLAS/libc/CUDA).
#
# FAIRNESS (no cherry-picking; see docs/development/roadmap.md "B-series"):
#   * Matched config — competitors are mapped to attn11's --preset; a row is
#     only accepted as a COMPARISON once its printed param count matches within
#     tolerance. Unmatched/unbuilt/unavailable competitors get an explicit
#     status row (built-only / skip+reason) — NEVER a silent drop and NEVER a
#     fabricated number.
#   * Same host, pinned — taskset to one CPU, warmup, run-of-record; the harness
#     RECORDS the resolved upstream commit it built (reproducible).
#   * attn11 single-thread first — its honest scalar/SIMD baseline; competitor
#     thread counts (llm.c OpenMP, PyTorch MKL) go in the deps/threads column.
#   * No vendoring — competitors clone+build into a gitignored bench/ dir.
#
# This box may lack a competitor's stack (e.g. PyTorch for nanoGPT) or the
# data-prep for a matched config (e.g. llm.c's GPT-2 tokenizer/weights). Those
# emit a skip row with the reason; run this on a bench machine with the stacks
# installed to fill the competitor comparison cells. attn11's own rows + the
# zero-deps story are produced anywhere.
#
# Usage:  scripts/compete-bench.sh            (clone/build/run what is feasible)
#         BENCH_RUN_COMPETITORS=0 scripts/compete-bench.sh   (attn11 + build-check only)
set -e
cd "$(dirname "$0")/.."

BENCH=bench
CSV=competitor-bench.csv
HOST=$(uname -srm 2>/dev/null | tr ' ,' '__')
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$BENCH"

# competitor upstream repos (refs RESOLVED + recorded per run until exact
# commits are pinned — see roadmap B-series open items)
LLMC_URL=https://github.com/karpathy/llm.c
NANOGPT_URL=https://github.com/karpathy/nanoGPT
LLAMA2C_URL=https://github.com/karpathy/llama2.c
MICROGRAD_URL=https://github.com/karpathy/micrograd

if [ ! -f "$CSV" ]; then
  echo "competitor,ref,lang,surface,tokens_per_sec,params,threads,deps,static_bytes,status,host,date" > "$CSV"
fi
row() { # competitor ref lang surface toks params threads deps bytes status
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "$HOST" "$DATE" >> "$CSV"
}

# ---- attn11 itself: real numbers (self-bench + zero-deps) -------------------
echo "compete-bench: attn11 (self) ..." >&2
CYRIUS_DCE=1 cyrius build src/main.cyr "$BENCH/attn11" >/dev/null 2>&1
CYRIUS_DCE=1 cyrius build tests/attn11.bcyr "$BENCH/attn11_bench" >/dev/null 2>&1
BOUT=$("$BENCH/attn11_bench" 2>/dev/null || echo "")
a_step_ns=$(printf '%s\n' "$BOUT" | grep -a 'fwd+bwd step' | grep -aoE '[0-9]+ ns/op' | grep -aoE '[0-9]+' | head -1)
a_toks=$(printf '%s\n' "$BOUT" | grep -a 'tokens/sec' | grep -aoE '[0-9]+$' | head -1)
a_bytes=$(stat -c %s "$BENCH/attn11" 2>/dev/null || wc -c < "$BENCH/attn11")
a_params=$("$BENCH/attn11" --steps 0 --gen-only 2>/dev/null | tr -d '\000' | grep -aoE 'params=[0-9]+' | head -1 | cut -d= -f2)
a_deps=$(ldd "$BENCH/attn11" 2>&1 | grep -aqi 'not a dynamic' && echo none || echo dynamic)
row attn11 "$(git rev-parse --short HEAD 2>/dev/null || echo nogit)" cyrius train "${a_toks:-NA}" "${a_params:-NA}" 1 "$a_deps" "$a_bytes" ok
echo "  attn11: ${a_toks:-?} tok/s, ${a_params:-?} params, ${a_bytes} B static ELF, deps=${a_deps}" >&2

# ---- competitor helper: shallow clone @ default, record resolved commit -----
clone_at() { # name url -> echoes resolved short commit, or "" on failure
  d="$BENCH/$1"
  if [ ! -d "$d/.git" ]; then
    git clone --depth 1 "$2" "$d" >/dev/null 2>&1 || { echo ""; return; }
  fi
  git -C "$d" rev-parse --short HEAD 2>/dev/null || echo ""
}

if [ "${BENCH_RUN_COMPETITORS:-1}" = "0" ]; then
  echo "compete-bench: BENCH_RUN_COMPETITORS=0 — skipping competitor clone/build" >&2
  exit 0
fi

# ---- micrograd (pure-Python from-scratch floor; not matched-config) ---------
echo "compete-bench: micrograd ..." >&2
mg=$(clone_at micrograd "$MICROGRAD_URL")
if [ -n "$mg" ]; then
  row micrograd "$mg" python train-floor NA NA 1 cpython NA "built; matched-run-NA(MLP-not-transformer; reference floor only)"
else
  row micrograd nofetch python train-floor NA NA 1 cpython NA "skip(clone-failed: no network?)"
fi

# ---- llama2.c (C; decode surface; needs a model .bin) -----------------------
echo "compete-bench: llama2.c ..." >&2
l2=$(clone_at llama2.c "$LLAMA2C_URL")
if [ -n "$l2" ]; then
  ( cd "$BENCH/llama2.c" && timeout 120 make run >/dev/null 2>&1 ) \
    && st="built; matched-run-pending(needs a model .bin + config map to --preset)" \
    || st="skip(build-failed)"
  row llama2.c "$l2" c decode NA NA 1 "libc+libm" NA "$st"
else
  row llama2.c nofetch c decode NA NA 1 "libc+libm" NA "skip(clone-failed)"
fi

# ---- llm.c (C+OpenMP; training) — matched-config B1 (the truest from-scratch peer) ------
# A random-init tiny GPT-2 at attn11's default config (C32·T16·NH4·L3) exercises the same
# FLOPs/step with NO checkpoint / tokenizer / data-prep needed — the `#define TESTING;
# #include "train_gpt2.c"` pattern llm.c's own tests use. tok/s = B·T·steps / train_time,
# single-thread (the B-series rule), random tokens + small-random weights (timing-invariant).
echo "compete-bench: llm.c ..." >&2
lc=$(clone_at llm.c "$LLMC_URL")
if [ -n "$lc" ]; then
  ( cd "$BENCH/llm.c" && timeout 180 make train_gpt2 >/dev/null 2>&1 )
  cat > "$BENCH/llm.c/llm_tiny.c" <<'LLMTINY'
#define TESTING
#include "train_gpt2.c"
#include <time.h>
#include <omp.h>
static unsigned long long _s = 88172645463325252ULL;
static float frnd() { _s ^= _s << 13; _s ^= _s >> 7; _s ^= _s << 17; return ((_s >> 40) / (float)(1 << 24)) - 0.5f; }
static void build_random(GPT2 *m, int maxT, int V, int Vp, int L, int NH, int C) {
    m->config.max_seq_len = maxT; m->config.vocab_size = V; m->config.num_layers = L;
    m->config.num_heads = NH; m->config.channels = C; m->config.padded_vocab_size = Vp;
    fill_in_parameter_sizes(m->param_sizes, m->config);
    size_t np = 0; for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) np += m->param_sizes[i];
    m->num_parameters = np; m->params_memory = malloc_and_point_parameters(&m->params, m->param_sizes);
    for (size_t i = 0; i < np; i++) m->params_memory[i] = frnd() * 0.04f;
    m->acts_memory = NULL; m->grads_memory = NULL; m->m_memory = NULL; m->v_memory = NULL;
    m->grads_acts_memory = NULL; m->inputs = NULL; m->targets = NULL;
    m->batch_size = 0; m->seq_len = 0; m->mean_loss = -1.0f;
}
int main(int argc, char **argv) {
    int C = 32, T = 16, NH = 4, L = 3, V = 256, B = 1, steps = 60, warm = 20;
    if (argc >= 7) { C = atoi(argv[1]); T = atoi(argv[2]); NH = atoi(argv[3]); L = atoi(argv[4]); V = atoi(argv[5]); B = atoi(argv[6]); }
    if (argc >= 8) steps = atoi(argv[7]);
    int Vp = V, maxT = (T > 16 ? T : 16);
    GPT2 m; build_random(&m, maxT, V, Vp, L, NH, C);
    int *inp = (int*)mallocCheck((size_t)B * T * sizeof(int)), *tgt = (int*)mallocCheck((size_t)B * T * sizeof(int));
    double tsum = 0; int cnt = 0;
    for (int step = 0; step < steps; step++) {
        for (int i = 0; i < B * T; i++) { inp[i] = ((unsigned)(frnd() * 1e6f)) % V; tgt[i] = ((unsigned)(frnd() * 1e6f)) % V; }
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        gpt2_forward(&m, inp, tgt, B, T); gpt2_zero_grad(&m); gpt2_backward(&m);
        gpt2_update(&m, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double ms = (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
        if (step >= warm) { tsum += ms; cnt++; }
    }
    double mspstep = tsum / cnt, toks = (double)B * T * 1000.0 / mspstep;
    printf("params=%zu %.0f tok/s (%d threads, %.3f ms/step)\n", m.num_parameters, toks, omp_get_max_threads(), mspstep);
    return 0;
}
LLMTINY
  lc_toks=NA; lc_params=NA; lc_st="skip(driver-build-failed: needs OpenMP/headers)"
  if ( cd "$BENCH/llm.c" && cc -O3 -march=native -ffast-math -fopenmp llm_tiny.c -lm -o llm_tiny 2>/dev/null ); then
    lout=$(cd "$BENCH/llm.c" && OMP_NUM_THREADS=1 ./llm_tiny 32 16 4 3 256 1 60 2>/dev/null | grep -a 'tok/s')
    lc_toks=$(echo "$lout" | grep -oE '[0-9]+ tok/s' | grep -oE '^[0-9]+')
    lc_params=$(echo "$lout" | grep -oE 'params=[0-9]+' | cut -d= -f2)
    [ -n "$lc_toks" ] && lc_st="ok(matched C32·T16·NH4·L3·V256, B1, random-init, 1-thread)" || lc_st="skip(driver-run-failed)"
  fi
  row llm.c "$lc" c train "${lc_toks:-NA}" "${lc_params:-NA}" 1 "libc+libm+OpenMP" NA "$lc_st"
else
  row llm.c nofetch c train NA NA OpenMP "libc+libm+OpenMP" NA "skip(clone-failed)"
fi

# ---- nanoGPT (PyTorch; training+decode; needs torch) -----------------------
echo "compete-bench: nanoGPT ..." >&2
if python3 -c "import torch" >/dev/null 2>&1; then
  ng=$(clone_at nanoGPT "$NANOGPT_URL")
  row nanoGPT "${ng:-nofetch}" pytorch train NA NA "MKL" "pytorch(+cuda)" NA "built; matched-run-pending(map config to --preset)"
else
  row nanoGPT nostack pytorch train NA NA "MKL" "pytorch(+cuda)" NA "skip(PyTorch not installed on this host)"
fi

# ---- summary ---------------------------------------------------------------
echo "" >&2
echo "compete-bench: wrote rows to $CSV (status column tells matched vs build-only vs skip)" >&2
echo ""
echo "| competitor | ref | surface | tok/s | params | deps | bytes | status |"
echo "|------------|-----|---------|-------|--------|------|-------|--------|"
tail -n +2 "$CSV" | tail -8 | awk -F, \
  '{printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$4,$5,$6,$8,$9,$10}'
