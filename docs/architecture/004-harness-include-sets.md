# 004 — Harness include sets: a symbol used by `model.cyr` must not live only in `persist.cyr`

**Status**: Active · **Affects**: `src/model.cyr` (and any `src/` file the bench
includes), `tests/attn11.bcyr`; any cross-file function reference in shared code

## The rule

The four build entry points do **not** all include the same `src/` files:

| entry point          | tensor | fileio | ops | attn | model | train | persist |
|----------------------|:------:|:------:|:---:|:----:|:-----:|:-----:|:-------:|
| `src/main.cyr`       |   ✓    |   ✓    |  ✓  |  ✓   |   ✓   |   ✓   |    ✓    |
| `tests/attn11.tcyr`  |   ✓    |   ✓    |  ✓  |  ✓   |   ✓   |   ✓   |    ✓    |
| `tests/attn11.fcyr`  |   ✓    |   ✓    |  ✓  |  ✓   |   ✓   |   ✓   |    ✓    |
| `tests/attn11.bcyr`  |   ✓    |   ✓    |  ✓  |  ✓   |   ✓   |   ✓   |  **✗**  |

The **benchmark harness (`bcyr`) is the only one that omits `persist.cyr`** — it
times training/generation and never touches checkpoints. So a function that
`model.cyr` (or anything in the bench's include set) *references* must be
*defined* in a file the bench includes — i.e. **not** `persist.cyr`. The
model-allocation cap `CKPT_MAX_MODEL_BYTES()` lives in `model.cyr` for exactly
this reason (it gates both `model_init`'s fresh-path pre-flight and the
checkpoint loader's `-18`); the BPE constants `model_config_ok` reads
(`BPE_VMAX()`) live in `train.cyr`, which the bench *does* include.

## Why it bites silently

cyrius treats an **undefined function as a warning, not an error**, and emits a
**trap** at the call site:

```
warning: undefined function 'CKPT_MAX_MODEL_BYTES'
compile tests/attn11.bcyr -> build/bench [x86_64] OK     # <- build "succeeds"
./build/bench  ->  Illegal instruction (core dumped)     # <- SIGILL at runtime
```

The build prints `OK` and produces a binary, so a green compile proves nothing.
The trap only fires when control reaches the call (here, the first
`model_init`), which a `cyrius build` without a run never exercises. This is how
a `model.cyr → persist.cyr` reference passed every gate that *runs* a
persist-including harness (test/fuzz) and the main binary, yet crashed the bench
in CI (exit 132).

## The discipline

After changing any function reference in a **shared** `src/` file (`model.cyr`
most of all), **rebuild _and run_ all four harnesses** — `main`, `tcyr`,
`fcyr`, **and `bcyr`**. A passing `cyrius build` (or even a passing `cyrius
test`) is not enough: the bench has a smaller include set, and undefined
cross-file references surface only when the harness with the gap actually
*executes* the offending call.
