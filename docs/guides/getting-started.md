# Getting started with attn11

attn11 trains a small char-level GPT-style transformer from scratch in Cyrius —
forward pass, hand-derived backprop, and Adam, all on raw `f64` arrays.

## Build & run

```sh
cyrius deps                                # resolve stdlib deps
cyrius build src/main.cyr build/attn11     # compile (static ELF)
./build/attn11                             # train on the embedded corpus, then sample
cyrius test                                # gradient-check suite + smoke tests
```

## CLI

```
attn11                       train on the embedded corpus, then sample
attn11 --corpus PATH         train on a corpus file (O_NOFOLLOW, 4 MB cap)
attn11 --stdin               train on a corpus read from stdin
attn11 --steps N             total training steps (also the LR-schedule horizon)
attn11 --save PATH           write a crash-atomic checkpoint after training
attn11 --load PATH           resume from a checkpoint (+ --corpus to continue training)
attn11 --gen-only            skip training, just sample
```

Example — train on your own text, checkpoint, then resume:

```sh
attn11 --corpus mytext.txt --steps 2000 --save run.ckpt
attn11 --load run.ckpt --corpus mytext.txt --steps 4000   # resume to 4000
attn11 --load run.ckpt --gen-only                         # sample from it
```

## Layout

- `src/main.cyr` — CLI + orchestration; `src/{tensor,fileio,ops,attn,model,train,persist}.cyr` — the model.
- `tests/attn11.tcyr` — gradient-check + persistence + robustness suite (`cyrius test`).
- `tests/attn11.fcyr` — fuzz harness for the loaders.
- `tests/attn11.bcyr` — benchmarks (`./scripts/bench-history.sh`).
- A minimal embedding-of-the-API example: [`../examples/minimal_train.cyr`](../examples/minimal_train.cyr).

## aarch64

The model cross-builds and runs under qemu (grad checks pass on both arches):

```sh
cyrius build --aarch64 tests/attn11.tcyr build/test_a64 && qemu-aarch64 build/test_a64
```

## Adding a feature

1. Edit/add a module under `src/` and `include` it from `src/main.cyr`.
2. If it adds a backward op, **grad-check it** in `tests/attn11.tcyr` (finite
   differences) — a backward without a passing grad check is incomplete.
3. `cyrius test`; benchmark with `./scripts/bench-history.sh` if it touches a hot path.
4. Bump `VERSION`, update `CHANGELOG.md` and `docs/development/state.md`; add an
   ADR ([`../adr/template.md`](../adr/template.md)) for any non-trivial choice.

See [`../sources.md`](../sources.md) for the citation map and
[`../architecture/001-tensors-and-floats.md`](../architecture/001-tensors-and-floats.md)
for the f64/SIMD conventions.
