# Architecture notes

Non-obvious constraints, quirks, and invariants that a reader cannot derive from the code alone. Numbered chronologically — never renumber.

Not decisions (those live in [`../adr/`](../adr/)) and not guides (those live in [`../guides/`](../guides/)). An item here describes *how the world is*, not *what we chose* or *how to do something*.

## Items

- [001 — Tensors and floats](001-tensors-and-floats.md) — everything is i64; an
  `f64` is its bit pattern; tensors are flat row-major f64 arrays; the SIMD
  matmul convention (memory accumulators, never reassign a SIMD var); and the
  toolchain gotchas (long-literal mis-parse, byte-counting `print`, f64
  comparisons not being NaN-correct). **Affects**: all numeric code in `src/`.
- [002 — agnos entry epilogue](002-agnos-entry-epilogue.md) — **retired at
  pin ≥ 6.1.32** (upstream fix: r15-parked init rsp). Was: `r = main()`
  must be a top-level *statement*, never a `var` initializer: on agnos, a
  call-bearing gvar init runs before the compiler-emitted argv capture, so
  `main()` would see `argc()==0` and silently ignore every CLI flag.
  Residual rule: the pin and the `lib/` snapshot must move together.
  **Affects**: every entry file (`src/main.cyr`, `src/test.cyr`,
  `tests/attn11.{tcyr,bcyr,fcyr}`, `docs/examples/minimal_train.cyr`).
- [003 — The cached-inference bit contract](003-cached-inference-bit-contract.md) —
  the KV-cached row path (`attn_fwd_row`/`model_fwd_row`) must stay
  arithmetically identical, per row, to the batch path; what that forbids
  when changing kernels. **Affects**: `src/attn.cyr`, `src/model.cyr`,
  `src/ops.cyr` — any forward-kernel change.
- [004 — Harness include sets](004-harness-include-sets.md) — the bench
  (`tests/attn11.bcyr`) is the only entry point that omits `persist.cyr`, and
  cyrius compiles an undefined function to a runtime *trap* (not a build
  error), so a symbol `model.cyr` references must not live only in
  `persist.cyr` — rebuild **and run** all four harnesses after touching shared
  `src/`. **Affects**: `src/model.cyr`, `tests/attn11.bcyr`.
