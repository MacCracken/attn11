# Architecture notes

Non-obvious constraints, quirks, and invariants that a reader cannot derive from the code alone. Numbered chronologically — never renumber.

Not decisions (those live in [`../adr/`](../adr/)) and not guides (those live in [`../guides/`](../guides/)). An item here describes *how the world is*, not *what we chose* or *how to do something*.

## Items

- [001 — Tensors and floats](001-tensors-and-floats.md) — everything is i64; an
  `f64` is its bit pattern; tensors are flat row-major f64 arrays; the SIMD
  matmul convention (memory accumulators, never reassign a SIMD var); and the
  toolchain gotchas (long-literal mis-parse, byte-counting `print`, f64
  comparisons not being NaN-correct). **Affects**: all numeric code in `src/`.
- [002 — agnos entry epilogue](002-agnos-entry-epilogue.md) — `r = main()`
  must be a top-level *statement*, never a `var` initializer: on agnos, a
  call-bearing gvar init runs before the compiler-emitted argv capture, so
  `main()` would see `argc()==0` and silently ignore every CLI flag.
  **Affects**: every entry file (`src/main.cyr`, `src/test.cyr`,
  `tests/attn11.{tcyr,bcyr,fcyr}`).
