# Contributing to attn11

## Development

1. Install the Cyrius toolchain at the version pinned in `cyrius.cyml`
   (`[package].cyrius`). The pin is the single source of truth — never hardcode
   a version elsewhere.
2. `cyrius deps` — resolve stdlib deps into `lib/`.
3. `cyrius build src/main.cyr build/attn11` — compile (train + sample).
4. `cyrius test` — run the gradient-check suite (`tests/attn11.tcyr`) plus the
   `[build].test` entry.
5. `cyrius build tests/attn11.fcyr build/fuzz && ./build/fuzz` — fuzz harness.
6. `cyrius build tests/attn11.bcyr build/bench && ./build/bench` — benchmarks.

See [`CLAUDE.md`](CLAUDE.md) for the full development loop and
[`docs/architecture/tensors-and-floats.md`](docs/architecture/tensors-and-floats.md)
for the numeric conventions.

## Gradient checks are the contract

Every backward function MUST be verified against central finite differences in
`tests/attn11.tcyr` before it lands. Hand-derived backprop is the part most
likely to be subtly wrong; a new op without a passing grad check is incomplete.
The gate is max relative error `< 1e-5` per op (`< 1e-4` for the full model).

## Numeric rules

- Cyrius has no float type — an `f64` is its bit pattern in an i64. Use the
  `f64_*` builtins, never `+`/`*` on float values.
- Build precise constants from integer ratios or runtime math; long-digit float
  literals mis-parse.
- Forward writes; backward accumulates into parameter grads and writes input
  grads. No allocation inside the training loop.

## Process

- One change at a time. Never bundle unrelated changes in a single PR.
- Test after every change; grad-check after every backward-touching change;
  benchmark after every perf-touching change.
- Performance claims must include numbers — `before → after` with the bench name.
- Breaking changes get a `Breaking` section in [`CHANGELOG.md`](CHANGELOG.md)
  with a migration paragraph.
- Do not commit/push or use `gh` — the maintainer handles git operations.

## License

GPL-3.0-only.
