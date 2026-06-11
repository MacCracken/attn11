# attn11 — Claude Code Instructions

> **Core rule**: this file is **preferences, process, and procedures** —
> durable rules that change rarely. Volatile state (current version, binary
> size, test counts, in-flight work, consumers) lives in
> [`docs/development/state.md`](docs/development/state.md), refreshed every
> release; historical release narrative lives in
> [`CHANGELOG.md`](CHANGELOG.md). Do not inline state here — it rots within a
> minor.

## Project Identity

**attn11** ("attn" = attention + "11") — a from-scratch, dependency-free
GPT-style transformer (**trained**, not just inference) written in Cyrius.

- **Type**: Binary
- **License**: GPL-3.0-only
- **Language**: Cyrius (toolchain pinned in `cyrius.cyml [package].cyrius`)
- **Version**: `VERSION` at the project root is the source of truth — do not inline the number here
- **Genesis repo**: [agnosticos](https://github.com/MacCracken/agnosticos)
- **Standards**: [First-Party Standards](https://github.com/MacCracken/agnosticos/blob/main/docs/development/first-party/first-party-standards.md) · [First-Party Documentation](https://github.com/MacCracken/agnosticos/blob/main/docs/development/first-party/first-party-documentation.md)
- **Shared crates**: [shared-crates.md](https://github.com/MacCracken/agnosticos/blob/main/docs/development/planning/shared-crates.md)

## Goal

attn11 owns the **end-to-end training** of a small char-level transformer
entirely in Cyrius — forward pass, hand-derived backprop, and the Adam
optimizer, all on raw `f64` arrays (no BLAS, no libc, no autodiff). It is the
ecosystem's reference that real gradient-based learning is expressible in an
"assembly-up", everything-is-i64 systems language. Hand-written gradients are
kept honest by finite-difference gradient checks.

## Current State

> Volatile state lives in [`docs/development/state.md`](docs/development/state.md) —
> current version, binary size, test/assertion counts, in-flight work,
> consumers, performance numbers. Refreshed every release. Historical release
> narrative lives in [`CHANGELOG.md`](CHANGELOG.md).

This file (`CLAUDE.md`) is durable rules.

## Scaffolding

Project was scaffolded with `cyrius init` (greenfield) or `cyrius port` (Rust →
Cyrius migration). **Do not manually create project structure** — use the
tools. If a tool is missing something, fix the tool.

## Quick Start

```sh
cyrius deps                                   # resolve stdlib deps
cyrius build src/main.cyr build/attn11        # build (train + sample)
cyrius test                                   # [build].test + tests/*.tcyr (grad checks)
cyrius build tests/attn11.fcyr build/fuzz && ./build/fuzz   # fuzz harness
cyrius build tests/attn11.bcyr build/bench && ./build/bench # benchmarks
cyrius lint src/*.cyr                         # static checks
CYRIUS_DCE=1 cyrius build src/main.cyr build/attn11         # release build (DCE)
cyrius build --aarch64 tests/attn11.tcyr build/test_a64 && qemu-aarch64 build/test_a64  # aarch64 via qemu
cyrius build --agnos src/main.cyr build/attn11_agnos        # AGNOS ring-3 target (build-only here; see docs/guides/agnos.md)
```

## Key Principles

- **Correctness over cleverness** — if it's wrong, the bugs own you
- Test after every change, not after the feature is "done"
- ONE change at a time — never bundle unrelated changes
- **Grad-check discipline**: every hand-derived backward op is verified against
  central finite differences before it lands — a backward without a passing
  grad check is incomplete
- Benchmark before claiming perf — numbers or it didn't happen
- Research before implementation — check vidya / existing patterns
- Build with `cyrius build`, not raw `cat file | cc5` — the manifest
  auto-resolves deps and prepends includes
- Source files only need project includes — stdlib / external deps auto-resolve
  from `cyrius.cyml`
- Every buffer declaration is a contract: `var buf[N]` = N **bytes**, not N entries
- **Own the stack** — where AGNOS already owns a domain (stats → `pramana`,
  vectors/matrices → `hisab`), prefer the AGNOS crate over rolling our own. The
  transformer math is attn11's own domain; the rest is borrowed.

## Rules (Hard Constraints)

- **Read the genesis repo's CLAUDE.md first** — [agnosticos/CLAUDE.md](https://github.com/MacCracken/agnosticos/blob/main/CLAUDE.md)
- **Do not commit or push** — the user handles all git operations
- **NEVER use `gh` CLI** — use `curl` to the GitHub API if needed
- Do not add unnecessary dependencies
- Do not skip tests / fuzz / benchmark verification before claiming work done
- Do not use `sys_system()` with unsanitized input — command injection risk
- Do not trust external data (file / network / args) without validation
- Do not modify `lib/` files (vendored stdlib / dep symlinks)
- Do not hardcode toolchain versions in CI YAML — `cyrius = "X.Y.Z"` in
  `cyrius.cyml` is the only source of truth

## Cyrius Conventions

- Everything is i64; an `f64` is its bit pattern carried in an i64. Use the
  `f64_*` builtins (`f64_add`/`f64_mul`/…), never `+`/`*` on float values.
- This toolchain's f64 comparisons are **not** NaN-correct; check finiteness on
  the bit pattern (`f64_is_finite`).
- Long-digit float literals mis-parse — build precise constants from integer
  ratios or runtime math.
- `print(s, len)` counts **bytes**; use `puts(s)` (strlen-based) for UTF-8.
- SIMD-typed vars must never be reassigned (miscompiles) — accumulate in plain
  byte buffers via `f64v_fmadd`. See [`docs/architecture/`](docs/architecture/).
- No negative literals — write `(0 - N)`. `>>` is a logical shift.

## CI / Release

- **Toolchain pin**: `cyrius = "X.Y.Z"` in `cyrius.cyml [package]` — CI and
  release read this; no version strings hardcoded in YAML.
- **DCE**: production builds run `CYRIUS_DCE=1`.
- **Tag filter**: release triggers on semver tags; verifies `VERSION == tag`
  before building. Prereleases for `0.x`.
- **Workflows**: `ci.yml` (lint, DCE build + ELF verify, grad-check suite, fuzz,
  bench, aarch64 cross+qemu, security, docs) — reusable via `workflow_call`;
  `release.yml` (version gate → CI gate → build → source tarball + binary +
  SHA256SUMS).

## Documentation

- [`docs/adr/`](docs/adr/) — Architecture Decision Records (*why X over Y?*)
- [`docs/architecture/`](docs/architecture/) — Non-obvious constraints (*what's true about the code?*)
- [`docs/guides/`](docs/guides/) — Task-oriented how-tos
- [`docs/examples/`](docs/examples/) — Runnable examples
- [`docs/sources.md`](docs/sources.md) — Academic citations for every algorithm (transformer, Adam, GELU, …)
- [`docs/audit/`](docs/audit/) — Security/correctness audit reports
- [`docs/benchmarks.md`](docs/benchmarks.md) — Performance history
- [`docs/development/state.md`](docs/development/state.md) — Live state snapshot
- [`docs/development/roadmap.md`](docs/development/roadmap.md) — Milestones through v1.0

New quirks land in `docs/architecture/` as numbered `NNN-kebab-case.md`; new
decisions land in `docs/adr/` from [`template.md`](docs/adr/template.md).
**Never renumber either series.**

## Process

### P(-1): Hardening (before features, at minor cuts, before v1.0)

1. **Cleanliness** — `cyrius build`, `cyrius lint`; all tests pass
2. **Benchmark baseline** — capture CSV for comparison
3. **Deep review** — correctness, memory, edge cases, docs
4. **Security audit** — input handling, syscalls, buffers, pointers; file in `docs/audit/YYYY-MM-DD-audit.md`
5. **Tests/benchmarks from findings**; prove wins against the baseline
6. **Documentation audit** — ADRs, source citations, guides

### Work Loop (continuous)

1. **Work phase** — features, roadmap items, bug fixes
2. **Build + grad-check** — `cyrius build`; grad-check any new backward op
3. **Test + benchmark additions** for new code
4. **Internal review** — performance, memory, correctness, edge cases
5. **Documentation** — CHANGELOG, `docs/development/state.md`, roadmap, any ADR earned
6. **Version sync** — `VERSION`, `cyrius.cyml` (`${file:VERSION}`), CHANGELOG header

### Security Hardening (before every release)

Input validation · buffer safety (N is bytes) · syscall return handling ·
pointer/bounds validation · no `sys_system` with unsanitized input · no path
traversal · document findings in `docs/audit/`.
