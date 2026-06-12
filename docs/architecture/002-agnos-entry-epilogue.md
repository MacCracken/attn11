# 002 — agnos entry epilogue: `r = main()` must be a statement, not an initializer

**Status**: RETIRED as a load-bearing rule at pin ≥ 6.1.32 (was: active
workaround at 6.1.31) · **Affects**: every entry file (`src/main.cyr`,
`src/test.cyr`, `tests/attn11.{tcyr,bcyr,fcyr}`, `docs/examples/minimal_train.cyr`)

> **Resolution (2026-06-11, pin 6.1.31 → 6.1.33; pin is now 6.1.37, see
> state.md)**: cyrius 6.1.32 fixed the
> upstream gap attn11 filed — the agnos init rsp is now parked in
> callee-saved **r15 at the entry landing** (the first runtime instruction,
> ahead of any gvar-init call), and `lib/args_agnos.cyr` reads it via
> `_agnos_argv_base()`; the `_agnos_init_rsp` global and
> `_agnos_capture_rsp` are gone. Either entry shape now works on agnos.
> The statement-call epilogues below are kept in the entry files (harmless,
> and they document the history). **Drift warning**: a 6.1.32+ compiler
> against a ≤ 6.1.31 `lib/` snapshot reproduces the `argc()==0` symptom —
> the 0.7.0 run gate caught exactly that; keep the pin and `lib/` in sync.
> The rest of this note describes the ≤ 6.1.31 behavior for the record.

## The rule

```cyr
# WRONG on agnos — main() runs before the argv capture; argc()==0 inside it
var r = main();
sys_exit(r);

# RIGHT — the call is a top-level statement, emitted after the capture
var r = 0;
r = main();
sys_exit(r);
```

## Why

On the agnos target, `argc()`/`argv()` read the kernel exec init stack
(`[rsp]=argc, [rsp+8+i*8]=argv[i]`) through `_agnos_init_rsp`, a global that
cycc populates by emitting `call _agnos_capture_rsp` in the entry — placed
(since cyrius v6.1.14) *after* the gvar initializers and *before* the
top-level statements (`PARSE_PROG`).

The hole: a top-level `var r = main();` is a **gvar initializer with a call**,
so it is emitted in the gvar-init block — *before* the capture call. `main()`
then runs with `_agnos_init_rsp == 0` and `argc()` returns 0: every CLI flag
is silently ignored, on agnos only. Splitting declaration (`var r = 0;`, a
constant initializer) from the call (`r = main();`, a plain statement) moves
the call after the capture and argv works.

Diagnosed 2026-06-10 by disassembling the agnos binary: the entry's init
region called `main` at `init+0x549` and `_agnos_capture_rsp` at `init+0x55b`
— one call too late. A minimal probe (`run /bin/argtest alpha beta 7`) showed
`_agnos_init_rsp == 0` inside `main` with the old shape and
`argc=4 / argv[1..3] = alpha beta 7` with the new one, on agnos 1.44.15.
Linux is unaffected either way (argv comes from `/proc/self/cmdline`-free
stack reads at `args_init()` time, not an entry capture).

Upstream: this is a cyrius codegen gap (the v6.1.14 capture placement assumed
gvar initializers are inert). Filed as
[`docs/development/issues/2026-06-10-cyrius-agnos-capture-after-gvar-init-call.md`](../development/issues/2026-06-10-cyrius-agnos-capture-after-gvar-init-call.md);
when cycc emits the capture ahead of call-bearing gvar inits, this workaround
becomes harmless style and the rule can be retired.
