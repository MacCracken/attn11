# cyrius/agnos: argv init-stack capture emitted after call-bearing gvar inits

**Date**: 2026-06-10 · **Found in**: cyrius 6.1.31, agnos 1.44.15 ·
**Affects**: any agnos-target program whose entry is the scaffold-standard
`var r = main(); sys_exit(r);` · **Upstream**: cyrius (cycc codegen)
· **Severity**: HIGH (CLI flags silently ignored on agnos — same user-visible
symptom the v6.1.14 fix addressed)

## Symptom

`run /bin/attn11 --steps 50 --save /ck.ckpt` under agnsh launches attn11, but
it trains the default 2000 steps and never saves: `argc() == 0` inside
`main()`. The same binary's flags work on Linux. A minimal probe printing
`_agnos_init_rsp` shows it is still `0` inside `main`.

## Root cause

cycc (src/main.cyr, the v6.1.14 fix) emits `call _agnos_capture_rsp` after
`EMIT_GVAR_INITS` and before `PARSE_PROG`, on the assumption that gvar
initializers don't move `rsp`-relevant state. But a top-level
`var r = main();` is a gvar initializer **containing a call** — it is emitted
inside the gvar-init block, so the program's whole `main()` runs *before* the
capture. Disassembly of an affected binary (attn11's argtest probe):

```
init+0x549   call main                 ; emitted by EMIT_GVAR_INITS (var r = main())
init+0x55b   call _agnos_capture_rsp   ; v6.1.14 placement — one call too late
```

The v6.1.14 validation target (bnrmr) evidently doesn't use the
call-initializer entry shape, which is why the fix held there and not here.

## Repro

Any agnos build of:

```cyr
fn main(): i64 { ...print argc()...; return 0; }
var r = main();
sys_exit(r);
```

→ `argc()==0`. Change the tail to `var r = 0; r = main(); sys_exit(r);`
(call as a statement → emitted in PARSE_PROG, after the capture) → argv
works (`argc=4` for `run /bin/argtest alpha beta 7`, agnos 1.44.15, kernel
staging verified fine).

## Suggested upstream fix

Emit the capture call **before** `EMIT_GVAR_INITS` and drop the
`var _agnos_init_rsp = 0;` initializer in `lib/args_agnos.cyr` (declare it
BSS-zero) so the gvar-init pass can't clobber the captured value — or detect
call-bearing initializers and emit the capture ahead of the first one. Same
consideration applies to the macho `_macho_capture_args` twin (same
placement, same assumption).

## Downstream state

attn11 works around it by using the statement-call epilogue in all five entry
files — see `docs/architecture/002-agnos-entry-epilogue.md`. The workaround
is correct on every target and can stay after the upstream fix.
