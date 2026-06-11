# attn11 — M5 (AGNOS port) Delta Audit

> **Date**: 2026-06-10 | **Scope**: the v0.6.0 de-Linux diff (post-0.5.1) |
> **Method**: multi-agent adversarial review (6 lenses, 3 adversarial
> refuters per finding) + manual checklist over the new target-conditional
> code. Baseline: [2026-06-08-audit.md](2026-06-08-audit.md) — its findings
> and guarantees are unchanged on Linux; this file records only the *delta*
> the AGNOS target introduces.

## Summary

| Source | Confirmed | Severity | Status |
|--------|-----------|----------|--------|
| Adversarial review (semantics/CI/docs lenses) | 1 | high | fixed |
| Run-gate testing under the booted kernel      | 1 | high | worked around + filed upstream |
| Manual checklist (ABI/regression/security)    | 0 confirmed, 2 documented deltas | info | documented |

The review's ABI/regression/security agents were partially lost to a session
limit mid-run; those lenses were covered instead by (a) direct verification of
every offset/arity/flag against `agnos-userland-abi.md` and
`lib/syscalls_x86_64_agnos.cyr`, (b) the full test matrix re-run (52 checks
x86_64 + aarch64, fuzz, bench), and (c) the end-to-end run gate
(`scripts/agnos-smoke.sh`) exercising the entire bridged file-I/O path under
the real kernel.

## Confirmed finding (fixed)

- **HIGH — toolchain pin 6.1.6 predates HIGH-sev agnos codegen fixes.**
  cyrius 6.1.13 fixed indirect calls returning 0 on the agnos target; 6.1.14
  fixed `argc()`/`argv()` returning 0/null (init-stack capture placed in the
  entry epilogue — after top-level code like attn11's `var r = main()` has
  moved rsp). A CI-built agnos binary at the 6.1.6 pin would *silently ignore
  every CLI flag* while the build-only lane stays green; local validation had
  used the drifted 6.1.31 wrapper, i.e. a different compiler than CI pins.
  **Fix**: pin bumped to `6.1.31`, `lib/` re-synced to the matching snapshot
  (deps reshuffle: + `ganita`, − `matrix`, − `random`), full matrix re-run
  green.

## Confirmed finding from run-gate testing (worked around)

- **HIGH — agnos argv silently empty with the scaffold entry shape.** cycc
  (through 6.1.31) emits a top-level `var r = main();` initializer inside the
  gvar-init block — *before* the v6.1.14 `_agnos_capture_rsp` call — so on
  agnos `main()` ran with `argc()==0` and every CLI flag was ignored, with no
  error. Found because the run gate's `--steps 50 --save` was ignored (it
  trained the default 2000 steps); root-caused by binary disassembly
  (`call main` at init+0x549, capture at init+0x55b) and a minimal argv probe
  on the booted kernel. **Workaround**: statement-call epilogues
  (`var r = 0; r = main();`) in all five entry files — correct on every
  target. **Upstream**: filed in
  `docs/development/issues/2026-06-10-cyrius-agnos-capture-after-gvar-init-call.md`.
  Security relevance: flags that *harden* a run (e.g. a size-capped `--corpus`
  path instead of attacker-chosen stdin) could be dropped without any
  indication — silent argument loss is a correctness *and* a hardening hole.

## Documented security deltas on the AGNOS target

These are *weaker guarantees than Linux*, inherent to the frozen agnos ABI
(syscalls 0–33), accepted and disclosed in
[`docs/guides/agnos.md`](../guides/agnos.md):

1. **No `O_NOFOLLOW`** — the `AO_*` flag set has no nofollow bit, so the
   symlink-refusal hardening on `secure_read_file`/`secure_write_atomic` does
   not exist under AGNOS (the stdlib `io.cyr` bridge drops the bit). Linux
   builds keep it. Exposure is bounded by AGNOS's single-user ring-3 model
   and the absence of user-writable symlink farms in its filesystems today.
2. **Path-stat instead of fstat** — `_file_size(fd, path)` must stat the
   *path* after the open (no fstat in the ABI), so the size read races the
   open. The size is used only as an allocation cap; the read loop stops at
   actual EOF, so a swap can at worst cause a clean reject or a short read —
   never an overflow (the cap still bounds the buffer).

## Checklist over the new code

- **Buffer safety** — agnos `var st[48]` exactly matches `STAT_BUFSZ` (ABI
  §4.1, 48-byte struct, size at +16); Linux `var st[144]` unchanged (st_size
  at +48). The `.tmp` path buffer (`alloc(plen + 6)`, writes `plen + 5`
  bytes incl. NUL) is unchanged by this diff. ✅
- **Arity/ABI** — `_unlink`/`_rename` shims match the explicit-length agnos
  signatures (§3.2: `unlink(path,len)`, `rename(old,olen,new,nlen)` with
  a4=r10); Linux branches keep the NUL-terminated wrappers. Verified against
  `lib/syscalls_x86_64_agnos.cyr` and exercised end-to-end by the smoke's
  crash-atomic save under the kernel. ✅
- **Durability** — `_fsync` on agnos falls back to global `sys_sync()` (no
  per-fd fsync in the frozen ABI); failure still aborts the atomic rename. ✅
- **Syscall returns** — agnos returns -1 (not -errno); all fileio callers
  branch on `< 0` only, so the coarser error code changes messages, not
  control flow. ✅
- **No new external input surface** — the diff adds no parser, no network,
  no new file format; the smoke harness consumes only local sibling-repo
  artifacts. ✅
