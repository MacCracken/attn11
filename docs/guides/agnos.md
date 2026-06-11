# Building attn11 for AGNOS

attn11 cross-builds for the **AGNOS kernel** (the sovereign OS Cyrius writes)
as a ring-3 userland application. This guide covers what the target is, how
the build works, where attn11 bridges the Linux/AGNOS ABI gaps, and how the
binary is meant to reach a running kernel. Roadmap context: milestone **M5**
in [`../development/roadmap.md`](../development/roadmap.md).

## Build

```sh
cyrius deps
cyrius build --agnos src/main.cyr build/attn11_agnos        # the model
cyrius build --agnos tests/attn11.tcyr build/test_agnos     # grad-check suite
```

`--agnos` defines `CYRIUS_TARGET_AGNOS`, which makes the stdlib selector pull
the AGNOS peers (`syscalls_x86_64_agnos.cyr`, `alloc_agnos.cyr`,
`args_agnos.cyr`) instead of the Linux ones. The output must be — and is — a
**static x86_64 ELF64**: AGNOS exec-from-disk (`elf_load_from_file`) loads
static ELF64 only. CI's `agnos` lane builds both targets and verifies the ELF
shape on every push.

## What the target lacks, and how attn11 bridges it

The AGNOS userland ABI (`agnos/docs/development/agnos-userland-abi.md`,
frozen syscalls 0–28 + 29–33) is deliberately smaller than Linux's. Four gaps
matter to attn11; each is bridged in [`src/fileio.cyr`](../../src/fileio.cyr)
behind `#ifdef CYRIUS_TARGET_AGNOS` so the same source compiles everywhere:

| Linux habit | AGNOS reality | attn11 bridge |
|---|---|---|
| `fstat(fd)`, st_size @ 48 | no fstat; path-based `stat(path, len, buf)`, size @ 16 (§4.1) | `_file_size(fd, path)` — fd on Linux, path-stat on AGNOS |
| `fsync(fd)` before atomic rename | no per-fd fsync; global `sync()` (syscall 12) | `_fsync` → `sys_sync()` on AGNOS |
| `unlink(path)` / `rename(old, new)`, NUL-terminated | explicit path lengths: `unlink(path, len)`, `rename(old, olen, new, nlen)` (§3.2) | `_unlink` / `_rename` shims |
| `O_*` open flags, `O_NOFOLLOW` | `AO_*` flags (§3.3); no nofollow bit | stdlib `io.cyr file_open` maps `O_*`→`AO_*`; `O_NOFOLLOW` is dropped (see below) |

Everything else was already portable: `sys_read`/`sys_write`/`sys_close`/
`sys_exit` have the same shape on both targets (only the numbers differ, and
those live in the per-target stdlib peer), the one-shot training arena sits on
`alloc()` (AGNOS: chunked bump over `mmap(27)`, 2 MB-granular), and the PRNG
is attn11's own deterministic xorshift64 — no kernel entropy needed.

Two security-relevant deltas to be aware of on AGNOS:

- **No `O_NOFOLLOW`**: the `AO_*` flag set has no nofollow bit, so the
  symlink-refusal hardening that `secure_read_file`/`secure_write_atomic`
  apply on Linux does not exist there. AGNOS's filesystem surface (no
  user-writable symlink farms, single-user ring-3 model) makes the TOCTOU
  class mostly theoretical today, but it is a *weaker* guarantee, not an
  equivalent one.
- **Path-stat instead of fstat**: `_file_size` stats the path after the open,
  so the size read races the open by a window. The size is only used as an
  allocation cap; the read loop stops at actual EOF regardless.

## Running under the kernel

Running (as opposed to building) needs a booted AGNOS: `qemu-system-x86_64` +
OVMF + gnoboot + an ext2 disk image with the binary staged into the rootfs.
attn11 ships its own end-to-end harness:

```sh
./scripts/agnos-smoke.sh          # needs sibling repos: agnos, gnoboot, agnoshi
```

It builds `build/attn11_agnos`, produces a native-Linux reference checkpoint
(`--steps N --save`), assembles a GPT boot image (gnoboot ESP + kernel + ext2
rootfs with `/bin/agnsh` and `/bin/attn11` — the recipe mirrors
`agnos/scripts/agnsh-smoke.sh`), boots it, types
`run /bin/attn11 --steps N --save /ck.ckpt` over the emulated xHCI keyboard
(agnsh is an AI-native shell, so a bare path would be parsed as natural
language — `run` is the committed launch verb, and the kernel's execwait #37
tokenizes the line into argv), waits for the save + samples on serial, then
extracts `/ck.ckpt` from the ext2 partition with `debugfs` and `cmp`s it
against the Linux reference. **PASS means the checkpoint is bit-for-bit
identical** — the M5 acceptance gate. Knobs: `STEPS=N` (default 50),
`AGNOS_SMOKE_KVM=1` (default TCG; see the comment in the script).

One subtlety the harness handles: the Linux reference runs under
`qemu-x86_64` (user-mode) when the guest is TCG. x87 transcendental
instructions (the `f64_exp`/`f64_tanh` paths) have implementation-defined
precision, so qemu's softfloat and real silicon differ by ULPs — comparing a
native-CPU run against a TCG guest shows ~11% of checkpoint bytes off by one
last bit while every displayed loss digit still matches. Holding the CPU
implementation constant isolates what the gate actually claims: the attn11 +
AGNOS software stack (syscall bridges, allocator, file I/O, checkpoint path)
is bit-exact with Linux. This mirrors how the aarch64 lane runs under
`qemu-aarch64` rather than comparing across silicon.

The agnos repo's own staging convention also fits: `stage-tools.sh` expects
`<repo>/build/<name>_agnos` from siblings and stages each as `/bin/<name>`;
adding an `attn11` row there is the upstream integration step.

The build-side gates (clean `--agnos` compile of binary + grad-check suite,
static ELF shape) are green in CI; CI does not boot the kernel, so the smoke
stays a developer-side gate.
