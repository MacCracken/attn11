# Audit — 0.8.0 (M8: security sweep), 2026-06-11

**Scope**: a research-driven hardening pass over attn11's whole hostile-input
and supply-chain surface — not a single diff. **Method**: a survey→map
workflow — six vulnerability classes each web-researched against recent
(2023–2026) CVEs, then **adversarially mapped** onto attn11's actual code
(each mapping agent instructed to *find* a way the class applies, citing
exact `file:line` guards before calling anything mitigated). 12 agents; per-class
dispositions recorded below (negative results included — the trail is the point).

## Per-class dispositions

| class | disposition | crux |
|---|---|---|
| ML model-file deserialization | **mitigated (format-immune)** | flat native-LE i64 array — no opcode interpreter, no callable revival, no embedded path; structurally immune to the pickle/Keras/numpy-`allow_pickle` RCE genre |
| Integer-overflow → OOB in the parser | **mitigated** | every header factor individually capped (`V≤768 −4`, `C≤4096 −5`, `T≤8192 −7`, `NL≤128 −8`, `np≤4M −13`, `Vb≤256 −33`, `K≤512 −34`); re-derived worst-case products all ≪ 2⁶³; the **exact** `need==len` size check (`−11`) is recomputed from validated fields |
| Alloc-bomb / resource exhaustion | **mitigated** | merge-table expansion bounded to 64 B (`−38`), the `model_alloc_bytes` 128 MB pre-flight (`−18`), the 4 MB corpus cap; no unbounded count-driven loop/alloc |
| File input (traversal / TOCTOU) | **partially-mitigated** | `O_NOFOLLOW` + `fstat`-then-loop-read on both loaders; **one real bug** (the dropped `_file_size` path arg) + accepted residuals (leaf-only `O_NOFOLLOW`, no `S_ISREG` gate) |
| Supply chain / toolchain / CI | **vulnerable → partially repaired** | floating action tags, `curl\|sh` installer, self-attested `SHA256SUMS`; the cheap mechanical items fixed, the infra-bound items documented as deferred |
| CLI / numeric parsing | **partially-mitigated** | `_atoi` had no overflow guard (now saturated); memory-safety already covered by `model_config_ok` |

## Confirmed findings → fixes

1. **[medium] `ckpt_load_file` dropped the `_file_size` path arg → crash on every
   AGNOS `--load`.** `_file_size(fd, path)`'s AGNOS branch path-stats via
   `strlen(path)`; the checkpoint loader called `_file_size(fd)` (one arg), so
   `path` was a garbage register fed to `strlen` — an OOB read / SIGSEGV under
   the kernel on every load, before any content is examined, violating the
   loader's "never a crash" contract. Masked on Linux (the `fstat` branch
   ignores `path`); the arity mismatch compiled silently. **Fixed**:
   `persist.cyr` now calls `_file_size(fd, path)` (matches the corpus loader).

2. **[medium] Checkpoint *save* was broken on the entire aarch64 lane (qemu
   `fsync` quirk).** Exposed by the new `test_ckpt_file_roundtrip` (the file-path
   loader had **never** been exercised on aarch64 — only the in-memory
   `ckpt_load_buf`). `secure_write_atomic`'s durability barrier issued a raw
   `fsync` syscall; under **qemu-user aarch64** `fsync` (82) returns `EFAULT`
   (an emulation quirk — it works on real aarch64), so every `ckpt_save_file`
   returned `−2` and wrote nothing. Probed it precisely: raw `getpid`/`write`/
   **`fdatasync`** all work under qemu-aarch64; only `fsync` is mis-emulated.
   **Fixed**: the barrier now uses **`fdatasync`** (75 x86_64 / 83 aarch64),
   which flushes the temp's data + the size/block metadata a read needs (the
   whole crash-atomic guarantee here — it skips only mtime/atime) and is
   emulated correctly. Verified: real aarch64 binary saves *and* loads under
   qemu; 247/247 on both arches.

3. **[low] `_atoi` had no overflow guard** (also flagged by the M7 review). A
   garbage-huge `--steps`/`--layers`/… could wrap mod 2⁶⁴ to a plausible
   small/negative value. Memory-safety was already covered by
   `model_config_ok`'s caps, but **fixed** as defense-in-depth: `_atoi`
   saturates at ~1e9 (≫ any real arg, ≪ overflow), so the parse is always a
   bounded non-negative number that downstream caps reject cleanly.

4. **[hardening] `lens[6144]` merge-validation scratch sat at the exact
   `BPE_VMAX` boundary.** `−33`/`−34`/`−35` already keep the minted id `< 768`,
   but a future `BPE_VMAX` bump that missed this stack buffer would silently
   overflow it. **Fixed**: an explicit `(Vb+j) ≥ 768 → −37` bound pins the
   minted id to the buffer's capacity regardless of the cap constants.

5. **[supply chain] CI hardening** (the "vulnerable" class — mechanical items):
   - **SHA-pinned** every GitHub Action (`actions/checkout@v4` →
     `34e1148…`, `softprops/action-gh-release@v2` → `3bb1273…`) — closes the
     floating-tag / retag-compromise vector (the tj-actions March-2025 class).
   - **Killed the `GITHUB_REF_NAME` awk-injection** in `release.yml`: the tag
     is now passed as `awk -v` data, never interpolated into the awk program.
   - **Scoped least-privilege**: `contents: write` moved from the workflow top
     level to the `release` job only; the CI-gate job now runs read-only.

## Coverage gaps closed

- **`test_ckpt_file_roundtrip`** — the file-path loader (`ckpt_save_file`/
  `ckpt_load_file`, hence `_file_size`/`_fdatasync`) was untested; only the
  in-memory `ckpt_load_buf` was. The new test drives a save→load→bit-compare
  round-trip and is what surfaced findings #1's path and #2.
- **`agnos-smoke.sh` now `--load`s on AGNOS** — the gate saved a checkpoint and
  byte-compared it to Linux but **never loaded** it, so the AGNOS loader's
  portable-syscall surface (where #1 lived) went ungated. It now runs
  `--load … --gen-only` and asserts the unique "resumed from checkpoint"
  marker. (Needs a developer AGNOS boot to validate end-to-end; the bash +
  embedded-Python both parse clean.)
- **Two new fuzz modes** in `tests/attn11.fcyr`: a **boundary-combination**
  mode (every size field at/over its cap simultaneously — stresses the size
  arithmetic + check ordering) and a **max-vocab triple** mode (`V=768,
  Vb=256, K=512` — the exact `BPE_VMAX`/`lens[6144]` boundary).

## Accepted residuals / deferred (documented, not repaired)

- **[high, deferred] `curl … install.sh | sh` from a mutable `main` branch**
  (toolchain bootstrap in all CI jobs). Recommendation: pin `install.sh` to a
  commit SHA on `raw.githubusercontent.com`, download-then-checksum before
  running. Deferred: it is a cross-repo (`MacCracken/cyrius`) toolchain-flow
  decision, and the `cyrius.cyml` pin (not the YAML) is the version source of
  truth (CLAUDE.md).
- **[medium, deferred] `SHA256SUMS` is self-attested** in the same unsigned
  release job as the binary. Recommendation: GitHub build-provenance
  attestation (OIDC) and/or minisign/cosign with a key absent from the runner.
  Deferred: needs signing infrastructure / key management.
- **[medium, deferred] No lockfile binds the resolved `lib/` closure to the
  cyrius pin** (the pin/snapshot drift trap, `docs/architecture/002`).
  Recommendation: a committed `toolchain.sha256` over `lib/` + compiler,
  diffed in CI. Deferred.
- **[hardening] `O_NOFOLLOW` guards only the leaf** — a symlinked *parent dir*
  of `--corpus`/`--load`/`--save` is still followed (and AGNOS has no
  `O_NOFOLLOW` at all, per `docs/audit/2026-06-10-agnos-audit.md`). Accepted
  for the local single-user trust model; revisit with `openat2
  RESOLVE_NO_SYMLINKS` if a hostile-parent-dir scenario enters scope.
- **[low] No `S_ISREG` gate** — a FIFO / `/dev/*` / `/proc/*` is accepted via
  `--corpus`. Accepted: the path is *user-supplied* (not hostile data), and
  the `fstat` size-cap + the `size < 1` empty-check neutralize the common cases
  (special files report `st_size == 0`).
- **[hardening] `apt install qemu-user-static`** is unpinned (aarch64 job).
  Accepted residual; pin the package version or use a digest-pinned container
  if reproducibility tightens.

## Result

247 checks green on x86_64 **and** aarch64 (qemu) — incl. the new file-path
round-trip, which now exercises checkpoint save+load on both arches; fuzz green
(500 byte + 500 BPE + the two new boundary modes + 100 corpora + round-trip);
lint clean; DCE static-ELF green; Linux + AGNOS builds green. All confirmed
findings fixed and regression-tested in this release; every surveyed class has
a recorded disposition.
