#!/bin/bash
# agnos-smoke.sh — the M5 run gate: boot attn11 under the REAL AGNOS kernel in
# QEMU, train + checkpoint there, and prove the saved checkpoint is BIT-FOR-BIT
# identical to a native Linux run with the same seed/steps.
#
# Flow:
#   1. cyrius build --agnos  -> build/attn11_agnos  (static x86_64 ELF64)
#   2. native Linux reference: ./build/attn11 --steps N --save ref.ckpt
#   3. assemble a GPT boot image (gnoboot ESP + agnos kernel + ext2 rootfs
#      seeded with /bin/agnsh and /bin/attn11) — same recipe as
#      agnos/scripts/agnsh-smoke.sh, built under build/agnos-smoke/
#   4. boot qemu-system-x86_64 (OVMF, NVMe, serial->log, xHCI keyboard via the
#      HMP monitor), wait for the agnsh banner, type
#      `/bin/attn11 --steps N --save /ck.ckpt`, wait for the save + samples
#   5. dd the ext2 partition out, e2fsck -fn it, debugfs-dump /ck.ckpt, and
#      cmp against the Linux reference
#
# PASS = the binary trains + samples under AGNOS AND the two checkpoints are
# byte-identical. See docs/guides/agnos.md.
#
# Requires sibling repos (override roots via env):
#   AGNOS_ROOT   (default ../agnos)    — built kernel at build/agnos
#   GNOBOOT_ROOT (default ../gnoboot)  — build/BOOTX64.EFI
#   AGNOSHI_ROOT (default ../agnoshi)  — build/agnsh_agnos
# Tools: qemu-system-x86_64, OVMF, parted, sgdisk, mtools, mkfs.ext2, e2fsck,
#        debugfs, dd, python3.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGNOS_ROOT="${AGNOS_ROOT:-$ROOT/../agnos}"
GNOBOOT_ROOT="${GNOBOOT_ROOT:-$ROOT/../gnoboot}"
AGNOSHI_ROOT="${AGNOSHI_ROOT:-$ROOT/../agnoshi}"
STEPS="${STEPS:-50}"

OVMF_CODE=""
for c in /usr/share/edk2/x64/OVMF_CODE.4m.fd /usr/share/edk2/x64/OVMF_CODE.fd \
         /usr/share/OVMF/OVMF_CODE.fd /usr/share/OVMF/OVMF_CODE_4M.fd; do
    [ -f "$c" ] && { OVMF_CODE="$c"; break; }
done
OVMF_VARS_SRC=""
for c in /usr/share/edk2/x64/OVMF_VARS.4m.fd /usr/share/edk2/x64/OVMF_VARS.fd \
         /usr/share/OVMF/OVMF_VARS.fd /usr/share/OVMF/OVMF_VARS_4M.fd; do
    [ -f "$c" ] && { OVMF_VARS_SRC="$c"; break; }
done
[ -z "$OVMF_CODE" ] && { echo "ERROR: OVMF not found"; exit 1; }
[ -z "$OVMF_VARS_SRC" ] && { echo "ERROR: OVMF vars not found"; exit 1; }
for tool in qemu-system-x86_64 parted mformat mmd mcopy sgdisk mkfs.ext2 \
            e2fsck debugfs dd strings python3 cyrius; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: missing tool '$tool'"; exit 1; }
done

GNOBOOT="$GNOBOOT_ROOT/build/BOOTX64.EFI"
AGNOS="$AGNOS_ROOT/build/agnos"
AGNSH="$AGNOSHI_ROOT/build/agnsh_agnos"
[ -f "$GNOBOOT" ] || { echo "ERROR: gnoboot not built at $GNOBOOT"; exit 1; }
[ -f "$AGNOS" ]   || { echo "ERROR: agnos kernel not built at $AGNOS"; exit 1; }
[ -f "$AGNSH" ]   || { echo "ERROR: agnsh (agnos build) not at $AGNSH"; exit 1; }

cd "$ROOT"
WORK="$ROOT/build/agnos-smoke"
rm -rf "$WORK"; mkdir -p "$WORK"

# ---- 1. the agnos binary --------------------------------------------------
echo "== build: attn11 (agnos target)"
cyrius build --agnos src/main.cyr build/attn11_agnos || exit 1
DESC="$(file -b build/attn11_agnos)"
case "$DESC" in
    *"ELF 64-bit"*"x86-64"*"statically linked"*) : ;;
    *) echo "ERROR: not a static x86-64 ELF64 ($DESC)"; exit 1 ;;
esac

# ---- 2. Linux reference checkpoint ----------------------------------------
# The reference must run on the SAME f64 implementation as the guest: x87
# transcendentals (the f64_exp/f64_tanh paths) are implementation-defined, so
# TCG softfloat and real silicon legitimately differ by ULPs (~11% of
# checkpoint bytes after 50 steps). Guest on TCG → reference under
# qemu-x86_64 (user-mode); guest on KVM → native reference. Either way the
# comparison isolates the SOFTWARE stack (attn11 + kernel + file I/O), which
# is the M5 gate.
echo "== build: attn11 (native) + Linux reference checkpoint ($STEPS steps)"
cyrius build src/main.cyr build/attn11 || exit 1
if [ "${AGNOS_SMOKE_KVM:-0}" = "1" ] && [ -w /dev/kvm ]; then
    REF_RUNNER=""
else
    command -v qemu-x86_64 >/dev/null 2>&1 || { echo "ERROR: qemu-x86_64 (user-mode) needed for the TCG-matched reference"; exit 1; }
    REF_RUNNER="qemu-x86_64"
fi
$REF_RUNNER ./build/attn11 --steps "$STEPS" --save "$WORK/ref.ckpt" > "$WORK/linux-run.log" 2>&1 \
    || { echo "ERROR: Linux reference run failed"; tail -5 "$WORK/linux-run.log"; exit 1; }
[ -f "$WORK/ref.ckpt" ] || { echo "ERROR: no reference checkpoint"; exit 1; }
echo "   ref.ckpt: $(stat -c%s "$WORK/ref.ckpt") bytes"

# ---- 3. boot image (recipe mirrors agnos/scripts/agnsh-smoke.sh) ----------
echo "== image: GPT (ESP: gnoboot+kernel; ext2: /bin/agnsh /bin/attn11)"
IMG="$WORK/agnos-attn11.img"
PART_OFFSET=$(( 33 * 1048576 )); PART_BYTES=$(( 67 * 1048576 ))
PART_BLOCKS=$(( PART_BYTES / 4096 ))
EXT2_FEATURES="${EXT2_SMOKE_FEATURES:-^resize_inode,^dir_index,^metadata_csum,^64bit,^uninit_bg}"

SEED="$WORK/seed"; mkdir -p "$SEED/bin"
cp "$AGNSH" "$SEED/bin/agnsh"
cp build/attn11_agnos "$SEED/bin/attn11"

dd if=/dev/zero of="$IMG" bs=1M count=128 status=none
parted -s "$IMG" mklabel gpt \
    mkpart ESP fat32 1MiB 33MiB set 1 esp on \
    mkpart agnos-fs ext2 33MiB 100MiB
sgdisk -t 2:8300 "$IMG" >/dev/null
mformat -i "$IMG"@@1048576 -F
mmd -i "$IMG"@@1048576 ::EFI ::EFI/BOOT ::boot
mcopy -i "$IMG"@@1048576 "$GNOBOOT" ::EFI/BOOT/BOOTX64.EFI
mcopy -i "$IMG"@@1048576 "$AGNOS" ::boot/agnos
mkfs.ext2 -F -q -L AGNOS-ATTN11 -b 4096 -m 0 \
    -O "$EXT2_FEATURES" \
    -d "$SEED" -E offset=$PART_OFFSET "$IMG" $PART_BLOCKS

cp "$OVMF_VARS_SRC" "$WORK/vars.fd"; chmod +w "$WORK/vars.fd"

# ---- 4. boot + drive ------------------------------------------------------
# TCG by default — it is what agnos's own smokes validate against, and its
# f64 is IEEE-correct so the bit-for-bit comparison holds. KVM (much faster)
# is opt-in via AGNOS_SMOKE_KVM=1: observed 2026-06-10 that agnsh does not
# reach its banner under -enable-kvm on this kernel (boot-timing dependent;
# kybernet sees the thread-selftest procs still live), so it is not default.
if [ "${AGNOS_SMOKE_KVM:-0}" = "1" ] && [ -w /dev/kvm ]; then
    QEMU_ACCEL="-enable-kvm -cpu host"
else
    QEMU_ACCEL="-cpu max"
fi
echo "== boot: qemu $QEMU_ACCEL (train $STEPS steps + save + sample under AGNOS)"
STEPS="$STEPS" WORK="$WORK" IMG="$IMG" OVMF_CODE="$OVMF_CODE" QEMU_ACCEL="$QEMU_ACCEL" python3 - <<'PYEOF'
import os, socket, subprocess, sys, time

WORK = os.environ["WORK"]; IMG = os.environ["IMG"]
OVMF = os.environ["OVMF_CODE"]; STEPS = os.environ["STEPS"]
SER = os.path.join(WORK, "serial.log")
MON = os.path.join(WORK, "monitor.sock")
open(SER, "w").close()

qemu = subprocess.Popen([
    "qemu-system-x86_64", "-machine", "q35", "-m", "512M",
    *os.environ["QEMU_ACCEL"].split(),
    "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF}",
    "-drive", f"if=pflash,format=raw,file={WORK}/vars.fd",
    "-drive", f"file={IMG},format=raw,if=none,id=disk0",
    "-device", "nvme,drive=disk0,serial=AGNOS-ATTN11",
    "-device", "qemu-xhci,id=xhci", "-device", "usb-kbd,bus=xhci.0",
    "-serial", f"file:{SER}", "-display", "none", "-no-reboot",
    "-monitor", f"unix:{MON},server,nowait",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def fail(msg):
    print(f"FAIL: {msg}")
    tail = ser()[-2000:]
    print("  --- serial tail ---")
    for ln in tail.splitlines()[-25:]:
        print("  " + ln)
    qemu.kill()
    sys.exit(1)

s = None
for _ in range(100):
    try:
        s = socket.socket(socket.AF_UNIX); s.connect(MON); break
    except OSError:
        time.sleep(0.2)
if s is None: fail("no QEMU monitor socket")
s.settimeout(1.0)

def drain():
    try:
        while True: s.recv(65536)
    except OSError: pass

def ser():
    try: return open(SER, "rb").read().decode("latin1")
    except OSError: return ""

km = {' ': 'spc', '\n': 'ret', '-': 'minus', '.': 'dot', '/': 'slash'}
def typ(word):
    for ch in word:
        key = km.get(ch, ch)
        if ch.isupper(): key = "shift-" + ch.lower()
        s.sendall(("sendkey " + key + "\n").encode())
        time.sleep(0.10); drain()
    time.sleep(1.6)

ok = False
for _ in range(480):                      # <=120s — TCG boot can be slow
    if "agnoshi" in ser(): ok = True; break
    time.sleep(0.25)
if not ok: fail("no agnsh banner on serial")
print("   agnsh banner seen; typing the run")
time.sleep(2.0)

# agnsh is AI-native: a bare path is parsed as natural language. The `run`
# verb is the committed program-launch path (execwait #37; the kernel
# tokenizes "PATH args.." into argv). ASSIST mode launches without confirm.
cmd = f"run /bin/attn11 --steps {STEPS} --save /ck.ckpt\n"
typ(cmd)

# log_every is 250, so a short run prints NOTHING between "training:" and the
# save — silence is normal. Budget 20 min (TCG soft-float is ~100-300x; KVM
# finishes in seconds). Nudge past any agnsh confirm prompt if one appears.
saved = False; nudged = False
for i in range(4800):
    out = ser()
    if "saved checkpoint" in out: saved = True; break
    if not nudged and i > 40 and ("[y" in out[-200:] or "(y/" in out[-200:]):
        typ("y\n"); nudged = True
    time.sleep(0.25)
if not saved: fail("no 'saved checkpoint' on serial")
print("   checkpoint saved under AGNOS")

# The samples follow the save; wait for serial to settle, then quit cleanly.
prev = -1
for _ in range(120):
    cur = os.path.getsize(SER)
    if cur == prev: break
    prev = cur; time.sleep(1.0)
if "greedy sample" not in ser(): fail("training finished but no sample output")
print("   samples generated under AGNOS")
try:
    s.sendall(b"quit\n"); time.sleep(1.0)
except OSError:
    pass
qemu.wait(timeout=10)
sys.exit(0)
PYEOF
[ $? -eq 0 ] || exit 1

# ---- 5. extract + compare -------------------------------------------------
echo "== extract: ext2 partition -> /ck.ckpt"
dd if="$IMG" of="$WORK/part.img" bs=1M skip=33 count=67 status=none
e2fsck -fn "$WORK/part.img" > "$WORK/e2fsck.log" 2>&1 \
    || { echo "ERROR: post-boot e2fsck not clean"; tail -10 "$WORK/e2fsck.log"; exit 1; }
echo "   e2fsck: clean"
debugfs -R "dump /ck.ckpt $WORK/agnos.ckpt" "$WORK/part.img" >/dev/null 2>&1
[ -s "$WORK/agnos.ckpt" ] || { echo "ERROR: /ck.ckpt not found in the image"; exit 1; }
echo "   agnos.ckpt: $(stat -c%s "$WORK/agnos.ckpt") bytes"

if cmp -s "$WORK/ref.ckpt" "$WORK/agnos.ckpt"; then
    echo ""
    echo "agnos-smoke: PASS — AGNOS checkpoint is bit-for-bit identical to Linux ($STEPS steps)"
    exit 0
else
    echo ""
    echo "agnos-smoke: FAIL — checkpoints differ:"
    cmp "$WORK/ref.ckpt" "$WORK/agnos.ckpt" | head -3
    exit 1
fi
