.PHONY: check release lint test fuzz bench build aarch64 smoke clean

check: lint test            # the local CI gate (lint + x86 grad-checks)

# Full release-cut gate: everything CI gates, locally, before tagging.
# (aarch64 needs qemu; bench appends to bench-history.csv.)
release: lint test aarch64 build fuzz smoke
	@echo "release gate: lint + x86 test + aarch64/qemu + DCE build + fuzz + smoke all green"

# Hard gate (mirrors ci.yml): `cyrius lint` exits 0 even WITH warnings, so the
# loop must fail the build itself on any `warn ` line — otherwise a lint warning
# slips through `make check` locally and only trips CI. Keep in lockstep with
# the Lint step in .github/workflows/ci.yml.
lint:
	@fail=0; \
	for f in src/*.cyr; do \
	  out=$$(cyrius lint "$$f" 2>&1 || true); \
	  echo "$$out"; \
	  if echo "$$out" | grep -qE '^[[:space:]]*warn '; then echo "lint: warnings in $$f"; fail=1; fi; \
	done; \
	[ $$fail -eq 0 ] || { echo "lint: warnings present"; exit 1; }

test:
	cyrius test

fuzz:
	CYRIUS_DCE=1 cyrius build tests/attn11.fcyr build/fuzz && ./build/fuzz

bench:
	./scripts/bench-history.sh

build:
	@mkdir -p build
	CYRIUS_DCE=1 cyrius build src/main.cyr build/attn11

# Cross-build the grad-check suite for aarch64 and run it under qemu.
aarch64:
	@mkdir -p build
	cyrius build --aarch64 tests/attn11.tcyr build/test_a64 && qemu-aarch64 build/test_a64

# CLI smoke: hostile / out-of-range argument combinations must be REJECTED
# cleanly (exit 1), never crash (the grad-check/fuzz harnesses exercise the math
# and the checkpoint loader, but NOT the CLI arg path). Pins the 1.4.5 fix — an
# unbounded `--layers N --attn-every K` was a stack-buffer overflow (SIGSEGV);
# any exit >= 128 here is a signal/crash and fails the gate.
smoke: build
	@echo "smoke: hostile CLI args must reject cleanly, never crash"
	@for args in "--layers 100000 --attn-every 2" "--layers 129 --attn-every 2" "--layers 0 --attn-every 2" "--ternary --experts 4" "--ternary --objective diffusion" "--ternary --attn-kind ssm"; do \
	  ./build/attn11 $$args --gen-only >/dev/null 2>&1; rc=$$?; \
	  if [ $$rc -ge 128 ]; then echo "smoke: CRASH (signal $$((rc-128))) on: $$args"; exit 1; fi; \
	  if [ $$rc -eq 0 ]; then echo "smoke: expected rejection but succeeded on: $$args"; exit 1; fi; \
	done; \
	./build/attn11 --preset --attn-kind ssm --attn-every 2 --gen-only >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid mha/ssm hybrid failed to build"; exit 1; fi; \
	./build/attn11 --ternary --steps 5 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid --ternary run failed"; exit 1; fi; \
	./build/attn11 --gen-only --eval-corpus /nonexistent/attn11-smoke >/dev/null 2>&1; rc=$$?; \
	if [ $$rc -ge 128 ]; then echo "smoke: CRASH (signal $$((rc-128))) on missing --eval-corpus file"; exit 1; fi; \
	if [ $$rc -eq 0 ]; then echo "smoke: missing --eval-corpus file should set a non-zero exit"; exit 1; fi; \
	printf 'streaming smoke corpus -- the quick brown fox jumps. ' > /tmp/attn11-smoke-seed.txt; \
	for i in 1 2 3 4 5 6 7 8; do cat /tmp/attn11-smoke-seed.txt; done > /tmp/attn11-smoke-corpus.txt; \
	./build/attn11 --corpus /tmp/attn11-smoke-corpus.txt --encode-shard /tmp/attn11-smoke.tsh >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: --encode-shard failed on a valid corpus"; exit 1; fi; \
	./build/attn11 --stream-corpus /tmp/attn11-smoke.tsh --steps 5 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid --stream-corpus run failed"; exit 1; fi; \
	for sa in "--stream-corpus /nonexistent/attn11-smoke.tsh" "--stream-corpus /tmp/attn11-smoke-corpus.txt" "--stream-corpus /tmp/attn11-smoke.tsh --corpus /tmp/attn11-smoke-corpus.txt" "--stream-corpus /tmp/attn11-smoke.tsh --bpe 8"; do \
	  ./build/attn11 $$sa --gen-only >/dev/null 2>&1; rc=$$?; \
	  if [ $$rc -ge 128 ]; then echo "smoke: CRASH (signal $$((rc-128))) on: $$sa"; exit 1; fi; \
	  if [ $$rc -eq 0 ]; then echo "smoke: expected rejection but succeeded on: $$sa"; exit 1; fi; \
	done; \
	rm -f /tmp/attn11-smoke-seed.txt /tmp/attn11-smoke-corpus.txt /tmp/attn11-smoke.tsh; \
	echo "smoke: ok (hostile --layers/--attn-every/--ternary rejected; valid hybrid + ternary + stream; bad shard / held-out eval error cleanly)"

clean:
	rm -rf build/
