.PHONY: check release lint test fuzz bench build aarch64 clean

check: lint test            # the local CI gate (lint + x86 grad-checks)

# Full release-cut gate: everything CI gates, locally, before tagging.
# (aarch64 needs qemu; bench appends to bench-history.csv.)
release: lint test aarch64 build fuzz
	@echo "release gate: lint + x86 test + aarch64/qemu + DCE build + fuzz all green"

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

clean:
	rm -rf build/
