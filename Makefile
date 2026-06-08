.PHONY: check lint test fuzz bench build aarch64 clean

check: lint test            # the local CI gate

lint:
	@for f in src/*.cyr; do cyrius lint $$f; done

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
