.PHONY: check release fmt lint test fuzz bench compete-bench gpu-test build aarch64 smoke clean

check: fmt lint test        # the local CI gate (fmt-check + lint + x86 grad-checks)

# Full release-cut gate: everything CI gates, locally, before tagging.
# (aarch64 needs qemu; bench appends to bench-history.csv.)
release: fmt lint test aarch64 build fuzz smoke
	@echo "release gate: fmt + lint + x86 test + aarch64/qemu + DCE build + fuzz + smoke all green"

# Format gate (mirrors the Format-check step in .github/workflows/ci.yml): `cyrius fmt
# --check` exits non-zero on an unformatted file. Same gate rosnet enforces — added at
# 1.10.0 so the two repos share it (the GPU backend now lives in rosnet; code moving
# attn11 -> rosnet must be fmt-clean first). Keep the file glob in lockstep with ci.yml.
fmt:
	@for f in src/*.cyr tests/*.tcyr tests/*.bcyr tests/*.fcyr tests/*.cyr; do \
	  cyrius fmt "$$f" --check || { echo "fmt: $$f is not formatted (run: cyrius fmt $$f)"; exit 1; }; \
	done; echo "fmt: all files clean"

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

# Competitor benchmarks (B-series): attn11 vs external references + the zero-deps
# story. Clones/builds competitors into a gitignored bench/ at recorded refs;
# emits competitor-bench.csv. Local/release-machine (needs network + competitor
# stacks), NOT a CI lane. See docs/benchmarks.md "Competitor benchmarks".
compete-bench:
	./scripts/compete-bench.sh

# M18 GPU matmul validation (tests/gpu_matmul.cyr): gpu_matmul_fwd vs the linear_fwd
# oracle, bit-exact, on attn11's real shapes. Environment-dependent (needs a native
# AMD f64 GPU) so it is NOT in the release gate — it SKIPS cleanly (exit 0) where no
# device exists, and is run on a GPU box. See docs/guides/gpu.md.
gpu-test:
	@mkdir -p build
	cyrius build tests/gpu_matmul.cyr build/gpu_matmul && ./build/gpu_matmul
	cyrius build tests/gpu_ln.cyr build/gpu_ln && ./build/gpu_ln
	cyrius build tests/gpu_gelu.cyr build/gpu_gelu && ./build/gpu_gelu
	cyrius build tests/gpu_head.cyr build/gpu_head && ./build/gpu_head
	cyrius build tests/gpu_attn.cyr build/gpu_attn && ./build/gpu_attn
	cyrius build tests/gpu_adam.cyr build/gpu_adam && ./build/gpu_adam
	cyrius build tests/gpu_gelu_bwd.cyr build/gpu_gelu_bwd && ./build/gpu_gelu_bwd
	cyrius build tests/gpu_linear_bwd.cyr build/gpu_linear_bwd && ./build/gpu_linear_bwd
	cyrius build tests/gpu_head_bwd.cyr build/gpu_head_bwd && ./build/gpu_head_bwd
	cyrius build tests/gpu_ln_bwd.cyr build/gpu_ln_bwd && ./build/gpu_ln_bwd
	cyrius build tests/gpu_attn_bwd.cyr build/gpu_attn_bwd && ./build/gpu_attn_bwd
	cyrius build tests/gpu_rope.cyr build/gpu_rope && ./build/gpu_rope

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
	@for args in "--layers 100000 --attn-every 2" "--layers 129 --attn-every 2" "--layers 0 --attn-every 2" "--ternary --experts 4" "--ternary --objective diffusion" "--ternary --attn-kind ssm" "--rl-target e" "--objective bogus" "--objective rl --experts 4"; do \
	  ./build/attn11 $$args --gen-only >/dev/null 2>&1; rc=$$?; \
	  if [ $$rc -ge 128 ]; then echo "smoke: CRASH (signal $$((rc-128))) on: $$args"; exit 1; fi; \
	  if [ $$rc -eq 0 ]; then echo "smoke: expected rejection but succeeded on: $$args"; exit 1; fi; \
	done; \
	./build/attn11 --preset --attn-kind ssm --attn-every 2 --gen-only >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid mha/ssm hybrid failed to build"; exit 1; fi; \
	./build/attn11 --ternary --steps 5 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid --ternary run failed"; exit 1; fi; \
	./build/attn11 --objective rl --steps 5 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid --objective rl run failed"; exit 1; fi; \
	./build/attn11 --objective rl --rl-target e --steps 5 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid --objective rl --rl-target run failed"; exit 1; fi; \
	./build/attn11 --gen-only --eval-corpus /nonexistent/attn11-smoke >/dev/null 2>&1; rc=$$?; \
	if [ $$rc -ge 128 ]; then echo "smoke: CRASH (signal $$((rc-128))) on missing --eval-corpus file"; exit 1; fi; \
	if [ $$rc -eq 0 ]; then echo "smoke: missing --eval-corpus file should set a non-zero exit"; exit 1; fi; \
	printf 'streaming smoke corpus -- the quick brown fox jumps. ' > /tmp/attn11-smoke-seed.txt; \
	for i in 1 2 3 4 5 6 7 8; do cat /tmp/attn11-smoke-seed.txt; done > /tmp/attn11-smoke-corpus.txt; \
	./build/attn11 --corpus /tmp/attn11-smoke-corpus.txt --encode-shard /tmp/attn11-smoke.tsh >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: --encode-shard failed on a valid corpus"; exit 1; fi; \
	./build/attn11 --stream-corpus /tmp/attn11-smoke.tsh --steps 5 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid --stream-corpus run failed"; exit 1; fi; \
	./build/attn11 --stream-corpus /tmp/attn11-smoke.tsh --steps 3 --save /tmp/attn11-smoke.ckpt >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: --stream-corpus + --save failed"; exit 1; fi; \
	./build/attn11 --stream-corpus /tmp/attn11-smoke.tsh --load /tmp/attn11-smoke.ckpt --steps 6 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: a valid resume-from-stream (--load + --stream-corpus) failed"; exit 1; fi; \
	./build/attn11 --corpus /tmp/attn11-smoke-corpus.txt --bpe 8 --encode-shard /tmp/attn11-smoke-bpe.tsh --stream-encode >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: --stream-encode (bpe) failed on a valid corpus"; exit 1; fi; \
	./build/attn11 --stream-corpus /tmp/attn11-smoke-bpe.tsh --steps 3 >/dev/null 2>&1; \
	if [ $$? -ne 0 ]; then echo "smoke: training from a --stream-encode bpe shard failed"; exit 1; fi; \
	for sa in "--stream-corpus /nonexistent/attn11-smoke.tsh" "--stream-corpus /tmp/attn11-smoke-corpus.txt" "--stream-corpus /tmp/attn11-smoke.tsh --corpus /tmp/attn11-smoke-corpus.txt" "--stream-corpus /tmp/attn11-smoke.tsh --bpe 8" "--stream-corpus /tmp/attn11-smoke-bpe.tsh --load /tmp/attn11-smoke.ckpt --steps 3" "--stream-encode --encode-shard /tmp/attn11-smoke-x.tsh" "--stream-encode --corpus /tmp/attn11-smoke-corpus.txt" "--stream-encode --stdin --corpus /tmp/attn11-smoke-corpus.txt --encode-shard /tmp/attn11-smoke-x.tsh"; do \
	  ./build/attn11 $$sa --gen-only >/dev/null 2>&1; rc=$$?; \
	  if [ $$rc -ge 128 ]; then echo "smoke: CRASH (signal $$((rc-128))) on: $$sa"; exit 1; fi; \
	  if [ $$rc -eq 0 ]; then echo "smoke: expected rejection but succeeded on: $$sa"; exit 1; fi; \
	done; \
	rm -f /tmp/attn11-smoke-seed.txt /tmp/attn11-smoke-corpus.txt /tmp/attn11-smoke.tsh /tmp/attn11-smoke-bpe.tsh /tmp/attn11-smoke.ckpt /tmp/attn11-smoke-x.tsh; \
	echo "smoke: ok (hostile args rejected; valid hybrid + ternary + stream + resume-from-stream + stream-encode; bad shard / vocab-mismatch / held-out eval error cleanly)"

clean:
	rm -rf build/
