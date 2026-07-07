# cyrius: `f64_sin` / `f64_cos` have no aarch64 polyfill — trig-consuming code is x86-only

**Filed:** 2026-07-07 (by an attn11 consumer — the 1.14.0 hearing lane)
**Toolchain:** cycc 6.4.14 (also reproduces at the 6.2.29 CI pin)
**Severity:** language/stdlib gap — arch parity. `f64_exp` / `f64_ln` / `f64_log2`
got aarch64 polyfills at v5.7.31 (stdlib `math.cyr` `_f64_ln_polyfill` etc., with
`parse_expr` auto-dispatch); **sin/cos never did**. Any aarch64 build whose
amalgamation *contains* a `f64_sin`/`f64_cos` call fails hard:

```
error: f64_sin is x86-only for v5.6.0; aarch64 has no native trig — needs polyfi
compile tests/attn11.tcyr -> build/test_a64 [aarch64] FAIL
```

## Trigger

attn11 1.14.0's hearing lane (`src/hearing.cyr`): the Hann window (`f64_cos`)
and the synthetic-audio synth (`f64_sin`), plus **hisab's `num_fft`** (twiddle
factors — `dist/hisab.cyr` calls both, unguarded), newly pulled in as
`[deps.hisab]`. Dep modules auto-prepend into every matching target's
amalgamation, so hisab's presence alone broke the aarch64 CI leg even with the
consumer's own calls `#ifndef`-gated.

## Consumer-side workaround (shipped in attn11 1.14.0)

- `[deps.hisab] target = "x86_64"` (the v6.3.1 dep target key — this forced the
  attn11 pin 6.2.29 → 6.4.14, since the CI installs the pin and older cbt
  ignores the key) so the dep never prepends on aarch64;
- `#ifndef CYRIUS_ARCH_AARCH64` around the manual includes + CLI flags + suite
  registrations — **the hearing lane is x86-only until this gap closes**.

## Affects (beyond attn11)

hisab itself (its dist cannot compile aarch64), shravan/naad-class DSP, any
future audio/graphics consumer on the Pi-ARM line (seema / the 1.6x aarch64
kernel line). The hearing lane on aarch64 is exactly the tok/s-on-Pi story the
ecosystem wants eventually.

## Ask

The v5.7.31 pattern, applied to trig: stdlib `_f64_sin_polyfill` /
`_f64_cos_polyfill` (range-reduce + polynomial, the `_f64_ln_polyfill`
precision bar) + `parse_expr` auto-dispatch on aarch64. NEON-native trig is not
needed — the polyfill unblocks; Phase-5 SIMD trig can come whenever.

## Unblock signal

When the polyfill lands: drop `target = "x86_64"` from attn11's `[deps.hisab]`,
remove the three `#ifndef CYRIUS_ARCH_AARCH64` gates (src/main.cyr ×3,
tests/attn11.tcyr ×3), and the hearing suite group runs under qemu-aarch64 like
the rest.

---
*Mirrored cyrius-side at
`cyrius/docs/development/issues/2026-07-07-aarch64-no-trig-polyfill.md`.*
