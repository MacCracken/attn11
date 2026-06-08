# Security Policy

## Reporting

Report vulnerabilities to **cyriusmaccken@gmail.com**. Include reproduction
steps and the attn11 version from `VERSION`. Expect an initial response within
one week. Coordinated disclosure is appreciated — do not open a public GitHub
issue with exploit details.

## Threat model

attn11 is a single-process, CPU-only transformer trainer. It has **no
networking**, and by default consumes only an **embedded, compile-time corpus**
— there is no untrusted input on the default path. A realistic attacker is
assumed able to:

- supply the build inputs (source / corpus) — i.e. they already control what
  gets compiled, which is equivalent to code execution, and/or
- run the resulting binary with attacker-chosen hyperparameters.

attn11 *does not* defend against:

- an attacker with arbitrary code execution in the build or host process
  (trivially owns the whole address space),
- remote / network attacks (attn11 has no networking),
- side channels (timing, cache),
- adversarial training data once a future release adds file/stdin corpus
  loading — see below.

## Attack surfaces & mitigations

| Surface | Mitigation |
|---|---|
| **Embedded corpus + tokenizer** | Corpus is a compile-time constant; the char→id map is a fixed 256-entry table with bounds-correct indexing. No untrusted input on the default path. |
| **Numeric / memory layout** | All buffers (parameters, optimizer state, activation caches, attention arena) are allocated once at fixed, dimension-derived sizes; no allocation or unbounded growth in the training loop. Indices are computed from validated dims. |
| **PRNG** | Deterministic xorshift64 with splitmix64 seeding — used for weight init and sampling only, **not** for any security purpose. |

## Future surfaces (not yet present)

When file/stdin corpus loading or checkpoint load/save lands (roadmap M2), the
loaders become a real surface: validate file size and contents, open with
`O_NOFOLLOW`, and bounds-check every read before trusting it. Update this file
in the same PR.
