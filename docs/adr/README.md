# Architecture Decision Records

Decisions about attn11 — what we chose, the context, and the consequences we accept. Use these when a future reader would reasonably ask *"why did we do it this way?"*

## Conventions

- **Filename**: `NNNN-kebab-case-title.md`, zero-padded to four digits. Never renumber.
- **One decision per ADR.** If a decision supersedes a prior one, add a new ADR and set the old one's status to `Superseded by NNNN`.
- **Status lifecycle**: `Proposed` → `Accepted` → (optionally) `Superseded` or `Deprecated`.
- Use [`template.md`](template.md) as the starting point.

## ADR vs. architecture note vs. guide

| Kind | Lives in | Answers |
|---|---|---|
| ADR | `docs/adr/` | *Why did we choose X over Y?* |
| Architecture note | `docs/architecture/` | *What non-obvious constraint is true about the code?* |
| Guide | `docs/guides/` | *How do I do X?* |

## Index

- [0001 — Hand-derived backprop over an autodiff engine](0001-hand-derived-backprop-over-autodiff.md) — each backward op is hand-written and finite-difference grad-checked; no autodiff graph.
- [0002 — Byte-level tokenizer over BPE](0002-byte-level-tokenizer-over-bpe.md) — adaptive ≤256 byte vocab; BPE deferred.
- [0003 — SIMD via memory-accumulator f64v_fmadd](0003-simd-memory-accumulators.md) — matmul vectorized with memory accumulators (never reassign a SIMD-typed var).
- [0004 — Native-endian raw checkpoint blob](0004-native-endian-checkpoint-format.md) — fast same-arch resume; not cross-architecture portable.
- [0005 — Context-shift generation under absolute positional embeddings](0005-context-shift-generation-under-absolute-positions.md) — KV-cached decode with drop-oldest-T/2 + re-prime when the window fills; cached path gated bit-identical to an uncached reference.
- [0006 — Opt-in BPE tokenizer, byte-level default](0006-opt-in-bpe-tokenizer.md) — `--bpe K` learns ≤512 merges over the byte base vocab; frozen deterministic tie-break; checkpoints (v3) carry the tokenizer; amends 0002.
- [0007 — Multi-head latent attention and a positional-encoding switch](0007-mla-and-positional-encoding-switch.md) — `--attn-kind {mha,gqa,mla}` + `--pos-kind {learned,rope,rope-decoupled}` (orthogonal, default-preserving); MLA keeps learned-abs positions for the grad-checkable core; checkpoint v4 reserves the architecture descriptor ahead of the math. **Fully shipped (M12 complete):** MLA core (1.2.0) + latent KV-cache decode (1.2.1) + coupled `rope` (1.2.2) + decoupled `rope-decoupled` (1.2.3) — the whole `--attn-kind` × `--pos-kind` switch.
- [0008 — Mixture-of-Experts router: combine, balance, and the dense invariant](0008-moe-router-design.md) — `--experts N --expert-topk K`: Mixtral-style renormalized top-K softmax combine (gradient only to the selected logits, straight-through pick), Switch-style load-balance aux loss (dispatch counts held constant), frozen lower-index tie-break, and `--experts 1` is the byte-identical dense MLP. Checkpoint v5. **Shipped (M13, 1.3.0).**
- [0009 — Gated linear attention as a sequence-mixer family](0009-gated-linear-attention-mixer.md) — `--attn-kind lin`: RetNet-style retention recurrence (fixed per-head decay, parameter-free), reusing the MHA projections so it rides the `attn_kind` slot (value 2) with no checkpoint bump; the decode cache is the constant `nh·hd²` state, not a growing K/V. Hand-derived backward with no state caching. **Shipped (M14 rung a, 1.4.0).**
- [0010 — Selective SSM as the third sequence-mixer family](0010-selective-ssm-mixer.md) — `--attn-kind ssm`: a minimal Mamba-lite diagonal SSM (input-dependent Δ/B/C, `exp(Δ·A)`, learned diagonal A), `attn_kind = 3` reusing Wq (W_dt) + Wo (output proj) + `latent_dim` (state size N) — no checkpoint bump. Hand-derived BPTT through the data-dependent scan; constant `C·N` decode cache. **Shipped (M14 rung b, 1.4.2).**
