# 0015 — REINFORCE policy-gradient training as reward-weighted softmax-CE

**Status**: Superseded — RL removed from attn11 and migrated to the tarka repo at 1.11.1 (see tarka ADR 0001)
**Date**: 2026-06-14

## Superseded (1.11.1)

This decision held while attn11 owned the reinforcement-learning objective. At
**1.11.1** RL was **removed from attn11** — the `--objective rl` / `--rl-target`
flags, the `rl_rollout`/`rl_reward`/`rl_train`/`rl_eval` path, the
`g_rl` / `g_reinforce_scale` gradient gate, and the M17 grad-check tests
(`test_rl_op` / `test_rl_rollout`) are all gone. RL migrated to a separate repo,
**tarka** (https://github.com/MacCracken/tarka), the sovereign RL/reasoning
counterpoint, where REINFORCE is re-expressed on the `rosnet` tensor lib. attn11
is now a pure supervised (SFT) + masked-diffusion training reference. The decision
record below is preserved verbatim as the original rationale; see tarka ADR 0001
for the continuation.

## Context

M17 (E9) is the last milestone in the chain by design: reinforcement learning is a
*training-objective layer*, not an architecture. It runs on whatever trunk exists
(any `--attn-kind` / mixer / precision) and fine-tunes a policy toward a reward
rather than toward a corpus. The reference question is the gate: **does the policy
move toward the reward?** — demonstrated, not assumed.

The 1.x arc's constraints apply: opt-in and **additive** (the no-flag run stays
byte-identical, every older checkpoint still loads), every new hand-derived
backward passes a finite-difference grad-check, and the result is reported
honestly. The opportunity that shapes the design: for a softmax policy the
policy-gradient term is *already* the supervised backward. With
`pi(a) = softmax(logits)[a]`, `grad log pi(a) = -grad CE(a)` where `CE(a)` is the
cross-entropy of the sampled action `a`. So REINFORCE's gradient
`(R - b) · grad log pi(a)` is the **existing softmax-CE backward over the sampled
rollout, scaled by the advantage `(R - b)`** — no new gradient math, exactly
attn11's wheelhouse.

## Decision

Add `--objective rl` (default off): on-policy **REINFORCE** (Williams 1992) at char
scale. The model is a **plain AR transformer** — RL changes only the training loss,
not the forward, the architecture, or the checkpoint.

- **Rollout** (`rl_rollout`): from a random corpus token, autoregressively sample
  `g_T` actions **at temperature 1** (the policy's own distribution, so the gradient
  differentiates the same distribution it sampled from — on-policy). Lay the rollout
  out as ordinary AR training data (`A_tokens[t] = x_t`, `A_targets[t] = x_{t+1}`),
  so softmax-CE over it is exactly `-log pi` of the rollout.
- **Reward** (`rl_reward`): a deterministic in-process scalar — the count of a target
  token (`--rl-target C`) in the rollout. Simple, monotone, demonstrable; not a
  learned reward model at this scale.
- **Advantage + baseline**: `(R - b)`, with `b` an EMA of past mean rewards
  (variance reduction, no learned value network). Updated after each step so a
  step's advantages use the average of *prior* rewards.
- **The gradient injection** (`g_rl` / `g_reinforce_scale`): `model_backward` scales
  the seeded logit gradient `D_logits` by `(R - b)` before backprop. The rest of the
  backward is linear in `D_logits`, so this scales the whole gradient to
  `(R - b) · grad CE` = the policy-gradient term. The scale is **gated on `g_rl`**,
  so AR and diffusion training never touch it and stay byte-identical.
- **No checkpoint bump**: the RL-trained model is structurally AR (`g_objective = 0`,
  causal, no mask_emb), so it serializes as a normal **v5** checkpoint. `g_rl` is a
  runtime training flag (like `g_training`), never serialized.
- **Composability**: because the forward is unchanged, RL composes with the mixer /
  position / precision axes (mha/mla/lin/ssm/hybrid, RoPE, ternary) — no bidirectional
  or learned-abs restriction like diffusion. The one exception is **MoE** (`--experts
  > 1`): its load-balance aux loss seeds a gradient *outside* `D_logits`, which the
  `(R - b)` scale would miss, so the RL gradient would be wrong. RL therefore requires a
  **dense MLP** (a CLI guard + a post-init `g_num_experts` backstop reject the
  combination); scaling-or-dropping the aux under RL is a documented fast-follow.

### Alternatives considered

- **A new backward op / autodiff for the policy gradient** — rejected: unnecessary.
  The reward-weighted-CE identity makes RL a one-scalar delta over the supervised
  backward, which the FD grad-check covers directly (ADR 0001 keeps gradients
  hand-derived).
- **Sampling at temperature ≠ 1 for exploration** — rejected for the reference: it
  makes the sampling distribution differ from the one the CE gradient differentiates
  (off-policy), needing an importance-weight correction. Temperature 1 is on-policy
  and exact; the model's own stochasticity is the exploration.
- **A new checkpoint version recording "RL-trained"** — rejected: the model *is* AR;
  provenance lives in logs/CHANGELOG, not the format. Keeps RL images byte-compatible
  with AR and avoids format churn.
- **PPO / GRPO (clipped ratio, group baselines)** — deferred: a heavier follow-on,
  warranted only if REINFORCE earns it. The EMA baseline is the minimal variance
  reduction that makes char-scale REINFORCE converge.

## Consequences

- The policy moves decisively toward the reward (X024: target-char frequency
  9–19% → ~99.7% across common/rare/space targets) — the M17 gate, met. The naive
  count reward also exhibits the **SFT→RL alignment tax** (corpus bits/byte
  0.24 → ~13): RL optimizes the reward it is given, not language modeling. An honest
  caveat, and the reason richer rewards (valid text, length/format targets) and
  PPO/GRPO are the noted follow-on.
- Grad-checked three ways (`test_rl_op`): RL grad == advantage × AR grad (to
  rounding), FD vs the numeric gradient of `advantage × CE`, and the sign-flip /
  zero-advantage limits. The rollout layout is pinned by `test_rl_rollout`.
- The no-flag run, every checkpoint, and all prior grad-checks are byte-identical
  (the `g_rl` gate). Source: Williams 1992 (`docs/sources.md`).
