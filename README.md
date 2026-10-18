# strategy

**Reusable decision strategies for FlossWare.**

This repository contains interchangeable strategy implementations. It is not a model router and does not define the Loom execution model.

## Boundary

```text
stable strategy contract
          |
          v
       strategy
          |
          v
decision + provenance + outcome
```

Strategies choose among feasible alternatives. They do not own authorization, policy, credentials, billing, quota enforcement, or safety constraints. Those constraints must be applied independently and cannot be overridden by a learned strategy.

## Current strategies

- Thompson Sampling
- epsilon-greedy
- deterministic baselines
- future UCB and contextual-bandit strategies

Thompson Sampling can support adaptive model or resource selection, but it remains a reusable strategy rather than a special Loom primitive.

## Relationship to Loom

Loom defines execution semantics and the contracts through which capabilities are invoked. This repository supplies strategy implementations that a Loom Worker, Arbiter, or other compatible caller may use. It does not own orchestration or provider invocation.

Optimization algorithms may be implemented in `optimization` and used by strategies where appropriate. Strategy selection and optimization algorithms are related but distinct responsibilities.
