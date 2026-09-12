# strategy-ai

**Reusable decision and optimization strategies for FlossWare.**

This repository contains interchangeable strategy implementations. It is not a model router and does not define the Loom execution model.

## Boundary

```text
Stable contract in Loom/capability
            │
            ▼
     strategy implementation
            │
            ▼
 decision + provenance + outcome
```

Strategies choose among feasible alternatives. They do not own authorization, policy, credentials, billing, quota enforcement, or safety constraints.

## Current strategies

- Thompson Sampling
- epsilon-greedy
- deterministic baselines
- future UCB/contextual-bandit strategies
- future optimization strategies where appropriate

Thompson Sampling is useful for adaptive model/resource selection, but the implementation remains a reusable strategy rather than a special Loom primitive.

## Learning boundary

Strategy state records decisions, rewards, and performance needed by the strategy itself. Durable engineering knowledge belongs in `knowledge`; authoritative task evaluation belongs in `evaluation`; model invocation belongs in `model-gateway`.

## Migration

The former documentation framed this package specifically around LLM model routing. That is now too narrow. Routing is one consumer of these strategies, alongside Worker selection, resource selection, evaluation policies, and other optimization problems.

## License

MIT
