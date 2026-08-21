# FlossWare Engineering Standards Compliance

This package adheres to the following ADRs from [FlossWare/engineering-standards](https://github.com/FlossWare/engineering-standards):

## ADR-0001: Explicit Opt-In

Strategy selection never activates automatically. All capabilities require explicit instantiation or decorator application by the developer.

- `ThompsonSamplingSelector` must be instantiated and `select()` called explicitly.
- `EpsilonGreedySelector` must be instantiated with an explicit epsilon parameter.
- `@with_thompson_sampling` and `@with_epsilon_greedy` decorators are opt-in.
- When no candidates are provided, decorators pass through to the wrapped function unchanged.

## ADR-0006: Cross-Cutting Decorators

Convenience decorators in `strategy_ai.decorators`:

- `@with_thompson_sampling(task_type="code")` -- wraps an async function with Thompson Sampling selection.
- `@with_epsilon_greedy(epsilon=0.1)` -- wraps an async function with epsilon-greedy selection.

## ADR-0008: Free-First

Zero external dependencies at runtime. The package uses only the Python standard library (`random`, `dataclasses`, `functools`, `typing`).

Development dependencies (pytest, pytest-asyncio) are optional.

## ADR-0009: Core Principles

- **Modular**: Each strategy (Thompson Sampling, epsilon-greedy) is a separate module.
- **Composable**: Selectors can be used independently or combined with other strategy-ai packages.
- **Contracts over implementations**: The `StrategySelector`, `RewardTracker`, and `PerformanceReporter` Protocols define the interfaces; any conforming object works.

## ADR-0013: Bandit-Based Model Selection

This package directly implements ADR-0013:

- Thompson Sampling with Beta-Bernoulli conjugate priors for exploration-exploitation.
- Per-(strategy, task_type) bandit arms with independent distributions.
- Reward feedback loop: `update()` adjusts arm parameters based on observed outcomes.
- Performance reporting via `performance()` for monitoring and debugging.
- Epsilon-greedy as a simpler alternative for cases where Thompson Sampling is overkill.

## ADR-0017: Agent-Neutral

The package works with any agent runtime. The `StrategySelector` Protocol is the only integration point -- any agent framework that can call `select()` and `update()` is compatible.

No assumptions are made about the calling agent's architecture, event loop, or lifecycle.

## ADR-0020: Capability-Protocol Separation

Strategy selection capabilities are transport-independent:

- `StrategySelector` Protocol defines what is needed (async select), not how it is delivered.
- No HTTP, gRPC, or other transport assumptions baked in.
- The same selection logic works whether used in a local test, an API server, or an agent runtime.
