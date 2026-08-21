#!/usr/bin/env python3
"""Verify strategy-ai installation and run a quick smoke test."""
import sys


def main():
    try:
        from strategy_ai import (
            BanditArm,
            EpsilonGreedySelector,
            PerformanceReporter,
            RewardTracker,
            SelectionResult,
            StrategySelector,
            StrategyStats,
            ThompsonSamplingSelector,
            with_epsilon_greedy,
            with_thompson_sampling,
        )
    except ImportError as e:
        print(f"FAIL: Could not import strategy_ai: {e}")
        print("Install: pip install 'git+https://github.com/FlossWare/strategy-ai.git'")
        sys.exit(1)

    import strategy_ai

    print(f"strategy-ai v{strategy_ai.__version__} installed successfully")
    print(f"Exports: {len(strategy_ai.__all__)} public symbols")

    # Smoke test: Thompson Sampling
    ts = ThompsonSamplingSelector(seed=42)
    print(f"Smoke test: ThompsonSamplingSelector created: {ts}")

    # Smoke test: Epsilon-greedy
    eg = EpsilonGreedySelector(epsilon=0.1, seed=42)
    print(f"Smoke test: EpsilonGreedySelector(epsilon=0.1) created: {eg}")

    # Smoke test: Protocol conformance
    assert isinstance(ts, StrategySelector), "ThompsonSamplingSelector must be StrategySelector"
    assert isinstance(ts, RewardTracker), "ThompsonSamplingSelector must be RewardTracker"
    assert isinstance(eg, StrategySelector), "EpsilonGreedySelector must be StrategySelector"
    print("Smoke test: protocol conformance OK")

    # Smoke test: decorators are callable
    assert callable(with_thompson_sampling), "with_thompson_sampling must be callable"
    assert callable(with_epsilon_greedy), "with_epsilon_greedy must be callable"
    print("Smoke test: decorators are callable")

    # Smoke test: types
    arm = BanditArm()
    assert arm.alpha == 1.0
    stats = StrategyStats(
        strategy="x", task_type="y", total_trials=0,
        successes=0, avg_reward=0.0, alpha=1.0, beta=1.0,
    )
    assert stats.strategy == "x"
    result = SelectionResult(selected="x")
    assert result.selected == "x"
    print("Smoke test: data types OK")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
