#!/usr/bin/env python3
"""Basic strategy-ai usage: Thompson Sampling and epsilon-greedy model selection."""
from __future__ import annotations

import asyncio

from strategy_ai import EpsilonGreedySelector, ThompsonSamplingSelector


async def main():
    # 1. Thompson Sampling
    ts = ThompsonSamplingSelector(seed=42)
    models = ["gpt-4o", "claude-sonnet", "gemini-flash", "llama-70b"]

    print("Thompson Sampling - Training phase:")
    rewards = {"gpt-4o": 0.85, "claude-sonnet": 0.92, "gemini-flash": 0.70, "llama-70b": 0.60}
    for i in range(30):
        selected = await ts.select("code_gen", candidates=models)
        reward = rewards[selected]
        await ts.update(selected, "code_gen", reward=reward)
        if (i + 1) % 10 == 0:
            print(f"  Round {i+1}: selected {selected} (reward={reward:.2f})")

    # Show learned stats
    print("\nLearned performance:")
    stats = await ts.performance(task_type="code_gen")
    for key, s in sorted(stats.items(), key=lambda x: x[1].avg_reward, reverse=True):
        print(f"  {s.strategy}: {s.total_trials} trials, avg_reward={s.avg_reward:.2f}")

    # 2. Epsilon-greedy
    eg = EpsilonGreedySelector(epsilon=0.1, seed=42)
    print("\nEpsilon-greedy (epsilon=0.1):")
    for i in range(10):
        selected = await eg.select("code_gen", candidates=models)
        reward = rewards[selected]
        await eg.update(selected, "code_gen", reward=reward)
        print(f"  Round {i+1}: selected {selected} (reward={reward:.2f})")

    # 3. Compare selections after training
    print("\nPost-training selections (100 rounds):")
    ts_selections = [await ts.select("code_gen", candidates=models) for _ in range(100)]
    eg_selections = [await eg.select("code_gen", candidates=models) for _ in range(100)]

    for model in models:
        ts_pct = ts_selections.count(model)
        eg_pct = eg_selections.count(model)
        print(f"  {model}: TS={ts_pct}%, EG={eg_pct}%")


if __name__ == "__main__":
    asyncio.run(main())
