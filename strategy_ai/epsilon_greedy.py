"""Epsilon-greedy strategy selector for strategy-ai.

A simpler alternative to Thompson Sampling: with probability epsilon,
explore a random candidate; otherwise exploit the candidate with the
highest observed average reward.  All data is held in memory -- no
external dependencies required.

Classes
-------
EpsilonGreedySelector -- in-memory epsilon-greedy bandit selector
"""

from __future__ import annotations

import asyncio
import random

from strategy_ai.types import BanditArm, StrategyStats


class EpsilonGreedySelector:
    """In-memory epsilon-greedy strategy selector.

    Satisfies :class:`~strategy_ai.protocols.StrategySelector`,
    :class:`~strategy_ai.protocols.RewardTracker`, and
    :class:`~strategy_ai.protocols.PerformanceReporter` via structural
    subtyping.

    Parameters
    ----------
    epsilon:
        Probability of exploring a random candidate (0.0 to 1.0).
        Default is 0.1 (10% exploration).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, *, epsilon: float = 0.1, seed: int | None = None) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be between 0.0 and 1.0, got {epsilon}")
        self._epsilon = epsilon
        self._rng = random.Random(seed)  # noqa: S311
        self._arms: dict[tuple[str, str], BanditArm] = {}
        self._lock = asyncio.Lock()

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    def _get_arm(self, strategy: str, task_type: str) -> BanditArm:
        """Return the arm for *(strategy, task_type)*, creating if needed."""
        key = (strategy, task_type)
        if key not in self._arms:
            self._arms[key] = BanditArm()
        return self._arms[key]

    def _avg_reward(self, strategy: str, task_type: str) -> float:
        """Return the average reward for a strategy/task_type pair."""
        arm = self._get_arm(strategy, task_type)
        if arm.total_trials == 0:
            return 0.0
        return arm.total_reward / arm.total_trials

    async def select(self, task_type: str, *, candidates: list[str]) -> str:
        """Choose a strategy for *task_type* from *candidates*.

        With probability *epsilon*, return a random candidate (explore).
        Otherwise, return the candidate with the highest average reward
        (exploit).  Ties are broken by candidate order.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")

        async with self._lock:
            if self._rng.random() < self._epsilon:
                return self._rng.choice(candidates)

            best_strategy = candidates[0]
            best_reward = self._avg_reward(candidates[0], task_type)

            for strategy in candidates[1:]:
                avg = self._avg_reward(strategy, task_type)
                if avg > best_reward:
                    best_reward = avg
                    best_strategy = strategy

        return best_strategy

    async def update(self, strategy: str, task_type: str, *, reward: float) -> None:
        """Record a reward observation for a strategy/task-type pair."""
        async with self._lock:
            arm = self._get_arm(strategy, task_type)
            arm.total_trials += 1
            arm.total_reward += reward

            if reward > 0.5:
                arm.alpha += 1.0
                arm.successes += 1
            else:
                arm.beta += 1.0

    async def performance(
        self, *, task_type: str | None = None
    ) -> dict[str, StrategyStats]:
        """Return performance statistics for all known arms."""
        async with self._lock:
            result: dict[str, StrategyStats] = {}

            for (strategy, tt), arm in self._arms.items():
                if task_type is not None and tt != task_type:
                    continue

                avg_reward = (
                    arm.total_reward / arm.total_trials if arm.total_trials > 0 else 0.0
                )
                key = f"{strategy}:{tt}"
                result[key] = StrategyStats(
                    strategy=strategy,
                    task_type=tt,
                    total_trials=arm.total_trials,
                    successes=arm.successes,
                    avg_reward=avg_reward,
                    alpha=arm.alpha,
                    beta=arm.beta,
                )

            return result

    async def reset(self) -> None:
        """Clear all bandit state."""
        async with self._lock:
            self._arms.clear()
