"""Protocol definitions for strategy-ai (ADR-0020: Capability-Protocol Separation).

Defines the structural contracts that strategy selectors and reward
trackers must satisfy.  Uses ``typing.Protocol`` with
``@runtime_checkable`` so conformance can be verified at runtime
without inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from strategy_ai.types import StrategyStats


@runtime_checkable
class StrategySelector(Protocol):
    """Select the best strategy from a list of candidates."""

    async def select(self, task_type: str, *, candidates: list[str]) -> str:
        """Choose the best strategy for *task_type* from *candidates*."""
        ...


@runtime_checkable
class RewardTracker(Protocol):
    """Record reward observations for strategy/task-type pairs."""

    async def update(self, strategy: str, task_type: str, *, reward: float) -> None:
        """Record a reward observation for a strategy/task-type pair."""
        ...


@runtime_checkable
class PerformanceReporter(Protocol):
    """Report performance statistics for strategy arms."""

    async def performance(
        self, *, task_type: str | None = None
    ) -> dict[str, StrategyStats]:
        """Return performance statistics for all known arms."""
        ...
