"""Data models for strategy-ai.

All models are plain dataclasses with no imports outside the standard
library.  Strategy protocols reference these types for their method
signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BanditArm:
    """Mutable state for one (strategy, task_type) bandit arm."""

    alpha: float = 1.0
    beta: float = 1.0
    total_trials: int = 0
    successes: int = 0
    total_reward: float = 0.0


@dataclass
class StrategyStats:
    """Thompson-Sampling bandit state for a strategy/task-type pair."""

    strategy: str
    task_type: str
    total_trials: int
    successes: int
    avg_reward: float
    alpha: float
    beta: float


@dataclass
class SelectionResult:
    """Outcome of a strategy selection decision."""

    selected: str
    candidates: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    method: str = ""
