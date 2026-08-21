"""strategy-ai -- Thompson Sampling and epsilon-greedy bandit-based strategy selection.

Public API
----------
Types:
    BanditArm, StrategyStats, SelectionResult

Protocols:
    StrategySelector, RewardTracker, PerformanceReporter

Selectors:
    ThompsonSamplingSelector, EpsilonGreedySelector

Decorators (ADR-0006):
    with_thompson_sampling, with_epsilon_greedy
"""

from __future__ import annotations

from strategy_ai.decorators import with_epsilon_greedy, with_thompson_sampling
from strategy_ai.epsilon_greedy import EpsilonGreedySelector
from strategy_ai.protocols import PerformanceReporter, RewardTracker, StrategySelector
from strategy_ai.strategy import ThompsonSamplingSelector
from strategy_ai.types import BanditArm, SelectionResult, StrategyStats

__all__ = [
    "BanditArm",
    "EpsilonGreedySelector",
    "PerformanceReporter",
    "RewardTracker",
    "SelectionResult",
    "StrategySelector",
    "StrategyStats",
    "ThompsonSamplingSelector",
    "with_epsilon_greedy",
    "with_thompson_sampling",
]

__version__ = "0.1"
