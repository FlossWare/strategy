"""Tests for ThompsonSamplingSelector."""

from __future__ import annotations

import asyncio

import pytest

from strategy_ai import ThompsonSamplingSelector
from strategy_ai.protocols import PerformanceReporter, RewardTracker, StrategySelector
from strategy_ai.types import BanditArm, StrategyStats


class TestThompsonSamplingSelector:
    """Tests for ThompsonSamplingSelector."""

    def test_protocol_conformance(self):
        selector = ThompsonSamplingSelector()
        assert isinstance(selector, StrategySelector)
        assert isinstance(selector, RewardTracker)
        assert isinstance(selector, PerformanceReporter)

    @pytest.mark.asyncio
    async def test_select_returns_candidate(self):
        selector = ThompsonSamplingSelector(seed=42)
        candidates = ["strategy_a", "strategy_b", "strategy_c"]
        result = await selector.select("code", candidates=candidates)
        assert result in candidates

    @pytest.mark.asyncio
    async def test_select_empty_candidates_raises(self):
        selector = ThompsonSamplingSelector()
        with pytest.raises(ValueError, match="candidates must not be empty"):
            await selector.select("code", candidates=[])

    @pytest.mark.asyncio
    async def test_select_single_candidate(self):
        selector = ThompsonSamplingSelector(seed=42)
        result = await selector.select("code", candidates=["only_one"])
        assert result == "only_one"

    @pytest.mark.asyncio
    async def test_update_success(self):
        selector = ThompsonSamplingSelector()
        await selector.update("strat_a", "code", reward=0.9)
        arm = selector._get_arm("strat_a", "code")
        assert arm.total_trials == 1
        assert arm.successes == 1
        assert arm.alpha == 2.0
        assert arm.beta == 1.0
        assert arm.total_reward == 0.9

    @pytest.mark.asyncio
    async def test_update_failure(self):
        selector = ThompsonSamplingSelector()
        await selector.update("strat_a", "code", reward=0.3)
        arm = selector._get_arm("strat_a", "code")
        assert arm.total_trials == 1
        assert arm.successes == 0
        assert arm.alpha == 1.0
        assert arm.beta == 2.0
        assert arm.total_reward == 0.3

    @pytest.mark.asyncio
    async def test_update_boundary_reward(self):
        selector = ThompsonSamplingSelector()
        await selector.update("strat_a", "code", reward=0.5)
        arm = selector._get_arm("strat_a", "code")
        assert arm.successes == 0
        assert arm.beta == 2.0

    @pytest.mark.asyncio
    async def test_performance_all(self):
        selector = ThompsonSamplingSelector()
        await selector.update("a", "code", reward=0.9)
        await selector.update("b", "code", reward=0.3)
        await selector.update("a", "docs", reward=0.8)

        stats = await selector.performance()
        assert len(stats) == 3
        assert "a:code" in stats
        assert "b:code" in stats
        assert "a:docs" in stats
        assert stats["a:code"].avg_reward == 0.9
        assert stats["b:code"].avg_reward == 0.3

    @pytest.mark.asyncio
    async def test_performance_filtered(self):
        selector = ThompsonSamplingSelector()
        await selector.update("a", "code", reward=0.9)
        await selector.update("b", "code", reward=0.3)
        await selector.update("a", "docs", reward=0.8)

        stats = await selector.performance(task_type="code")
        assert len(stats) == 2
        assert "a:code" in stats
        assert "b:code" in stats
        assert "a:docs" not in stats

    @pytest.mark.asyncio
    async def test_performance_empty(self):
        selector = ThompsonSamplingSelector()
        stats = await selector.performance()
        assert stats == {}

    @pytest.mark.asyncio
    async def test_exploitation_bias(self):
        selector = ThompsonSamplingSelector(seed=42)
        for _ in range(50):
            await selector.update("good", "code", reward=0.95)
        for _ in range(50):
            await selector.update("bad", "code", reward=0.1)

        selections = []
        for _ in range(100):
            s = await selector.select("code", candidates=["good", "bad"])
            selections.append(s)

        good_count = selections.count("good")
        assert good_count > 80

    @pytest.mark.asyncio
    async def test_independent_task_types(self):
        selector = ThompsonSamplingSelector(seed=42)
        for _ in range(20):
            await selector.update("strat_a", "code", reward=0.9)
            await selector.update("strat_b", "code", reward=0.1)
            await selector.update("strat_a", "docs", reward=0.1)
            await selector.update("strat_b", "docs", reward=0.9)

        code_selections = [
            await selector.select("code", candidates=["strat_a", "strat_b"])
            for _ in range(50)
        ]
        docs_selections = [
            await selector.select("docs", candidates=["strat_a", "strat_b"])
            for _ in range(50)
        ]

        assert code_selections.count("strat_a") > 30
        assert docs_selections.count("strat_b") > 30

    def test_reset(self):
        selector = ThompsonSamplingSelector()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(selector.update("a", "code", reward=0.9))
        loop.run_until_complete(selector.reset())
        stats = loop.run_until_complete(selector.performance())
        loop.close()
        assert stats == {}

    def test_seed_reproducibility(self):
        loop = asyncio.new_event_loop()
        s1 = ThompsonSamplingSelector(seed=123)
        s2 = ThompsonSamplingSelector(seed=123)
        candidates = ["a", "b", "c", "d"]

        r1 = [loop.run_until_complete(s1.select("t", candidates=candidates)) for _ in range(20)]
        r2 = [loop.run_until_complete(s2.select("t", candidates=candidates)) for _ in range(20)]
        loop.close()
        assert r1 == r2

    def test_bandit_arm_defaults(self):
        arm = BanditArm()
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.total_trials == 0
        assert arm.successes == 0
        assert arm.total_reward == 0.0

    def test_strategy_stats_fields(self):
        stats = StrategyStats(
            strategy="x", task_type="y",
            total_trials=10, successes=7,
            avg_reward=0.7, alpha=8.0, beta=4.0,
        )
        assert stats.strategy == "x"
        assert stats.task_type == "y"
        assert stats.total_trials == 10
