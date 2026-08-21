"""Tests for EpsilonGreedySelector."""

from __future__ import annotations

import asyncio

import pytest

from strategy_ai import EpsilonGreedySelector
from strategy_ai.protocols import PerformanceReporter, RewardTracker, StrategySelector


class TestEpsilonGreedySelector:
    """Tests for EpsilonGreedySelector."""

    def test_protocol_conformance(self):
        selector = EpsilonGreedySelector()
        assert isinstance(selector, StrategySelector)
        assert isinstance(selector, RewardTracker)
        assert isinstance(selector, PerformanceReporter)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be between"):
            EpsilonGreedySelector(epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon must be between"):
            EpsilonGreedySelector(epsilon=1.5)

    def test_epsilon_property(self):
        selector = EpsilonGreedySelector(epsilon=0.3)
        assert selector.epsilon == 0.3

    @pytest.mark.asyncio
    async def test_select_returns_candidate(self):
        selector = EpsilonGreedySelector(seed=42)
        candidates = ["a", "b", "c"]
        result = await selector.select("code", candidates=candidates)
        assert result in candidates

    @pytest.mark.asyncio
    async def test_select_empty_candidates_raises(self):
        selector = EpsilonGreedySelector()
        with pytest.raises(ValueError, match="candidates must not be empty"):
            await selector.select("code", candidates=[])

    @pytest.mark.asyncio
    async def test_select_single_candidate(self):
        selector = EpsilonGreedySelector(seed=42)
        result = await selector.select("code", candidates=["only_one"])
        assert result == "only_one"

    @pytest.mark.asyncio
    async def test_exploitation_with_zero_epsilon(self):
        selector = EpsilonGreedySelector(epsilon=0.0, seed=42)
        for _ in range(10):
            await selector.update("good", "code", reward=0.9)
        for _ in range(10):
            await selector.update("bad", "code", reward=0.1)

        selections = [
            await selector.select("code", candidates=["good", "bad"])
            for _ in range(20)
        ]
        assert all(s == "good" for s in selections)

    @pytest.mark.asyncio
    async def test_exploration_with_full_epsilon(self):
        selector = EpsilonGreedySelector(epsilon=1.0, seed=42)
        for _ in range(10):
            await selector.update("good", "code", reward=0.9)
        for _ in range(10):
            await selector.update("bad", "code", reward=0.1)

        selections = [
            await selector.select("code", candidates=["good", "bad"])
            for _ in range(100)
        ]
        good_count = selections.count("good")
        assert 30 < good_count < 70

    @pytest.mark.asyncio
    async def test_update_success(self):
        selector = EpsilonGreedySelector()
        await selector.update("a", "code", reward=0.9)
        arm = selector._get_arm("a", "code")
        assert arm.total_trials == 1
        assert arm.successes == 1
        assert arm.total_reward == 0.9

    @pytest.mark.asyncio
    async def test_update_failure(self):
        selector = EpsilonGreedySelector()
        await selector.update("a", "code", reward=0.2)
        arm = selector._get_arm("a", "code")
        assert arm.total_trials == 1
        assert arm.successes == 0
        assert arm.total_reward == 0.2

    @pytest.mark.asyncio
    async def test_performance(self):
        selector = EpsilonGreedySelector()
        await selector.update("a", "code", reward=0.9)
        await selector.update("b", "code", reward=0.3)

        stats = await selector.performance()
        assert len(stats) == 2
        assert "a:code" in stats
        assert "b:code" in stats

    @pytest.mark.asyncio
    async def test_performance_filtered(self):
        selector = EpsilonGreedySelector()
        await selector.update("a", "code", reward=0.9)
        await selector.update("a", "docs", reward=0.5)

        stats = await selector.performance(task_type="code")
        assert len(stats) == 1
        assert "a:code" in stats

    def test_reset(self):
        selector = EpsilonGreedySelector()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(selector.update("a", "code", reward=0.9))
        loop.run_until_complete(selector.reset())
        stats = loop.run_until_complete(selector.performance())
        loop.close()
        assert stats == {}

    @pytest.mark.asyncio
    async def test_new_arms_have_zero_reward(self):
        selector = EpsilonGreedySelector(epsilon=0.0, seed=42)
        result = await selector.select("code", candidates=["a", "b", "c"])
        assert result == "a"

    def test_seed_reproducibility(self):
        loop = asyncio.new_event_loop()
        s1 = EpsilonGreedySelector(epsilon=0.5, seed=99)
        s2 = EpsilonGreedySelector(epsilon=0.5, seed=99)
        candidates = ["a", "b", "c", "d"]

        r1 = [loop.run_until_complete(s1.select("t", candidates=candidates)) for _ in range(20)]
        r2 = [loop.run_until_complete(s2.select("t", candidates=candidates)) for _ in range(20)]
        loop.close()
        assert r1 == r2
