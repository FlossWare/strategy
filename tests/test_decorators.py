"""Tests for strategy-ai decorators."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from strategy_ai import (
    EpsilonGreedySelector,
    ThompsonSamplingSelector,
    with_epsilon_greedy,
    with_thompson_sampling,
)


@dataclass
class _Result:
    value: str
    reward: float = 1.0


class TestWithThompsonSampling:
    """Tests for @with_thompson_sampling decorator."""

    @pytest.mark.asyncio
    async def test_selects_from_candidates(self):
        @with_thompson_sampling(seed=42)
        async def pick(*, selected: str = "") -> _Result:
            return _Result(value=selected)

        result = await pick(candidates=["a", "b", "c"])
        assert result.value in ("a", "b", "c")

    @pytest.mark.asyncio
    async def test_no_candidates_passthrough(self):
        @with_thompson_sampling(seed=42)
        async def pick() -> str:
            return "no_candidates"

        result = await pick()
        assert result == "no_candidates"

    @pytest.mark.asyncio
    async def test_empty_candidates_passthrough(self):
        @with_thompson_sampling(seed=42)
        async def pick(*, candidates: list[str] | None = None) -> str:
            return "empty"

        result = await pick(candidates=[])
        assert result == "empty"

    @pytest.mark.asyncio
    async def test_records_reward(self):
        ts = ThompsonSamplingSelector(seed=42)

        @with_thompson_sampling(selector=ts, task_type="test")
        async def pick(*, selected: str = "") -> _Result:
            return _Result(value=selected, reward=0.9)

        await pick(candidates=["a"])
        stats = await ts.performance()
        assert len(stats) == 1
        key = list(stats.keys())[0]
        assert stats[key].total_trials == 1

    @pytest.mark.asyncio
    async def test_records_failure_on_exception(self):
        ts = ThompsonSamplingSelector(seed=42)

        @with_thompson_sampling(selector=ts, task_type="test")
        async def pick(*, selected: str = "") -> _Result:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await pick(candidates=["a"])

        stats = await ts.performance()
        key = list(stats.keys())[0]
        assert stats[key].avg_reward == 0.0

    @pytest.mark.asyncio
    async def test_selector_exposed(self):
        @with_thompson_sampling(seed=42)
        async def pick(*, selected: str = "") -> str:
            return selected

        assert hasattr(pick, "_selector")
        assert isinstance(pick._selector, ThompsonSamplingSelector)

    @pytest.mark.asyncio
    async def test_custom_candidates_kwarg(self):
        @with_thompson_sampling(seed=42, candidates_kwarg="models")
        async def pick(*, selected: str = "") -> _Result:
            return _Result(value=selected)

        result = await pick(models=["x", "y"])
        assert result.value in ("x", "y")


class TestWithEpsilonGreedy:
    """Tests for @with_epsilon_greedy decorator."""

    @pytest.mark.asyncio
    async def test_selects_from_candidates(self):
        @with_epsilon_greedy(epsilon=0.1, seed=42)
        async def pick(*, selected: str = "") -> _Result:
            return _Result(value=selected)

        result = await pick(candidates=["a", "b", "c"])
        assert result.value in ("a", "b", "c")

    @pytest.mark.asyncio
    async def test_no_candidates_passthrough(self):
        @with_epsilon_greedy(epsilon=0.1, seed=42)
        async def pick() -> str:
            return "no_candidates"

        result = await pick()
        assert result == "no_candidates"

    @pytest.mark.asyncio
    async def test_records_reward(self):
        eg = EpsilonGreedySelector(epsilon=0.0, seed=42)

        @with_epsilon_greedy(selector=eg, task_type="test")
        async def pick(*, selected: str = "") -> _Result:
            return _Result(value=selected, reward=0.8)

        await pick(candidates=["a"])
        stats = await eg.performance()
        assert len(stats) == 1
        key = list(stats.keys())[0]
        assert stats[key].total_trials == 1
        assert stats[key].avg_reward == 0.8

    @pytest.mark.asyncio
    async def test_records_failure_on_exception(self):
        eg = EpsilonGreedySelector(epsilon=0.0, seed=42)

        @with_epsilon_greedy(selector=eg, task_type="test")
        async def pick(*, selected: str = "") -> _Result:
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await pick(candidates=["a"])

        stats = await eg.performance()
        key = list(stats.keys())[0]
        assert stats[key].avg_reward == 0.0

    @pytest.mark.asyncio
    async def test_selector_exposed(self):
        @with_epsilon_greedy(epsilon=0.2, seed=42)
        async def pick(*, selected: str = "") -> str:
            return selected

        assert hasattr(pick, "_selector")
        assert isinstance(pick._selector, EpsilonGreedySelector)
