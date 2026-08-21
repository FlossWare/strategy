"""Convenience decorators for strategy-ai (ADR-0006: Cross-Cutting Decorators).

Provides ``@with_thompson_sampling`` and ``@with_epsilon_greedy`` decorators
that wrap async model/strategy selection functions with bandit-based
exploration-exploitation balancing.

Users must explicitly opt in by applying the decorator and configuring
the selector (ADR-0001: Explicit Opt-In).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from strategy_ai.epsilon_greedy import EpsilonGreedySelector
from strategy_ai.strategy import ThompsonSamplingSelector


def with_thompson_sampling(
    *,
    selector: ThompsonSamplingSelector | None = None,
    task_type: str = "default",
    candidates_kwarg: str = "candidates",
    reward_threshold: float = 0.5,
    seed: int | None = None,
) -> Callable:
    """Decorator that selects from candidates using Thompson Sampling.

    The decorated async function must accept a keyword argument named
    by *candidates_kwarg* (default ``"candidates"``) containing a list
    of strategy/model names.  The decorator selects the best candidate
    and passes only the selected one to the wrapped function.

    After the function returns, the decorator calls ``update()`` on the
    selector.  If the return value has a ``reward`` attribute, that
    value is used; otherwise ``1.0`` is recorded (success) if no
    exception was raised.

    Parameters
    ----------
    selector:
        An existing ``ThompsonSamplingSelector`` instance to use.
        If ``None``, a new instance is created.
    task_type:
        The task type to use for bandit arm lookups.
    candidates_kwarg:
        The keyword argument name containing the list of candidates.
    reward_threshold:
        Threshold for auto-reward: success >= threshold.
    seed:
        Optional RNG seed (only used if *selector* is None).
    """
    ts = selector or ThompsonSamplingSelector(seed=seed)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            candidates = kwargs.pop(candidates_kwarg, None)
            if candidates is None or not candidates:
                return await fn(*args, **kwargs)

            selected = await ts.select(task_type, candidates=candidates)
            kwargs["selected"] = selected

            try:
                result = await fn(*args, **kwargs)
                if isinstance(result, dict):
                    reward = float(result.get("reward", 1.0))
                else:
                    reward = float(getattr(result, "reward", 1.0))
                await ts.update(selected, task_type, reward=reward)
                return result
            except Exception:
                await ts.update(selected, task_type, reward=0.0)
                raise

        wrapper._selector = ts  # type: ignore[attr-defined]
        return wrapper

    return decorator


def with_epsilon_greedy(
    *,
    selector: EpsilonGreedySelector | None = None,
    task_type: str = "default",
    candidates_kwarg: str = "candidates",
    epsilon: float = 0.1,
    seed: int | None = None,
) -> Callable:
    """Decorator that selects from candidates using epsilon-greedy.

    Behaves identically to ``@with_thompson_sampling`` but uses an
    ``EpsilonGreedySelector`` for selection.

    Parameters
    ----------
    selector:
        An existing ``EpsilonGreedySelector`` instance to use.
        If ``None``, a new instance is created.
    task_type:
        The task type to use for bandit arm lookups.
    candidates_kwarg:
        The keyword argument name containing the list of candidates.
    epsilon:
        Exploration rate (only used if *selector* is None).
    seed:
        Optional RNG seed (only used if *selector* is None).
    """
    eg = selector or EpsilonGreedySelector(epsilon=epsilon, seed=seed)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            candidates = kwargs.pop(candidates_kwarg, None)
            if candidates is None or not candidates:
                return await fn(*args, **kwargs)

            selected = await eg.select(task_type, candidates=candidates)
            kwargs["selected"] = selected

            try:
                result = await fn(*args, **kwargs)
                if isinstance(result, dict):
                    reward = float(result.get("reward", 1.0))
                else:
                    reward = float(getattr(result, "reward", 1.0))
                await eg.update(selected, task_type, reward=reward)
                return result
            except Exception:
                await eg.update(selected, task_type, reward=0.0)
                raise

        wrapper._selector = eg  # type: ignore[attr-defined]
        return wrapper

    return decorator
