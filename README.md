# strategy-ai

Thompson Sampling and epsilon-greedy bandit-based strategy selection for LLM model routing.

Zero external dependencies. Stdlib only. Python 3.11+.

## Install

```bash
pip install "git+https://github.com/FlossWare/strategy-ai.git"
```

## Quick Start

```python
import asyncio
from strategy_ai import ThompsonSamplingSelector, EpsilonGreedySelector

async def main():
    # Thompson Sampling
    ts = ThompsonSamplingSelector()
    selected = await ts.select("code_gen", candidates=["gpt-4o", "claude-sonnet", "gemini"])
    print(f"Selected: {selected}")

    # Record outcome
    await ts.update(selected, "code_gen", reward=0.9)

    # Check stats
    stats = await ts.performance()
    for key, s in stats.items():
        print(f"  {key}: trials={s.total_trials}, avg_reward={s.avg_reward:.2f}")

    # Epsilon-greedy alternative
    eg = EpsilonGreedySelector(epsilon=0.1)
    selected = await eg.select("code_gen", candidates=["gpt-4o", "claude-sonnet", "gemini"])
    print(f"Epsilon-greedy selected: {selected}")

asyncio.run(main())
```

## Decorators

```python
from strategy_ai import with_thompson_sampling, with_epsilon_greedy

@with_thompson_sampling(task_type="code_gen", seed=42)
async def generate_code(prompt: str, *, selected: str = ""):
    # 'selected' is automatically set to the chosen candidate
    return await call_model(selected, prompt)

result = await generate_code("Write a function", candidates=["gpt-4o", "claude-sonnet"])
```

## Protocols

```python
from strategy_ai import StrategySelector, RewardTracker, PerformanceReporter

# Both ThompsonSamplingSelector and EpsilonGreedySelector satisfy all three
assert isinstance(ThompsonSamplingSelector(), StrategySelector)
assert isinstance(EpsilonGreedySelector(), RewardTracker)
```

## License

MIT
