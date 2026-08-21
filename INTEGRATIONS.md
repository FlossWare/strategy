# strategy-ai Integrations

Install from GitHub:

```bash
pip install "git+https://github.com/FlossWare/strategy-ai.git"
```

---

## Claude Code

### CLAUDE.md Snippet

```markdown
## Bandit Strategy Selection (strategy-ai)

This project uses `strategy-ai` for bandit-based model/strategy selection.

- Thompson Sampling: `from strategy_ai import ThompsonSamplingSelector, with_thompson_sampling`
- Epsilon-greedy: `from strategy_ai import EpsilonGreedySelector, with_epsilon_greedy`
- Protocols: `from strategy_ai import StrategySelector, RewardTracker`
- Use `@with_thompson_sampling` decorator for automatic bandit-based model selection
- Use `@with_epsilon_greedy` decorator for simpler exploration-exploitation
- Zero external dependencies (stdlib only)
```

### Hook Example

Create `.claude/hooks/post-tool-edit.py` to suggest strategy selection for model routing:

```python
#!/usr/bin/env python3
"""Post-edit hook: flag files that could benefit from bandit-based selection."""
from __future__ import annotations

import sys

ROUTING_PATTERNS = ["router", "model", "strategy", "select", "dispatch", "backend"]


def main():
    if len(sys.argv) < 2:
        sys.exit(0)

    file_path = sys.argv[1].lower()
    for keyword in ROUTING_PATTERNS:
        if keyword in file_path:
            print(f"[strategy-ai] MODEL ROUTING FILE: {sys.argv[1]}")
            print("[strategy-ai] Consider bandit-based selection:")
            print("  from strategy_ai import ThompsonSamplingSelector")
            break


if __name__ == "__main__":
    main()
```

### Skill Example

Create `.claude/skills/bandit-select.md`:

```markdown
---
name: bandit-select
description: Select optimal model using Thompson Sampling
---

Use strategy-ai for bandit-based model selection:

\```python
import asyncio
from strategy_ai import ThompsonSamplingSelector

selector = ThompsonSamplingSelector()

# Select best model
selected = await selector.select("code_gen", candidates=["gpt-4o", "claude-sonnet", "gemini"])

# After using the model, record the outcome
await selector.update(selected, "code_gen", reward=quality_score)

# View performance stats
stats = await selector.performance()
for key, s in stats.items():
    print(f"  {key}: trials={s.total_trials}, avg={s.avg_reward:.2f}")
\```
```

---

## OpenAI Codex

### AGENTS.md Snippet

```markdown
## Bandit Strategy Selection

Use strategy-ai for bandit-based model/strategy selection:
- Install: `pip install "git+https://github.com/FlossWare/strategy-ai.git"`
- Thompson Sampling: `ThompsonSamplingSelector()` for exploration-exploitation
- Epsilon-greedy: `EpsilonGreedySelector(epsilon=0.1)` for simpler selection
- Decorator: `@with_thompson_sampling(task_type="code_gen", candidates_kwarg="models")`
- Protocols: `StrategySelector`, `RewardTracker`, `PerformanceReporter`
```

### Tool Definition

```python
from strategy_ai import ThompsonSamplingSelector

selector = ThompsonSamplingSelector(seed=42)

# Select from available models
selected = await selector.select("code_gen", candidates=["gpt-4o", "claude-sonnet", "gemini"])

# Record outcome after execution
await selector.update(selected, "code_gen", reward=0.85)
```

---

## Cursor

### .cursorrules Snippet

```
When selecting between multiple LLM models or strategies, use strategy-ai:

- Import: from strategy_ai import ThompsonSamplingSelector, with_thompson_sampling
- Thompson Sampling: ThompsonSamplingSelector() for exploration-exploitation balance
- Epsilon-greedy: EpsilonGreedySelector(epsilon=0.1) for simpler selection
- Decorator: @with_thompson_sampling(task_type="...", seed=42)
- Always record outcomes with selector.update(model, task_type, reward=score)
- Zero dependencies - stdlib only
- Install: pip install "git+https://github.com/FlossWare/strategy-ai.git"
```

---

## Crush

### Configuration

```python
# crush.config.py
from strategy_ai import ThompsonSamplingSelector, EpsilonGreedySelector

# Shared selector instance (persists across calls within session)
model_selector = ThompsonSamplingSelector(seed=42)

async def select_model(task_type: str, candidates: list[str]) -> str:
    """Select the best model for the given task type."""
    return await model_selector.select(task_type, candidates=candidates)

async def record_outcome(model: str, task_type: str, reward: float) -> None:
    """Record the outcome of using a model."""
    await model_selector.update(model, task_type, reward=reward)

async def get_stats(task_type: str | None = None) -> dict:
    """Get performance statistics."""
    return await model_selector.performance(task_type=task_type)
```

---

## Generic Python Agent

### Thompson Sampling Model Selection

```python
import asyncio
from strategy_ai import ThompsonSamplingSelector, EpsilonGreedySelector, StrategyStats

async def main():
    # 1. Create a selector
    selector = ThompsonSamplingSelector(seed=42)

    # 2. Available models
    models = ["gpt-4o", "claude-sonnet", "gemini-flash", "llama-70b"]

    # 3. Run multiple rounds of selection + feedback
    for i in range(20):
        selected = await selector.select("code_gen", candidates=models)
        # Simulate varying quality by model
        rewards = {"gpt-4o": 0.85, "claude-sonnet": 0.90, "gemini-flash": 0.70, "llama-70b": 0.60}
        reward = rewards.get(selected, 0.5)
        await selector.update(selected, "code_gen", reward=reward)

    # 4. Check learned preferences
    stats = await selector.performance(task_type="code_gen")
    print("Learned model performance:")
    for key, s in sorted(stats.items(), key=lambda x: x[1].avg_reward, reverse=True):
        print(f"  {s.strategy}: {s.total_trials} trials, avg={s.avg_reward:.2f}, "
              f"alpha={s.alpha:.0f}, beta={s.beta:.0f}")

    # 5. Now selections should favor the best model
    selections = [await selector.select("code_gen", candidates=models) for _ in range(100)]
    for m in models:
        pct = selections.count(m)
        print(f"  {m}: selected {pct}% of time")

asyncio.run(main())
```

### Decorator Pattern

```python
from strategy_ai import with_thompson_sampling, with_epsilon_greedy

@with_thompson_sampling(task_type="code_gen", seed=42)
async def generate_code(prompt: str, *, selected: str = ""):
    """Model is automatically selected from candidates via Thompson Sampling."""
    return await call_model(selected, prompt)

# Pass candidates at call time
result = await generate_code("Write a sort function",
                              candidates=["gpt-4o", "claude-sonnet", "gemini"])
```

---

## Cross-Package Integration

### strategy-ai + model-router-ai

Route to the best model using bandit selection:

```python
from strategy_ai import with_thompson_sampling
from model_router_ai import with_model_routing

@with_thompson_sampling(task_type="code_gen")
@with_model_routing(fallback="gpt-4o")
async def smart_generate(prompt: str, *, selected: str = "", model: str = "default"):
    return await backend.chat([{"role": "user", "content": prompt}], model=selected or model)
```

### strategy-ai + resilience-ai

Bandit selection with circuit breaker:

```python
from strategy_ai import with_thompson_sampling
from resilience_ai import with_retry, with_circuit_breaker

@with_thompson_sampling(task_type="code_gen")
@with_retry(max_attempts=3)
@with_circuit_breaker(provider="llm")
async def resilient_select(prompt: str, *, selected: str = "", model: str = "default"):
    return await backend.chat([{"role": "user", "content": prompt}], model=selected)
```

### Full Stack: All Packages

```python
from strategy_ai import with_thompson_sampling
from consensus_ai import with_consensus
from structured_output_ai import structured_output
from resilience_ai import with_retry, with_circuit_breaker
from observability_ai import track_execution
from evaluation_ai import adversarial_verify

@structured_output(schema=SCHEMA)
@track_execution(telemetry=t)
@adversarial_verify(backend=eval_b)
@with_thompson_sampling(task_type="code")
@with_retry(max_attempts=3)
@with_circuit_breaker(provider="llm")
async def production_query(prompt, *, selected="", model="default"):
    return await backend.chat([{"role": "user", "content": prompt}], model=selected)
```
