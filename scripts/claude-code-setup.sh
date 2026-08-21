#!/bin/bash
# Add strategy-ai integration to your CLAUDE.md
set -e

CLAUDE_MD="${CLAUDE_MD:-./CLAUDE.md}"

if [ ! -f "$CLAUDE_MD" ]; then
    echo "Creating $CLAUDE_MD"
    touch "$CLAUDE_MD"
fi

cat >> "$CLAUDE_MD" << 'EOF'

## Bandit Strategy Selection (strategy-ai)

This project uses [strategy-ai](https://github.com/FlossWare/strategy-ai) for bandit-based model/strategy selection.

**Install:** `pip install "git+https://github.com/FlossWare/strategy-ai.git"`

**Key imports:**
```python
from strategy_ai import ThompsonSamplingSelector, EpsilonGreedySelector, with_thompson_sampling
```

**Usage patterns:**
- Thompson Sampling: `ThompsonSamplingSelector()` for exploration-exploitation
- Epsilon-greedy: `EpsilonGreedySelector(epsilon=0.1)` for simpler selection
- Decorator: `@with_thompson_sampling(task_type="code_gen", seed=42)`
- Always record outcomes: `await selector.update(model, task_type, reward=score)`
- Zero external dependencies (stdlib only)
EOF

echo "Added strategy-ai integration to $CLAUDE_MD"
