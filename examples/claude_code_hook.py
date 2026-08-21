#!/usr/bin/env python3
"""Claude Code hook: suggest bandit-based selection for model routing files.

Usage as a post-tool-edit hook in .claude/hooks/post-tool-edit.py:
    python3 examples/claude_code_hook.py "$FILE_PATH"
"""
from __future__ import annotations

import sys

ROUTING_PATTERNS = [
    "router", "model", "strategy", "select", "dispatch",
    "backend", "provider", "route", "balance",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: claude_code_hook.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    lower_path = file_path.lower()

    matches = [p for p in ROUTING_PATTERNS if p in lower_path]
    if not matches:
        sys.exit(0)

    print(f"[strategy-ai] MODEL ROUTING FILE: {file_path}")
    print(f"[strategy-ai] Matched patterns: {', '.join(matches)}")
    print("[strategy-ai] Consider bandit-based model selection:")
    print("  from strategy_ai import ThompsonSamplingSelector")
    print("  selector = ThompsonSamplingSelector()")
    print('  selected = await selector.select("task", candidates=models)')


if __name__ == "__main__":
    main()
