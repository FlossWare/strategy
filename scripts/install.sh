#!/bin/bash
# Install strategy-ai from GitHub
set -e

pip install "git+https://github.com/FlossWare/strategy-ai.git"

echo "strategy-ai installed successfully"
echo "Verify: python3 -c 'import strategy_ai; print(strategy_ai.__version__)'"
