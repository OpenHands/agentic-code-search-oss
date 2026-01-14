#!/bin/bash
# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ✅ Change to repo root instead of scripts/
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# ✅ Use uv run with correct working directory
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    exec "$REPO_ROOT/.venv/bin/python" src/mcp_server/training_semantic_search_server.py
else
    # Use uv run from repo root
    exec uv run --directory "$REPO_ROOT" python src/mcp_server/training_semantic_search_server.py
fi