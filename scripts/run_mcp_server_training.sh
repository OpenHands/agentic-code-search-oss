#!/bin/bash
# MCP server wrapper for training (lightweight, uses Ray actor)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    exec "$SCRIPT_DIR/.venv/bin/python" src/mcp_server/training_semantic_search_server.py
else
    exec python src/mcp_server/training_semantic_search_server.py
fi
