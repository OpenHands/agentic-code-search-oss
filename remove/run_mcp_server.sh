#!/bin/bash
# Wrapper to run MCP server from correct directory
# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use uv run --no-sync to skip dependency installation (already installed)
exec uv run --no-sync python src/mcp_server/training_semantic_search_server.py
