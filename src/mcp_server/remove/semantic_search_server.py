"""
MCP Server for Semantic Code Search.

This server exposes semantic search capabilities through the Model Context Protocol (MCP),
allowing AI agents to perform vector-based code search.
"""

import json
import os
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, EmbeddedResource
except ImportError as e:
    raise ImportError(
        f"Please install MCP SDK: uv pip install mcp fastmcp\nError: {e}"
    )

from src.tools.semantic_search import SemanticSearch


# Global state for the MCP server
server = Server("semantic-code-search")
indices = {}  # Store indices per repository


def get_workspace_path() -> str:
    """Get workspace path from environment variable."""
    workspace = os.getenv("WORKSPACE_PATH")
    if not workspace:
        raise ValueError(
            "WORKSPACE_PATH environment variable not set. "
            "Please configure the MCP server with the workspace path."
        )
    return workspace


def get_cache_dir() -> Path:
    """Get cache directory for persistent indices."""
    cache_dir = os.getenv("INDEX_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    # Default to persistent cache location
    return Path.home() / ".cache" / "swebench_indices"


def get_repo_info(repo_path: Path) -> tuple[str, str]:
    """Extract repo name and commit hash from git repository."""
    try:
        # Get current commit
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()

        # Get remote URL to extract repo name
        result = subprocess.run(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        )
        url = result.stdout.strip()

        # Parse repo name from URL (e.g., https://github.com/owner/repo.git -> owner/repo)
        if "github.com" in url:
            parts = url.rstrip(".git").split("/")
            repo_name = "/".join(parts[-2:])
        else:
            repo_name = repo_path.name

        return repo_name, commit
    except Exception:
        # Fallback: use directory name
        return repo_path.name, "unknown"


def get_repo_commit_hash(repo_name: str, commit: str) -> str:
    """Get unique hash for (repo, commit) pair for index keying."""
    key = f"{repo_name}:{commit}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="index_repository",
            description=(
                "Index a code repository for semantic search. "
                "This creates a vector index of all code files in the repository. "
                "Should be called once before searching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository (optional, defaults to current workspace)",
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to index (e.g., ['.py', '.js'])",
                        "default": [".py"],
                    },
                    "force_rebuild": {
                        "type": "boolean",
                        "description": "Force rebuild the index even if it exists",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="semantic_search",
            description=(
                "Search a repository using semantic similarity. "
                "Finds code based on natural language descriptions, not just keywords. "
                "More powerful than grep for finding code by meaning. "
                "Example: 'function that parses git diffs' or 'code for calculating precision and recall'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what you're looking for",
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository (optional, defaults to current workspace)",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "return_content": {
                        "type": "boolean",
                        "description": "Whether to return full code content or just file paths",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_index_stats",
            description=(
                "Get statistics about an indexed repository, "
                "including number of indexed files and chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository (optional, defaults to current workspace)",
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    try:
        if name == "index_repository":
            return await handle_index_repository(arguments)
        elif name == "semantic_search":
            return await handle_semantic_search(arguments)
        elif name == "get_index_stats":
            return await handle_get_index_stats(arguments)
        else:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Unknown tool '{name}'",
                )
            ]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}",
            )
        ]


async def handle_index_repository(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle repository indexing."""
    repo_path = Path(arguments.get("repo_path") or get_workspace_path()).resolve()
    file_extensions = arguments.get("file_extensions", [".py"])
    force_rebuild = arguments.get("force_rebuild", False)

    if not repo_path.exists():
        return [
            TextContent(
                type="text",
                text=f"Error: Repository path does not exist: {repo_path}",
            )
        ]

    # Get repo info for proper cache keying
    repo_name, commit = get_repo_info(repo_path)
    repo_commit_hash = get_repo_commit_hash(repo_name, commit)

    # Use persistent cache directory keyed by (repo, commit)
    cache_dir = get_cache_dir()
    persist_dir = cache_dir / repo_commit_hash
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Key by (repo, commit) hash for reuse across instances
    repo_key = repo_commit_hash
    if repo_key not in indices or force_rebuild:
        index = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(persist_dir),
        )

        if force_rebuild:
            index.clear_index()

        # Index the repository
        stats = index.index_code_files(
            str(repo_path), file_extensions=file_extensions
        )
        indices[repo_key] = index

        result = (
            f"Successfully indexed repository: {repo_name}@{commit[:8]}\n"
            f"Files indexed: {stats['indexed_files']}\n"
            f"Total chunks: {stats['total_chunks']}\n"
            f"Cache key: {repo_commit_hash}\n"
            f"Cache dir: {persist_dir}"
        )
    else:
        index = indices[repo_key]
        stats = index.get_stats()
        result = (
            f"Repository already indexed: {repo_name}@{commit[:8]}\n"
            f"Total documents: {stats['total_documents']}\n"
            f"Cache key: {repo_commit_hash}\n"
            f"Use force_rebuild=true to rebuild the index"
        )

    return [TextContent(type="text", text=result)]


async def handle_semantic_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle semantic search."""
    query = arguments["query"]
    repo_path = Path(arguments.get("repo_path") or get_workspace_path()).resolve()
    n_results = arguments.get("n_results", 10)
    return_content = arguments.get("return_content", True)

    if not repo_path.exists():
        return [
            TextContent(
                type="text",
                text=f"Error: Repository path does not exist: {repo_path}",
            )
        ]

    # Get repo info for proper cache keying
    repo_name, commit = get_repo_info(repo_path)
    repo_commit_hash = get_repo_commit_hash(repo_name, commit)

    # Use persistent cache directory keyed by (repo, commit)
    cache_dir = get_cache_dir()
    persist_dir = cache_dir / repo_commit_hash
    persist_dir.mkdir(parents=True, exist_ok=True)

    repo_key = repo_commit_hash
    if repo_key not in indices:
        index = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(persist_dir),
        )
        indices[repo_key] = index

        # Check if index exists
        stats = index.get_stats()
        if stats["total_documents"] == 0:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Repository not indexed. Please call index_repository first.",
                )
            ]
    else:
        index = indices[repo_key]

    # Perform search
    results = index.search(query, n_results=n_results)

    if not results:
        return [
            TextContent(
                type="text",
                text=f"No results found for query: {query}",
            )
        ]

    # Format results
    output_lines = [f"Found {len(results)} relevant code chunks for: '{query}'\n"]

    for i, result in enumerate(results, 1):
        similarity = result["similarity_score"]
        file_path = result["file_path"]
        chunk_idx = result["chunk_index"]
        total_chunks = result["metadata"]["total_chunks"]

        output_lines.append(
            f"\n{i}. {file_path} (similarity: {similarity:.3f})"
        )
        output_lines.append(f"   Chunk {chunk_idx + 1}/{total_chunks}")

        if return_content:
            # Show content with limited preview
            content = result["content"]
            lines = content.split("\n")
            if len(lines) > 20:
                preview = "\n".join(lines[:20])
                output_lines.append(f"\n{preview}\n   ... ({len(lines)} total lines)")
            else:
                output_lines.append(f"\n{content}")

    # Add unique files summary
    unique_files = index.get_unique_files(results)
    output_lines.append(f"\n\nUnique files ({len(unique_files)}):")
    for file_path in unique_files:
        output_lines.append(f"  - {file_path}")

    result_text = "\n".join(output_lines)

    return [TextContent(type="text", text=result_text)]


async def handle_get_index_stats(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle getting index statistics."""
    repo_path = Path(arguments.get("repo_path") or get_workspace_path()).resolve()

    if not repo_path.exists():
        return [
            TextContent(
                type="text",
                text=f"Error: Repository path does not exist: {repo_path}",
            )
        ]

    # Get repo info for proper cache keying
    repo_name, commit = get_repo_info(repo_path)
    repo_commit_hash = get_repo_commit_hash(repo_name, commit)

    # Use persistent cache directory keyed by (repo, commit)
    cache_dir = get_cache_dir()
    persist_dir = cache_dir / repo_commit_hash
    persist_dir.mkdir(parents=True, exist_ok=True)

    repo_key = repo_commit_hash
    if repo_key not in indices:
        index = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(persist_dir),
        )
        indices[repo_key] = index
    else:
        index = indices[repo_key]

    stats = index.get_stats()

    result = (
        f"Index Statistics for {repo_name}@{commit[:8]}:\n"
        f"Cache key: {repo_commit_hash}\n"
        f"Total documents: {stats['total_documents']}\n"
        f"Embedding model: {stats['embedding_model']}\n"
        f"Cache directory: {persist_dir}"
    )

    return [TextContent(type="text", text=result)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
