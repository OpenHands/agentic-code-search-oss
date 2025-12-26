from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any
import hashlib
import subprocess
import gc

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError as e:
    raise ImportError(f"Please install MCP SDK: uv pip install mcp fastmcp\nError: {e}")

from src.tools.semantic_search import SemanticSearch


def log(msg: str):
    """Log to stderr to avoid polluting MCP's stdout JSON-RPC channel."""
    print(msg, file=sys.stderr, flush=True)


server = Server("semantic-code-search-training")

# ✅ Global cache for loaded index (one per MCP server process)
_loaded_index = None
_loaded_hash = None


def get_workspace_path() -> str:
    """Get workspace path from environment variable."""
    workspace = os.getenv("WORKSPACE_PATH")
    if not workspace:
        raise ValueError("WORKSPACE_PATH environment variable not set.")
    return workspace


def get_repo_info(repo_path: Path) -> tuple[str, str]:
    """Extract repo name and commit hash from git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10
        )
        commit = result.stdout.strip()

        result = subprocess.run(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, check=True, timeout=10
        )
        url = result.stdout.strip()

        if "github.com" in url:
            parts = url.rstrip(".git").split("/")
            repo_name = "/".join(parts[-2:])
        else:
            repo_name = repo_path.name

        return repo_name, commit
    except Exception as e:
        log(f"[get_repo_info] Error: {e}")
        return repo_path.name, "unknown"


def get_repo_commit_hash(repo_name: str, commit: str) -> str:
    """
    Get unique hash for (repo, commit) pair.
    
    CRITICAL: This must match the hash function used in:
    - src/services/batched_index_manager.py
    - src/services/embedding_service.py
    - scripts/clone_and_index_repos.py
    """
    key = f"{repo_name}:{commit}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_index(repo_commit_hash: str) -> SemanticSearch:
    """
    Load index from disk in read-only mode.
    
    Caches the loaded index globally to avoid reloading on every search.
    Memory-efficient: only loads embedder on CPU with limited threads.
    
    Args:
        repo_commit_hash: 16-character hash identifying the repo+commit
        
    Returns:
        SemanticSearch instance loaded in read-only mode
        
    Raises:
        FileNotFoundError: If index or .ready marker doesn't exist
        ValueError: If index is empty
    """
    global _loaded_index, _loaded_hash
    
    # ✅ Return cached index if same repo
    if _loaded_index is not None and _loaded_hash == repo_commit_hash:
        log(f"[load_index] Using cached index for {repo_commit_hash}")
        return _loaded_index
    
    # ✅ Clear previous index to free memory
    if _loaded_index is not None:
        log(f"[load_index] Clearing previous index {_loaded_hash}")
        del _loaded_index
        _loaded_index = None
        _loaded_hash = None
        gc.collect()
    
    # ✅ CRITICAL: Path must match what batched indexing uses
    cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "/data/user_data/sanidhyv/tmp/embedding_cache")
    index_path = Path(cache_dir) / repo_commit_hash
    ready_file = index_path / ".ready"
    
    log(f"[load_index] Looking for index at: {index_path}")
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_path}\n"
            f"Expected hash: {repo_commit_hash}"
        )
    
    if not ready_file.exists():
        raise FileNotFoundError(
            f"Index not ready for {repo_commit_hash}. "
            f"Missing .ready marker at: {ready_file}\n"
            f"This usually means indexing failed or is incomplete."
        )
    
    log(f"[load_index] Loading index from {index_path}")
    
    # ✅ CRITICAL: Model names must match what was used during indexing
    embedding_model = os.getenv("EMBEDDING_MODEL", "jinaai/jina-code-embeddings-0.5b")
    
    # ✅ Load in read-only mode, CPU-only, limited threads
    try:
        index = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(index_path),
            embedding_model_name=embedding_model,
            reranker_model_name=None,  # ✅ No reranker to save memory
            device="cpu",
            num_threads=2,  # ✅ Limit CPU threads to 2 per process
            read_only=True,  # ✅ Read-only mode prevents write conflicts
        )
    except Exception as e:
        log(f"[load_index] Failed to create SemanticSearch: {e}")
        raise
    
    # Verify it loaded
    try:
        stats = index.get_stats()
        log(f"[load_index] ✓ Loaded {stats['total_documents']} documents")
        
        if stats["total_documents"] == 0:
            raise ValueError(f"Index is empty for {repo_commit_hash}")
    except Exception as e:
        log(f"[load_index] Failed to get stats: {e}")
        raise
    
    # ✅ Cache it globally
    _loaded_index = index
    _loaded_hash = repo_commit_hash
    
    return index


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="semantic_search",
            description=(
                "Search the current repository using semantic similarity. "
                "Automatically uses the workspace repository. "
                "Returns relevant code chunks with file paths and similarity scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what you're looking for",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "semantic_search":
            return await handle_semantic_search(arguments)
        else:
            return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]
    except Exception as e:
        import traceback
        error_msg = f"Error executing {name}: {str(e)}\n{traceback.format_exc()}"
        log(error_msg)
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


async def handle_semantic_search(arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle semantic search by loading index directly from disk.
    
    This is the main entry point for semantic search during training.
    Compatible with the batched indexing workflow in train_batched.py.
    """
    query = arguments["query"]
    n_results = arguments.get("n_results", 10)
    
    # Get workspace path
    try:
        repo_path = Path(get_workspace_path()).resolve()
    except ValueError as e:
        error_msg = f"WORKSPACE_PATH not set: {str(e)}"
        log(error_msg)
        return [TextContent(type="text", text=error_msg)]
    
    log(f"[Semantic Search] Query: '{query}'")
    log(f"[Semantic Search] Repo: {repo_path}")
    
    if not repo_path.exists():
        return [TextContent(type="text", text=f"Error: Repository path does not exist: {repo_path}")]

    # Get repo info from git
    try:
        repo_name, commit = get_repo_info(repo_path)
        repo_commit_hash = get_repo_commit_hash(repo_name, commit)
        log(f"[Semantic Search] Repo: {repo_name}@{commit[:8]}, Hash: {repo_commit_hash}")
    except Exception as e:
        error_msg = f"Failed to get repo info: {str(e)}"
        log(error_msg)
        return [TextContent(type="text", text=error_msg)]
    
    # Load index from disk
    try:
        index = load_index(repo_commit_hash)
    except FileNotFoundError as e:
        error_msg = str(e)
        log(f"[Semantic Search] {error_msg}")
        return [TextContent(type="text", text=f"Index not available: {error_msg}")]
    except ValueError as e:
        error_msg = str(e)
        log(f"[Semantic Search] {error_msg}")
        return [TextContent(type="text", text=error_msg)]
    except Exception as e:
        import traceback
        error_msg = f"Failed to load index: {str(e)}\n{traceback.format_exc()}"
        log(error_msg)
        return [TextContent(type="text", text=f"Error loading index: {str(e)}")]
    
    # Perform search
    try:
        log(f"[Semantic Search] Searching with query: '{query}', n_results: {n_results}")
        results = index.search(query, n_results=n_results, use_reranker=False)
        
        if not results:
            log(f"[Semantic Search] No results found")
            return [TextContent(type="text", text=f"No results found for query: {query}")]
        
        log(f"[Semantic Search] Found {len(results)} results")
        
        # Format results
        output_lines = [f"Found {len(results)} relevant code chunks for: '{query}'\n"]
        
        for i, result in enumerate(results, 1):
            similarity = result.get("rerank_score", result.get("similarity_score", 0))
            score_type = "rerank" if "rerank_score" in result else "similarity"
            file_path = result["file_path"]
            chunk_idx = result["chunk_index"]
            total_chunks = result["metadata"]["total_chunks"]
            
            output_lines.append(f"\n{i}. {file_path} ({score_type}: {similarity:.3f})")
            output_lines.append(f"   Chunk {chunk_idx + 1}/{total_chunks}")
            
            content = result["content"]
            lines = content.split("\n")
            
            # ✅ Truncate long chunks to avoid token limits
            if len(lines) > 20:
                preview = "\n".join(lines[:20])
                output_lines.append(f"\n{preview}\n   ... ({len(lines)} total lines)")
            else:
                output_lines.append(f"\n{content}")
        
        # ✅ Add unique files summary
        unique_files = list(set(r["file_path"] for r in results))
        output_lines.append(f"\n\nUnique files ({len(unique_files)}):")
        for file_path in unique_files:
            output_lines.append(f"  - {file_path}")
        
        result_text = "\n".join(output_lines)
        return [TextContent(type="text", text=result_text)]
        
    except Exception as e:
        import traceback
        error_msg = f"Error in semantic search: {str(e)}\n{traceback.format_exc()}"
        log(error_msg)
        return [TextContent(type="text", text=f"Search error: {str(e)}")]


async def main():
    """Run the MCP server."""
    log("[MCP Server] Starting semantic-code-search-training server")
    log(f"[MCP Server] Python: {sys.version}")
    log(f"[MCP Server] Working dir: {os.getcwd()}")
    log(f"[MCP Server] WORKSPACE_PATH: {os.getenv('WORKSPACE_PATH', 'not set')}")
    log(f"[MCP Server] EMBEDDING_CACHE_DIR: {os.getenv('EMBEDDING_CACHE_DIR', 'using default')}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())