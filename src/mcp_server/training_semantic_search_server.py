from __future__ import annotations
import os
import sys
import time
import fcntl
from pathlib import Path
from typing import Optional, List, Any
import hashlib
import subprocess

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError as e:
    raise ImportError(f"Please install MCP SDK: uv pip install mcp fastmcp\nError: {e}")

# Import SemanticSearch at the top
from src.tools.semantic_search import SemanticSearch


def log(msg: str):
    """Log to stderr to avoid polluting MCP's stdout JSON-RPC channel."""
    print(msg, file=sys.stderr, flush=True)


server = Server("semantic-code-search-training")

# Global embedding service (initialized once)
embedding_service = None


def get_embedding_service():
    """Get or create the global embedding service."""
    global embedding_service
    if embedding_service is None:
        # Initialize your embedding service here
        # This should match whatever you use in SemanticSearch
        try:
            import ray
            from src.tools.embedding_service import EmbeddingService
            
            # Get the embedding service from Ray if available
            if ray.is_initialized():
                embedding_service = ray.get_actor("EmbeddingService")
            else:
                # Create a local one if Ray is not available
                log("[EmbeddingService] Ray not initialized, using local service")
                embedding_service = None  # SemanticSearch will create its own
        except Exception as e:
            log(f"[EmbeddingService] Could not get service: {e}, using local")
            embedding_service = None
    
    return embedding_service


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
            capture_output=True, text=True, check=True,
        )
        commit = result.stdout.strip()

        result = subprocess.run(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, check=True,
        )
        url = result.stdout.strip()

        if "github.com" in url:
            parts = url.rstrip(".git").split("/")
            repo_name = "/".join(parts[-2:])
        else:
            repo_name = repo_path.name

        return repo_name, commit
    except Exception:
        return repo_path.name, "unknown"


def get_repo_commit_hash(repo_name: str, commit: str) -> str:
    """Get unique hash for (repo, commit) pair."""
    key = f"{repo_name}:{commit}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class FileLock:
    """Simple file-based lock for coordinating ChromaDB access."""
    
    def __init__(self, lock_file: Path | str):
        self.lock_file = Path(lock_file) if isinstance(lock_file, str) else lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fp = None
    
    def __enter__(self):
        self.fp = open(self.lock_file, 'w')
        log(f"[FileLock] Waiting for lock: {self.lock_file}")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
        log(f"[FileLock] Acquired lock: {self.lock_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()
            log(f"[FileLock] Released lock: {self.lock_file}")


def ensure_index_exists(repo_commit_hash: str, workspace_path: str) -> tuple[SemanticSearch, str]:
    """Load pre-existing index - DO NOT create new indices during training."""
    index_dir = Path(f"/data/user_data/sanidhyv/tmp/embedding_cache/{repo_commit_hash}")
    ready_file = index_dir / ".ready"
    worker_id = os.getpid()
    
    # Check if index exists WITH .ready marker
    if not ready_file.exists():
        raise FileNotFoundError(
            f"[Worker {worker_id}] Index not found for {repo_commit_hash}. "
            f"Expected at: {index_dir} with .ready marker"
        )
    
    # Load in read-only mode (will fail if collection doesn't exist)
    index = SemanticSearch(
        collection_name=f"code_{repo_commit_hash}",
        persist_directory=str(index_dir),
        device="cpu",
        num_threads=4,
        read_only=True,  # CRITICAL: Don't create, only load
    )
    
    return index, str(index_dir)
    

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="semantic_search",
            description=(
                "Search the current repository using semantic similarity. "
                "Automatically uses the workspace repository."
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
    """Handle semantic search with proper concurrency control."""
    query = arguments["query"]
    repo_path = Path(get_workspace_path()).resolve()
    n_results = arguments.get("n_results", 10)
    
    log(f"[Semantic Search] Query: '{query}'")
    log(f"[Semantic Search] Repo: {repo_path}")
    
    if not repo_path.exists():
        return [TextContent(type="text", text=f"Error: Repository path does not exist: {repo_path}")]

    # Get repo info
    repo_name, commit = get_repo_info(repo_path)
    repo_commit_hash = get_repo_commit_hash(repo_name, commit)
    
    log(f"[Semantic Search] Repo: {repo_name}@{commit[:8]}, Hash: {repo_commit_hash}")
    
    # Ensure index exists (with locking to prevent concurrent creation)
    # This returns (index, index_path) tuple
    try:
        index, index_path = ensure_index_exists(repo_commit_hash, str(repo_path))
        log(f"[Semantic Search] Index ready at: {index_path}")
    except Exception as e:
        error_msg = f"Failed to create/load index: {str(e)}"
        log(error_msg)
        return [TextContent(type="text", text=error_msg)]
    
    # Now perform the search
    try:
        stats = index.get_stats()
        log(f"[Semantic Search] Searching {stats['total_documents']} documents")
        
        if stats["total_documents"] == 0:
            return [TextContent(type="text", text=f"Index is empty (no documents found)")]
        
        results = index.search(query, n_results=n_results, use_reranker=False)
        
        if not results:
            return [TextContent(type="text", text=f"No results found for query: {query}")]
        
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
            if len(lines) > 20:
                preview = "\n".join(lines[:20])
                output_lines.append(f"\n{preview}\n   ... ({len(lines)} total lines)")
            else:
                output_lines.append(f"\n{content}")
        
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
        return [TextContent(type="text", text=f"Error: {str(e)}")]


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