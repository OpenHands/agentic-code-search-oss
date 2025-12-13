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


def log(msg: str):
    """Log to stderr to avoid polluting MCP's stdout JSON-RPC channel."""
    print(msg, file=sys.stderr, flush=True)


server = Server("semantic-code-search-training")


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
    
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
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


def ensure_index_exists(repo_path: Path, cache_dir: Path, repo_commit_hash: str) -> bool:
    """
    Ensure index is created (with file locking to prevent concurrent creation).
    
    Returns:
        bool: True if index exists/was created successfully
    """
    lock_file = cache_dir / ".lock"
    marker_file = cache_dir / ".indexed"
    
    # Quick check: if marker exists, we're done
    if marker_file.exists():
        log(f"[Index] Already indexed (marker exists): {cache_dir}")
        return True
    
    # Need to index: acquire lock to prevent concurrent indexing
    with FileLock(lock_file):
        # Double-check after acquiring lock (another worker might have finished)
        if marker_file.exists():
            log(f"[Index] Already indexed (marker exists after lock): {cache_dir}")
            return True
        
        log(f"[Index] Creating index for {repo_commit_hash}...")
        
        try:
            from src.tools.semantic_search import SemanticSearch
            
            # Create index with exclusive access
            index = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(cache_dir),
                device="cpu",
                embedding_model_name="all-MiniLM-L6-v2",
                reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                num_threads=4,
            )
            
            exclude_patterns = [
                "__pycache__", ".pytest_cache",
                "node_modules", ".venv", "venv", "env", ".git",
                ".tox", ".eggs", "dist", "build",
                "*_test.py", "test_*.py", "*Test.py", "*Tests.py"
            ]
            
            stats = index.index_code_files(
                str(repo_path),
                file_extensions=[".py"],
                batch_size=32,
                exclude_patterns=exclude_patterns
            )
            
            log(f"[Index] Indexed {stats['total_chunks']} chunks from {stats['indexed_files']} files")
            
            # Verify indexing succeeded
            final_stats = index.get_stats()
            if final_stats["total_documents"] == 0:
                log(f"[Index] ERROR: Index is empty after indexing!")
                return False
            
            # Create marker file to indicate indexing is complete
            marker_file.write_text(f"{stats['total_chunks']}")
            log(f"[Index] Created marker file: {marker_file}")
            
            return True
            
        except Exception as e:
            import traceback
            log(f"[Index] ERROR during indexing: {e}")
            log(traceback.format_exc())
            return False


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
    
    cache_dir = Path(f"/data/user_data/sanidhyv/tmp/embedding_cache/{repo_commit_hash}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"[Semantic Search] Repo: {repo_name}@{commit[:8]}, Hash: {repo_commit_hash}")
    
    # Ensure index exists (with locking to prevent concurrent creation)
    if not ensure_index_exists(repo_path, cache_dir, repo_commit_hash):
        return [TextContent(type="text", text=f"Failed to create index for {repo_name}")]
    
    # Now safely search (read-only, no locking needed)
    try:
        from src.tools.semantic_search import SemanticSearch
        
        # Open in read-only mode
        index = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(cache_dir),
            device="cpu",
            embedding_model_name="all-MiniLM-L6-v2",
            reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            num_threads=4,
        )
        
        stats = index.get_stats()
        log(f"[Semantic Search] Searching {stats['total_documents']} documents")
        
        if stats["total_documents"] == 0:
            return [TextContent(type="text", text=f"Index is empty (no documents found)")]
        
        results = index.search(query, n_results=n_results, use_reranker=True)
        
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