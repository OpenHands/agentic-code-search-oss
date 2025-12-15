import os
import sys
import time
import fcntl
from pathlib import Path
from typing import Optional, List, Any
import hashlib
import subprocess
from filelock import FileLock, Timeout

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

def ensure_index_exists_with_robust_locking(
    repo_commit_hash: str,
    workspace_path: str,
    embedding_service,
    cache_dir: str = "/data/user_data/sanidhyv/tmp/embedding_cache"
) -> tuple[SemanticSearch, str]:
    """
    Ensure the semantic search index exists with robust file locking.
    
    This prevents multiple workers from corrupting the ChromaDB database
    by ensuring only one worker can initialize the index at a time.
    """
    index_dir = Path(cache_dir) / repo_commit_hash
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a lock file to prevent concurrent initialization
    lock_file = index_dir / ".init.lock"
    lock = FileLock(str(lock_file), timeout=300)  # 5 minute timeout
    
    # Also check for a "ready" marker file
    ready_file = index_dir / ".ready"
    
    worker_id = os.getpid()
    
    # Fast path: if index is already ready, just load it
    if ready_file.exists():
        print(f"[Worker {worker_id}] Index already exists and is ready")
        try:
            index = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(index_dir),
                embedding_service=embedding_service,
                device="cpu",
                num_threads=4,
            )
            return index, str(index_dir)
        except Exception as e:
            print(f"[Worker {worker_id}] Failed to load existing index: {e}")
            # Fall through to recreation logic
            ready_file.unlink(missing_ok=True)
    
    # Slow path: need to create or recreate the index
    print(f"[Worker {worker_id}] Acquiring lock to initialize index...")
    
    try:
        with lock:
            print(f"[Worker {worker_id}] Lock acquired, checking if index needs creation...")
            
            # Double-check: another worker might have created it while we waited
            if ready_file.exists():
                print(f"[Worker {worker_id}] Another worker created the index while we waited")
                index = SemanticSearch(
                    collection_name=f"code_{repo_commit_hash}",
                    persist_directory=str(index_dir),
                    embedding_service=embedding_service,
                    device="cpu",
                    num_threads=4,
                )
                return index, str(index_dir)
            
            # We need to create the index
            print(f"[Worker {worker_id}] Creating new index...")
            
            # Clean up any corrupted database files
            chroma_db_file = index_dir / "chroma.sqlite3"
            if chroma_db_file.exists():
                print(f"[Worker {worker_id}] Removing existing database file")
                chroma_db_file.unlink()
            
            # Create the index
            index = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(index_dir),
                embedding_service=embedding_service,
                device="cpu",
                num_threads=4,
            )
            
            # Index the workspace
            print(f"[Worker {worker_id}] Indexing workspace: {workspace_path}")
            index.index_codebase(workspace_path)
            
            # Mark as ready
            ready_file.touch()
            print(f"[Worker {worker_id}] Index created and marked as ready")
            
            return index, str(index_dir)
            
    except Timeout:
        print(f"[Worker {worker_id}] Timeout waiting for lock - another worker is creating the index")
        # Wait a bit more for the other worker to finish
        time.sleep(10)
        
        if ready_file.exists():
            print(f"[Worker {worker_id}] Index ready after timeout, loading...")
            index = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(index_dir),
                embedding_service=embedding_service,
                device="cpu",
                num_threads=4,
            )
            return index, str(index_dir)
        else:
            raise RuntimeError("Timeout waiting for index creation by another worker")
    
    except Exception as e:
        print(f"[Worker {worker_id}] Error during index creation: {e}")
        # Clean up the ready file if something went wrong
        ready_file.unlink(missing_ok=True)
        raise

def ensure_index_exists(repo_commit_hash: str, workspace_path: str) -> tuple[SemanticSearch, str]:
    """Wrapper that uses the robust locking mechanism."""
    return ensure_index_exists_with_robust_locking(
        repo_commit_hash=repo_commit_hash,
        workspace_path=workspace_path,
        embedding_service=embedding_service,  # Your global embedding service
        cache_dir="/data/user_data/sanidhyv/tmp/embedding_cache"
    )

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