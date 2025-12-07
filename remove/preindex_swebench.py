#!/usr/bin/env python3
"""
Pre-index SWE-bench Lite repositories for faster evaluation.

This script:
1. Downloads SWE-bench Lite dataset
2. Identifies unique (repo, commit) combinations
3. Clones repos at specific commits
4. Creates vector indices for all unique repos
5. Stores indices in persistent cache for reuse

Usage:
    # Index all repos (recommended for SWE-bench Lite)
    python preindex_swebench.py

    # Index only frequently-used repos (for larger datasets)
    python preindex_swebench.py --min-frequency 3

    # Use GPU for faster indexing
    python preindex_swebench.py --gpu

Note: For SWE-bench Lite (300 instances, ~297 unique repos), using --min-frequency
will skip almost all repos since each appears only ~1 time. Use without filters
or use scripts/clone_and_index_repos.py which clones and indexes in one pass.
"""

import argparse
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

from src.tools.semantic_search import SemanticSearch


def get_repo_commit_hash(repo_name: str, commit: str) -> str:
    """Get unique hash for (repo, commit) pair."""
    key = f"{repo_name}:{commit}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def clone_repo_at_commit(repo_name: str, commit: str, clone_dir: Path) -> Path:
    """Clone repo at specific commit."""
    repo_path = clone_dir / repo_name.replace("/", "_")

    if repo_path.exists():
        # Verify commit
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() == commit:
            return repo_path
        else:
            print(f"  Commit mismatch, re-cloning...")
            subprocess.run(["rm", "-rf", str(repo_path)], check=True)

    print(f"  Cloning {repo_name}@{commit[:8]}...")
    subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{repo_name}.git", str(repo_path)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "checkout", "--quiet", commit],
        check=True,
    )

    return repo_path


def main():
    parser = argparse.ArgumentParser(description="Pre-index SWE-bench repositories")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "swebench_indices",
        help="Directory to store indices",
    )
    parser.add_argument(
        "--clone-dir",
        type=Path,
        default=Path("./swebench_repos_temp"),
        help="Temporary directory for cloning repos",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for embedding (requires sufficient VRAM)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Limit to first N instances (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Index only N repos per run (for incremental indexing)",
    )
    parser.add_argument(
        "--filter-repo",
        type=str,
        default=None,
        help="Only index repos matching this pattern (e.g., 'django' or 'django/django')",
    )
    parser.add_argument(
        "--priority-repos",
        type=str,
        nargs="+",
        help="Index these repos first (space-separated list)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already indexed repos (default behavior, kept for backward compat)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=None,
        help="Only index repos appearing in >= N instances (space optimization)",
    )
    args = parser.parse_args()

    # Create directories
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.clone_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SWE-bench Lite Pre-Indexing")
    print("=" * 80)
    print(f"Cache directory: {args.cache_dir}")
    print(f"Clone directory: {args.clone_dir}")
    print(f"Device: {'GPU' if args.gpu else 'CPU'}")
    print()

    # Load dataset
    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    if args.num_instances:
        dataset = dataset.select(range(args.num_instances))
        print(f"Limited to first {args.num_instances} instances")

    print(f"Total instances: {len(dataset)}")
    print()

    # Find unique (repo, commit) combinations
    repo_commits = {}
    for row in dataset:
        key = (row["repo"], row["base_commit"])
        if key not in repo_commits:
            repo_commits[key] = []
        repo_commits[key].append(row["instance_id"])

    print(f"Unique (repo, commit) combinations: {len(repo_commits)}")
    print(f"Average instances per combination: {len(dataset) / len(repo_commits):.1f}")
    print()

    # Apply frequency filter FIRST (most important for space optimization)
    filtered_repos = list(repo_commits.items())

    if args.min_frequency:
        # Filter to only repos with >= min_frequency instances
        filtered_repos = [
            (key, val) for key, val in filtered_repos
            if len(val) >= args.min_frequency
        ]
        instances_covered = sum(len(val) for _, val in filtered_repos)
        coverage_pct = (instances_covered / len(dataset)) * 100
        print(f"After --min-frequency {args.min_frequency}:")
        print(f"  Repos to index: {len(filtered_repos)}/{len(repo_commits)} ({len(filtered_repos)/len(repo_commits)*100:.0f}%)")
        print(f"  Instances covered: {instances_covered}/{len(dataset)} ({coverage_pct:.1f}%)")
        print(f"  Space saved: ~{len(repo_commits) - len(filtered_repos)} indices not created")
        print()

    if args.filter_repo:
        filtered_repos = [
            (key, val) for key, val in filtered_repos
            if args.filter_repo.lower() in key[0].lower()
        ]
        print(f"After --filter-repo '{args.filter_repo}': {len(filtered_repos)} repos")

    # Sort by priority if specified
    if args.priority_repos:
        priority_set = set(args.priority_repos)
        priority_repos = [(k, v) for k, v in filtered_repos if k[0] in priority_set]
        other_repos = [(k, v) for k, v in filtered_repos if k[0] not in priority_set]
        filtered_repos = priority_repos + other_repos
        print(f"Prioritized {len(priority_repos)} repos: {args.priority_repos}")

    # Apply batch size
    if args.batch_size:
        # Find how many already indexed to determine batch offset
        already_indexed = []
        for (repo_name, commit), _ in filtered_repos:
            repo_commit_hash = get_repo_commit_hash(repo_name, commit)
            persist_dir = args.cache_dir / repo_commit_hash
            if persist_dir.exists():
                already_indexed.append((repo_name, commit))

        remaining_repos = [
            (k, v) for k, v in filtered_repos if k not in already_indexed
        ]

        if len(remaining_repos) == 0:
            print(f"\n✓ All {len(filtered_repos)} repos already indexed!")
            print(f"Use --force-rebuild if you want to re-index.")
            return

        batch_repos = remaining_repos[:args.batch_size]
        print(f"\nBatch indexing: {len(batch_repos)} repos")
        print(f"Remaining after this batch: {len(remaining_repos) - len(batch_repos)}")
        print()
        filtered_repos = batch_repos

    # Index each unique combination
    indexed_count = 0
    skipped_count = 0

    for (repo_name, commit), instance_ids in tqdm(
        filtered_repos,
        desc="Indexing repositories",
    ):
        repo_commit_hash = get_repo_commit_hash(repo_name, commit)
        persist_dir = args.cache_dir / repo_commit_hash

        # Check if already indexed
        if persist_dir.exists():
            try:
                # Verify index is valid
                search = SemanticSearch(
                    collection_name=f"code_{repo_commit_hash}",
                    persist_directory=str(persist_dir),
                )
                stats = search.get_stats()
                if stats["total_documents"] > 0:
                    tqdm.write(
                        f"✓ Skipping {repo_name}@{commit[:8]} "
                        f"({stats['total_documents']} docs, {len(instance_ids)} instances)"
                    )
                    skipped_count += 1
                    continue
            except Exception:
                # Index corrupted, rebuild
                tqdm.write(f"  Index corrupted, rebuilding...")

        # Clone repo
        try:
            repo_path = clone_repo_at_commit(repo_name, commit, args.clone_dir)
        except Exception as e:
            tqdm.write(f"✗ Failed to clone {repo_name}: {e}")
            continue

        # Create index
        try:
            device = "cuda" if args.gpu else "cpu"
            search = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(persist_dir),
                embedding_model_name="jinaai/jina-code-embeddings-0.5b",
                reranker_model_name="jinaai/jina-reranker-v3",
            )

            # Override device if needed
            if not args.gpu:
                search.embedder.device = "cpu"
                if search.reranker:
                    search.reranker.device = "cpu"

            stats = search.index_code_files(str(repo_path), file_extensions=[".py"])

            tqdm.write(
                f"✓ Indexed {repo_name}@{commit[:8]}: "
                f"{stats['indexed_files']} files, {stats['total_chunks']} chunks "
                f"({len(instance_ids)} instances will reuse)"
            )
            indexed_count += 1

        except Exception as e:
            tqdm.write(f"✗ Failed to index {repo_name}: {e}")
            continue

    print()
    print("=" * 80)
    print("Indexing Complete")
    print("=" * 80)
    print(f"Newly indexed: {indexed_count}")
    print(f"Already indexed (skipped): {skipped_count}")
    print(f"Processed this run: {indexed_count + skipped_count}")

    # Count total indices in cache
    total_in_cache = len(list(args.cache_dir.iterdir())) if args.cache_dir.exists() else 0
    print(f"Total indices in cache: {total_in_cache}")

    # Show remaining if batch mode
    if args.batch_size:
        total_unique = len(repo_commits)
        remaining = total_unique - total_in_cache
        if remaining > 0:
            print(f"\n⚠️  {remaining} repos still need indexing")
            print(f"   Run again to index next batch: python preindex_swebench.py --batch-size {args.batch_size}")
        else:
            print(f"\n✓ All {total_unique} unique repos indexed!")

    cache_size = sum(f.stat().st_size for f in args.cache_dir.rglob('*') if f.is_file()) / 1024**3
    print(f"\nCache size: {cache_size:.2f} GB")
    print(f"Cache location: {args.cache_dir}")
    print()
    print("Indices ready for training! Set generator.use_semantic_search=true")
    print()


if __name__ == "__main__":
    main()
