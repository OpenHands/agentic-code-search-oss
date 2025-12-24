#!/usr/bin/env python3
"""
Clone and index repositories in a single pass.

This combines the functionality of clone_repos.py and preindex_swebench.py,
cloning each repository and creating semantic search indices at the same time.

Supports both SWE-Gym and SWE-bench datasets.

Usage:
    # SWE-Gym (default)
    python scripts/clone_and_index_repos.py --output-dir /data/user_data/sanidhyv/grep

    # SWE-bench Lite
    python scripts/clone_and_index_repos.py \
        --dataset princeton-nlp/SWE-bench_Lite \
        --split test
"""

import argparse
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict
import shutil
from datasets import load_dataset
from tqdm import tqdm

from src.tools.semantic_search import SemanticSearch


def get_repo_commit_hash(repo_name: str, commit: str) -> str:
    """Get unique hash for (repo, commit) pair."""
    key = f"{repo_name}:{commit}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def clone_instance(
    repo_name: str,
    commit_id: str,
    instance_id: str,
    output_dir: Path,
    cache_dir: Path,
    use_gpu: bool = False,
    skip_indexing: bool = False,
) -> tuple[bool, bool]:
    """
    Clone a repository at a specific commit and create semantic search index.

    Args:
        repo_name: Repository name in format 'owner/repo'
        commit_id: Commit hash to checkout
        instance_id: Instance ID for directory naming
        output_dir: Base output directory for clones
        cache_dir: Directory to store indices
        use_gpu: Whether to use GPU for indexing
        skip_indexing: If True, only clone without indexing

    Returns:
        (clone_success, index_success) tuple
    """
    # Create instance directory name
    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = output_dir / instance_dir_name

    # Clone if needed
    clone_existed = instance_path.exists()
    if not clone_existed:
        try:
            # Clone the repository
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    f"https://github.com/{repo_name}.git",
                    str(instance_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # Checkout the specific commit
            subprocess.run(
                ["git", "-C", str(instance_path), "checkout", "--quiet", commit_id],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return False, False

    # Index the repository
    if skip_indexing:
        return True, True

    repo_commit_hash = get_repo_commit_hash(repo_name, commit_id)
    persist_dir = cache_dir / repo_commit_hash
    ready_file = persist_dir / ".ready"
    if ready_file.exists():  # CHANGE THIS CHECK
        try:
            search = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(persist_dir),
            )
            stats = search.get_stats()
            if stats["total_documents"] > 0:
                index_existed = True
                return True, True  # Already indexed successfully
        except Exception:
            # Index corrupted, will rebuild
            pass
    # Create index
    try:
        device = "cuda" if use_gpu else "cpu"
        search = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(persist_dir),
            embedding_model_name="jinaai/jina-code-embeddings-0.5b",
            reranker_model_name="jinaai/jina-reranker-v3",
        )

        if not use_gpu:
            search.embedder.device = "cpu"
            if search.reranker:
                search.reranker.device = "cpu"

        stats = search.index_code_files(str(instance_path), file_extensions=[".py"])

        # ADD THIS: Create .ready marker after successful indexing
        if stats["total_chunks"] > 0:
            ready_file.touch()
            print(f"✓ Created .ready marker for {repo_commit_hash}")
            return True, True
        else:
            print(f"✗ No chunks indexed for {repo_commit_hash}")
            return True, False

    except Exception as e:
        print(f"✗ Indexing failed for {repo_commit_hash}: {e}")
        # Clean up partial index
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
        return True, False

    except Exception as e:
        # Clone succeeded but indexing failed
        return True, False


def main():
    parser = argparse.ArgumentParser(
        description="Clone and index repositories from SWE-bench dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./swebench_repos",
        help="Directory to clone repositories into (default: ./swebench_repos)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "swebench_indices",
        help="Directory to store indices (default: ~/.cache/swebench_indices)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SWE-Gym/SWE-Gym",
        help="Dataset to use (default: SWE-Gym/SWE-Gym, or use princeton-nlp/SWE-bench_Lite)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of instances to process (for testing)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Maximum number of repositories to clone (for testing)",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Only clone repositories without creating indices",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for embedding (requires sufficient VRAM)",
    )
    parser.add_argument(
        "--show-fields",
        action="store_true",
        help="Show available fields in the dataset and exit",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Repository Clone & Index")
    print("=" * 80)
    print(f"Dataset: {args.dataset} (split: {args.split})")
    print(f"Clone directory: {output_dir.absolute()}")
    if not args.skip_indexing:
        print(f"Index cache directory: {args.cache_dir}")
        print(f"Embedding device: {'GPU' if args.gpu else 'CPU'}")
    print()

    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"✓ Loaded {len(dataset)} instances")

    # Show available fields if requested
    if args.show_fields:
        print("\n" + "=" * 80)
        print("Available fields in dataset:")
        print("=" * 80)
        if len(dataset) > 0:
            first_instance = dataset[0]
            for key in sorted(first_instance.keys()):
                value = first_instance[key]
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"{key:25s}: {value_str}")
        print("=" * 80)
        return

    # Detect field names (SWE-Gym vs SWE-bench)
    first_instance = dataset[0]

    # SWE-Gym uses 'repo_name' and 'instance_id', SWE-bench uses 'repo' and 'instance_id'
    repo_field = "repo_name" if "repo_name" in first_instance else "repo"
    commit_field = "base_commit"
    id_field = "instance_id"

    print(f"Detected field names: repo={repo_field}, commit={commit_field}, id={id_field}")
    print()

    # Collect all instances to process
    instances_to_process = []
    repo_commit_instances = defaultdict(list)

    for instance in dataset:
        instances_to_process.append(
            {
                "repo": instance[repo_field],
                "instance_id": instance[id_field],
                "base_commit": instance[commit_field],
            }
        )
        # Track which instances share (repo, commit)
        key = (instance[repo_field], instance[commit_field])
        repo_commit_instances[key].append(instance[id_field])

    # Apply max-repos filter
    if args.max_repos:
        repos_seen = set()
        filtered_instances = []
        for instance in instances_to_process:
            if instance["repo"] not in repos_seen:
                if len(repos_seen) >= args.max_repos:
                    continue
                repos_seen.add(instance["repo"])
            if instance["repo"] in repos_seen:
                filtered_instances.append(instance)
        instances_to_process = filtered_instances
        print(f"\n(Limited to {args.max_repos} repositories)")

    # Apply max-instances filter
    if args.max_instances:
        instances_to_process = instances_to_process[: args.max_instances]
        print(f"(Limited to {args.max_instances} instances)")

    print(f"\nProcessing {len(instances_to_process)} instances")
    print(f"Unique (repo, commit) pairs: {len(repo_commit_instances)}")
    if not args.skip_indexing:
        print("(Each unique pair will be indexed once)")
    print("=" * 80)

    # Track results
    clones_successful = 0
    clones_existed = 0
    indices_created = 0
    indices_existed = 0
    indices_failed = 0

    # Process each instance
    for instance in tqdm(instances_to_process, desc="Cloning and indexing"):
        instance_path = (
            output_dir / f"{instance['repo'].replace('/', '_')}_{instance['instance_id']}"
        )
        clone_existed_before = instance_path.exists()

        clone_success, index_success = clone_instance(
            instance["repo"],
            instance["base_commit"],
            instance["instance_id"],
            output_dir,
            args.cache_dir,
            use_gpu=args.gpu,
            skip_indexing=args.skip_indexing,
        )

        if clone_success:
            if clone_existed_before:
                clones_existed += 1
            else:
                clones_successful += 1

            if not args.skip_indexing:
                # Check if index was newly created
                repo_commit_hash = get_repo_commit_hash(
                    instance["repo"], instance["base_commit"]
                )
                persist_dir = args.cache_dir / repo_commit_hash

                if index_success:
                    # Check if this is the first instance for this (repo, commit)
                    key = (instance["repo"], instance["base_commit"])
                    if instance["instance_id"] == repo_commit_instances[key][0]:
                        # First instance created the index
                        indices_created += 1
                    else:
                        # Subsequent instance reused existing index
                        indices_existed += 1
                else:
                    indices_failed += 1

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    total = len(instances_to_process)
    print(
        f"Successfully cloned: {clones_successful}/{total} new instances "
        f"({clones_existed} already existed)"
    )

    if not args.skip_indexing:
        print(f"\nIndices created: {indices_created}")
        print(f"Indices reused: {indices_existed + len(instances_to_process) - indices_created - indices_failed}")
        if indices_failed > 0:
            print(f"Indexing failed: {indices_failed}")

        # Count total indices in cache
        total_in_cache = (
            len(list(args.cache_dir.iterdir())) if args.cache_dir.exists() else 0
        )
        cache_size = (
            sum(f.stat().st_size for f in args.cache_dir.rglob("*") if f.is_file())
            / 1024**3
        )
        print(f"\nTotal indices in cache: {total_in_cache}")
        print(f"Cache size: {cache_size:.2f} GB")
        print(f"Cache location: {args.cache_dir}")

    print(
        "\nNote: Each instance is in its own directory named <repo>_<instance_id>"
    )
    if not args.skip_indexing:
        print(
            "\nIndices ready for training! Set generator.use_semantic_search=true"
        )
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()