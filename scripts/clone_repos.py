import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def clone_instance(
    repo_name: str, commit_id: str, instance_id: str, output_dir: Path
) -> bool:
    """
    Clone a repository at a specific commit into a separate directory.

    Args:
        repo_name: Repository name in format 'owner/repo'
        commit_id: Commit hash to checkout
        instance_id: Instance ID for directory naming
        output_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    # Create instance directory name: repo_instance-id
    # E.g., astropy_astropy-12907
    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = output_dir / instance_dir_name

    # Skip if already exists
    if instance_path.exists():
        return True

    try:
        # Clone the repository
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                str(instance_path),
                "--depth",
                "1",
                "--revision",
                commit_id,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        return True
    except subprocess.CalledProcessError as e:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clone repositories from SWE-bench dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./swebench_repos",
        help="Directory to clone repositories into (default: ./swebench_repos)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset to use (default: princeton-nlp/SWE-bench_Lite)",
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
        "--show-fields",
        action="store_true",
        help="Show available fields in the dataset and exit",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent clone operations (default: 4)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SWE-bench dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test")
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
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"{key:25s}: {value_str}")
        print("=" * 80)
        return

    # Collect all instances to process
    instances_to_process = []
    for instance in dataset:
        instances_to_process.append(
            {
                "repo": instance["repo"],
                "instance_id": instance["instance_id"],
                "base_commit": instance["base_commit"],
            }
        )

    # Apply max-repos filter
    if args.max_repos:
        # Group by repo and take first N repos
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
    print(f"Using {args.max_workers} concurrent workers")
    print("=" * 80)

    # Clone each instance concurrently
    successful = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(
                clone_instance,
                instance["repo"],
                instance["base_commit"],
                instance["instance_id"],
                output_dir,
            ): instance
            for instance in instances_to_process
        }

        # Process completed tasks with progress bar
        for future in tqdm(
            as_completed(future_to_instance),
            total=len(instances_to_process),
            desc="Cloning instances",
        ):
            if future.result():
                successful += 1

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    total = len(instances_to_process)
    print(f"Successfully cloned: {successful}/{total} instances")
    print(
        "Note: Each instance is in its own directory named <repo>_<instance_id>"
    )
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
