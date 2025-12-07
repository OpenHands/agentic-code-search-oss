import argparse
import subprocess
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
        print(f"  ✓ Instance {instance_id} already exists")
        return True, instance_path

    try:
        # Clone the repository
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                str(instance_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Checkout the specific commit
        subprocess.run(
            ["git", "-C", str(instance_path), "checkout", commit_id],
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"  ✓ Cloned {instance_id} at commit {commit_id[:8]}")
        return True, instance_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error cloning {instance_id}: {e.stderr}")
        return False, None
