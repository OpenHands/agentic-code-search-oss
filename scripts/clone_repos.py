import argparse
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
import polars as pl
from tqdm import tqdm


def _run(cmd: list[str], *, verbose: bool, **kwargs):
    if verbose:
        return subprocess.run(cmd, check=True, **kwargs)
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def fetch_commits(
    repo_name: str,
    base_commits: list[str],
    output_dir: Path,
    verbose: bool,
) -> tuple[Path, Path, list[str]]:
    """
    Fetch a list of commits from a repository.
    """
    repo_slug = repo_name.replace("/", "__")
    repo_dir = output_dir / repo_slug
    repo_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "cache" / f"{repo_slug}.git"
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        _run(
            ["git", "init", "--bare", str(cache_dir)],
            verbose=verbose,
        )
        _run(
            [
                "git",
                "-C",
                str(cache_dir),
                "remote",
                "add",
                "origin",
                f"https://github.com/{repo_name}.git",
            ],
            verbose=verbose,
        )

    _run(
        [
            "git",
            "-C",
            str(cache_dir),
            "fetch",
            "--depth",
            "1",
            "origin",
            *base_commits,
        ],
        verbose=verbose,
    )
    return cache_dir, repo_dir, base_commits


def export_repo(output_dir: Path, repo_name: str, base_commit: str, *, verbose: bool) -> bool:
    """
    Export a repository at a specific commit.
    """
    repo_slug = repo_name.replace("/", "__")
    cache_dir = output_dir / "cache" / f"{repo_slug}.git"
    repo_dir = output_dir / repo_slug
    out_dir = repo_dir / base_commit
    out_dir.mkdir(parents=True, exist_ok=False)

    git_archive = subprocess.Popen(
        ["git", "-C", str(cache_dir), "archive", base_commit],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verbose:
        subprocess.run(
            ["tar", "-x", "-C", str(out_dir)],
            check=True,
            stdin=git_archive.stdout,
        )
    else:
        subprocess.run(
            ["tar", "-x", "-C", str(out_dir)],
            check=True,
            stdin=git_archive.stdout,
            capture_output=True,
            text=True,
        )
    git_archive.stdout.close()
    git_archive_stderr = git_archive.stderr.read()
    git_archive.stderr.close()
    git_archive_rc = git_archive.wait()
    if git_archive_rc != 0:
        raise subprocess.CalledProcessError(
            git_archive_rc,
            git_archive.args,
            stderr=git_archive_stderr.decode("utf-8", errors="replace"),
        )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clone repos from the input dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for cloned repos (default: ./swebench_repos)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adityasoni17/SWE-bench_Lite-code-search",
        help="Dataset to use (default: adityasoni17/SWE-bench_Lite-code-search)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent clone operations (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print git/tar output to the terminal",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        output_dir = (project_root / "repos").resolve()
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    dataset = (
        list(load_dataset(args.dataset, columns=["repo", "base_commit"]).values())[0] # get the first split
        .to_polars()
        .unique()
    )

    print(f"\nProcessing {len(dataset)} repo instances")
    print(f"Using {args.max_workers} concurrent workers")
    print("=" * 80)

    # Clone each instance concurrently
    total_instances = len(dataset)
    grouped = (
        dataset.group_by("repo")
        .agg(pl.col("base_commit").unique().alias("base_commits"))
    )

    successful = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for _ in tqdm(
            executor.map(
                lambda row: fetch_commits(
                    row["repo"],
                    row["base_commits"],
                    output_dir,
                    args.verbose,
                ),
                grouped.iter_rows(named=True),
            ),
            total=len(grouped),
            desc="Fetching repos",
        ):
            pass

    total_exports = total_instances
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = executor.map(
            lambda row: export_repo(
                output_dir,
                row["repo"],
                row["base_commit"],
                verbose=args.verbose,
            ),
            dataset.iter_rows(named=True),
        )
        for ok in tqdm(results, total=total_exports, desc="Exporting commits"):
            if ok:
                successful += 1

    print("Removing cache...")
    shutil.rmtree(output_dir / "cache")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Successfully cloned: {successful}/{total_instances} instances")
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
