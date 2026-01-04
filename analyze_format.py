"""Analyze rollout JSON formatting issues across training steps.

Given a root folder like:
  /data/user_data/.../instruct_trajectoryleveltrajectories

We expect:
  step_*/train/*.json
(and sometimes non-json sidecar files like .error/.errors which are ignored).

For each step, we compute and plot:
1) % of rollouts at each step where parsed_final_message is "empty or incorrectly formatted".
2) % of rollouts at each step where parsed_final_message is empty or incorrectly formatted AND total_reward > 0.
3) % of rollout-groups at each step where at least one rollout in group is empty or incorrectly formatted.
4) % of rollout-groups at each step where all available rollouts in group are empty or incorrectly formatted.

"Bad" parsed_final_message means:
- empty / whitespace-only string, OR
- contains "<tool_call>" or "</tool_call>".

Outputs: PNG plots saved under --out-dir (default: ./format_plots).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_BAD_TOOL_CALL_RE = re.compile(r"<tool_call>|</tool_call>")


def _step_index(step_name: str) -> int:
    """Extract numeric suffix from a step directory name like 'step_14'.

    Returns a large value for unparseable names so they sort last.
    """
    m = re.match(r"^step_(\d+)$", step_name)
    if not m:
        return 10**18
    return int(m.group(1))


def is_bad_parsed_final_message(msg: Optional[str]) -> bool:
    if msg is None:
        return True
    if msg.strip() == "":
        return True
    return _BAD_TOOL_CALL_RE.search(msg) is not None


def rollout_group_key(file_path: Path) -> str:
    """Group tweepy__..._0.json, tweepy__..._1.json -> same key.

    We strip the trailing _<digits>.json suffix when present.
    """
    name = file_path.name
    return re.sub(r"_\d+\.json$", ".json", name)


@dataclass
class FileStats:
    bad: bool
    reward_gt_zero: bool


@dataclass
class StepStats:
    step_name: str
    file_stats: List[FileStats]
    group_to_stats: Dict[str, List[FileStats]]

    def pct(self, numer: int, denom: int) -> float:
        return 0.0 if denom == 0 else 100.0 * numer / denom

    def percent_bad_files(self) -> float:
        denom = len(self.file_stats)
        numer = sum(1 for s in self.file_stats if s.bad)
        return self.pct(numer, denom)

    def percent_bad_files_reward_gt_zero(self) -> float:
        denom = len(self.file_stats)
        numer = sum(1 for s in self.file_stats if s.bad and s.reward_gt_zero)
        if numer > 0:
            print(f"Step {self.step_name}: {numer} groups with any bad formatting")
        return self.pct(numer, denom)

    def percent_groups_any_bad(self) -> float:
        denom = len(self.group_to_stats)
        numer = sum(
            1 for stats in self.group_to_stats.values() if any(s.bad for s in stats)
        )        
        return self.pct(numer, denom)

    def percent_groups_all_bad(self) -> float:
        denom = len(self.group_to_stats)
        numer = sum(
            1 for stats in self.group_to_stats.values() if all(s.bad for s in stats)
        )
        return self.pct(numer, denom)


def iter_step_train_jsons(root: Path) -> Iterable[Tuple[str, Path]]:
    for step_dir in sorted(root.glob("step_*"), key=lambda p: _step_index(p.name)):
        if not step_dir.is_dir():
            continue
        train_dir = step_dir / "train"
        if not train_dir.is_dir():
            continue
        for p in sorted(train_dir.glob("*.json")):
            yield step_dir.name, p


def read_one_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_stats(root: Path) -> List[StepStats]:
    step_to_file_stats: Dict[str, List[FileStats]] = {}
    step_to_group_stats: Dict[str, Dict[str, List[FileStats]]] = {}

    for step_name, json_path in iter_step_train_jsons(root):
        try:
            data = read_one_json(json_path)
        except Exception:
            # If a json is corrupted, treat it as "bad" formatting.
            fs = FileStats(bad=True, reward_gt_zero=False)
        else:
            bad = is_bad_parsed_final_message(data.get("parsed_final_message"))
            reward = data.get("total_reward")
            reward_gt_zero = False
            try:
                reward_gt_zero = float(reward) > 0.0
            except Exception:
                reward_gt_zero = False
            fs = FileStats(bad=bad, reward_gt_zero=reward_gt_zero)

        step_to_file_stats.setdefault(step_name, []).append(fs)
        gk = rollout_group_key(json_path)
        step_to_group_stats.setdefault(step_name, {}).setdefault(gk, []).append(fs)

    out: List[StepStats] = []
    for step_name in sorted(step_to_file_stats.keys(), key=_step_index):
        out.append(
            StepStats(
                step_name=step_name,
                file_stats=step_to_file_stats[step_name],
                group_to_stats=step_to_group_stats.get(step_name, {}),
            )
        )
    return out


def plot_series(
    *,
    steps: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(steps)), values, marker="o")
    # plt.xticks(range(len(steps)), [], rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("step")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze parsed_final_message formatting across steps"
    )
    parser.add_argument(
        "root",
        help="Path to instruct_trajectoryleveltrajectories (contains step_* subdirs)",
    )
    parser.add_argument(
        "--out-dir",
        default="format_plots",
        help="Directory to save plots (default: format_plots)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    stats = compute_stats(root)
    if not stats:
        raise SystemExit(
            "No stats computed. Expected step_*/train/*.json under the given root."
        )

    steps = [s.step_name for s in stats]

    s1 = [s.percent_bad_files() for s in stats]
    s2 = [s.percent_bad_files_reward_gt_zero() for s in stats]
    s3 = [s.percent_groups_any_bad() for s in stats]
    s4 = [s.percent_groups_all_bad() for s in stats]

    plot_series(
        steps=steps,
        values=s1,
        title="% json files with incorrectly formatted/empty parsed_final_message",
        ylabel="percent",
        out_path=out_dir / "stat1_bad_files.png",
    )
    plot_series(
        steps=steps,
        values=s2,
        title="% json files with incorrectly formatted/empty parsed_final_message AND total_reward > 0",
        ylabel="percent",
        out_path=out_dir / "stat2_bad_files_reward_gt_zero.png",
    )
    plot_series(
        steps=steps,
        values=s3,
        title="% rollout groups where ANY rollout has incorrectly formatted/empty parsed_final_message",
        ylabel="percent",
        out_path=out_dir / "stat3_groups_any_bad.png",
    )
    plot_series(
        steps=steps,
        values=s4,
        title="% rollout groups where ALL rollouts have incorrectly formatted/empty parsed_final_message",
        ylabel="percent",
        out_path=out_dir / "stat4_groups_all_bad.png",
    )

    # Small textual summary for convenience.
    print("step\tbad_files%\tbad_files_reward>0%\tgroups_any_bad%\tgroups_all_bad%")
    for step, a, b, c, d in zip(steps, s1, s2, s3, s4):
        print(f"{step}\t{a:.2f}\t{b:.2f}\t{c:.2f}\t{d:.2f}")
    print(f"\nSaved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
