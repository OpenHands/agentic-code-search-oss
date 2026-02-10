import json
from pathlib import Path

from datasets import load_dataset


def main():
    # Configuration
    dataset_name = "princeton-nlp/SWE-bench_Verified"
    split = "test"
    search_results_path = Path(
        "/path/to/search_results.jsonl"
    )
    output_path = (
        Path(__file__).parent.parent
        / "data"
        / f"{dataset_name.split('/')[1]}_with_search_results.jsonl"
    )

    # Load the dataset
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    print(f"✓ Loaded {len(dataset)} instances\n")

    # Load JSONL file as a dataset
    print(f"Loading JSONL file: {search_results_path}")
    if not search_results_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {search_results_path}")

    by_instance_id: dict[str, object] = {}
    with search_results_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prediction = obj["test_result"]["reward"]["prediction"]
            by_instance_id[obj["instance_id"]] = {k: list(v) for k, v in prediction.items()}

    # Merge datasets on instance_id
    print("Merging datasets on instance_id...")
    search_results = [by_instance_id.get(i) for i in dataset["instance_id"]]
    merged_dataset = dataset.add_column(name="search_results", column=search_results)
    print("✓ Merged datasets\n")

    # Save as JSONL
    print(f"Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        for ex in merged_dataset:
            f.write(json.dumps(ex) + "\n")
    print(f"✓ Saved dataset with {len(merged_dataset)} instances to {output_path}")


if __name__ == "__main__":
    main()
