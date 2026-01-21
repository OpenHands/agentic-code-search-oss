import argparse
import os

from datasets import load_dataset

from src.utils.dataset import extract_functions_from_patch


def main():
    parser = argparse.ArgumentParser(description="Build dataset from patches")
    parser.add_argument("--dataset", default="SWE-Gym/SWE-Gym", help="Input dataset path")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--output", required=True, help="Output file path for processed dataset")
    parser.add_argument("--use_patch", action="store_true", help="Whether to use patches to extract target functions")
    args = parser.parse_args()

    # Load and process dataset
    dataset = load_dataset(args.dataset, split=args.split).to_pandas()

    dataset["target"] = dataset.apply(
        lambda row: f"{extract_functions_from_patch(row['patch'])}", axis=1
    )

    # Remove rows with empty problem_statement
    dataset = dataset[dataset["problem_statement"].str.strip().astype(bool)]

    dataset["prompt"] = dataset.apply(
        lambda row: [{"role": "user", "content": row["problem_statement"]}], axis=1
    )

    if args.use_patch:
        dataset["use_patch"] = True
    else:
        dataset["use_patch"] = False

    # Drop "PASS_TO_PASS" and "FAIL_TO_PASS" columns
    try:
        dataset = dataset.drop(columns=["PASS_TO_PASS", "FAIL_TO_PASS"])
    except KeyError:
        pass

    # shuffle dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # train_size = int(0.975 * len(dataset))
    train_dataset = dataset.iloc[:-100]
    validation_dataset = dataset.iloc[-100:]

    # if output does not exist, create it
    output_dir = os.path.join(args.output, args.dataset.replace("/", "__") + "_" + args.split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "train.parquet")
    train_dataset.to_parquet(output_path)

    output_path = os.path.join(output_dir, "validation.parquet")
    validation_dataset.to_parquet(output_path)


if __name__ == "__main__":
    main()