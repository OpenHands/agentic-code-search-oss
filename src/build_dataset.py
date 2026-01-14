import argparse
import os

from datasets import load_dataset


def format_target(file_changes):
    """
    Format file_changes into expected output format.
    """
    if file_changes is None or len(file_changes) == 0:
        return "```\n```"
    
    # Group by file, then by class
    file_to_classes = {}
    
    for change in file_changes:
        file_path = change["file"]
        changes = change["changes"]
        
        if file_path not in file_to_classes:
            file_to_classes[file_path] = {}
        
        # Get modules and entities, handling None and NumPy arrays
        edited_modules = changes.get("edited_modules")
        added_modules = changes.get("added_modules")
        edited_entities = changes.get("edited_entities")
        added_entities = changes.get("added_entities")
        
        # Convert to lists if not None
        modules = []
        if edited_modules is not None and len(edited_modules) > 0:
            modules.extend(list(edited_modules))
        if added_modules is not None and len(added_modules) > 0:
            modules.extend(list(added_modules))
        
        entities = []
        if edited_entities is not None and len(edited_entities) > 0:
            entities.extend(list(edited_entities))
        if added_entities is not None and len(added_entities) > 0:
            entities.extend(list(added_entities))
        
        # Extract classes from modules
        # CRITICAL: Only add if it's actually a class (has no dots in entity part)
        for module in modules:
            # Format: "file.py:ClassName" or incorrectly "file.py:function"
            if ":" in module:
                entity_part = module.split(":")[-1]
                
                # Only add if it's a class (no dot means it's a class or standalone function)
                # We need to check if this also appears as an entity with a dot
                # For now, we'll add all modules and let entities override
                if "." not in entity_part:
                    # Could be a class or standalone function
                    # Check if any entity has this as a class prefix
                    is_class = False
                    for entity in entities:
                        if ":" in entity:
                            e_part = entity.split(":")[-1]
                            if "." in e_part:
                                e_class = e_part.split(".")[0]
                                if e_class == entity_part:
                                    is_class = True
                                    break
                    
                    if is_class:
                        # It's a real class
                        if entity_part not in file_to_classes[file_path]:
                            file_to_classes[file_path][entity_part] = []
        
        # Extract entities and group by class
        for entity in entities:
            # Format: "file.py:ClassName.method" or "file.py:function"
            if ":" in entity:
                entity_part = entity.split(":")[-1]
                
                if "." in entity_part:
                    # Method: "ClassName.method"
                    class_name, func_name = entity_part.rsplit(".", 1)
                    
                    # Ensure class exists
                    if class_name not in file_to_classes[file_path]:
                        file_to_classes[file_path][class_name] = []
                    
                    # Add function to class
                    if func_name not in file_to_classes[file_path][class_name]:
                        file_to_classes[file_path][class_name].append(func_name)
                else:
                    # Standalone function
                    if None not in file_to_classes[file_path]:
                        file_to_classes[file_path][None] = []
                    
                    if entity_part not in file_to_classes[file_path][None]:
                        file_to_classes[file_path][None].append(entity_part)
    
    # Build output - each class followed immediately by its functions
    file_blocks = []
    
    for file_path in sorted(file_to_classes.keys()):
        lines = [file_path]
        
        classes = file_to_classes[file_path]
        
        # Process standalone functions first (if any)
        if None in classes:
            for func_name in sorted(classes[None]):
                lines.append(f"function: {func_name}")
        
        # Process each class with its methods immediately following
        for class_name in sorted([c for c in classes.keys() if c is not None]):
            lines.append(f"class: {class_name}")
            for func_name in sorted(classes[class_name]):
                lines.append(f"function: {func_name}")
        
        file_blocks.append("\n".join(lines))
    
    return "```\n" + "\n\n".join(file_blocks) + "\n```"


def main():
    parser = argparse.ArgumentParser(description="Build dataset from patches")
    parser.add_argument(
        "--dataset", 
        default="adityasoni17/SWE-Gym-code-search", 
        help="Input dataset path"
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--output", required=True, help="Output file path")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.dataset} ({args.split} split)...")
    dataset = load_dataset(args.dataset, split=args.split).to_pandas()
    print(f"Loaded {len(dataset)} examples")

    # Build target from file_changes
    print("Formatting targets...")
    dataset["target"] = dataset["file_changes"].apply(format_target)
    
    # Build prompts
    print("Building prompts...")
    dataset["prompt"] = dataset.apply(
        lambda row: [{"role": "user", "content": row["problem_statement"]}], axis=1
    )

    # Shuffle and split (keep last 100 for validation)
    print("Shuffling and splitting...")
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    train_dataset = dataset.iloc[:-100]
    validation_dataset = dataset.iloc[-100:]
    
    print(f"Train: {len(train_dataset)}, Validation: {len(validation_dataset)}")

    # Save
    output_dir = os.path.join(
        args.output, 
        args.dataset.replace("/", "__") + "_" + args.split
    )
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")
    
    print(f"Saving to {output_dir}...")
    train_dataset.to_parquet(train_path)
    validation_dataset.to_parquet(val_path)
    
    print("Done!")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    
    # Print a few examples to verify
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS (verify correctness):")
    print("="*80)
    for i in range(min(3, len(train_dataset))):
        print(f"\n--- Example {i+1} ---")
        print("Problem Statement (truncated):")
        print(train_dataset.iloc[i]["problem_statement"][:150] + "...")
        print("\nFile Changes:")
        print(train_dataset.iloc[i]["file_changes"])
        print("\nTarget:")
        print(train_dataset.iloc[i]["target"])
        print()


if __name__ == "__main__":
    main()