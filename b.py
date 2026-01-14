"""
Test script to verify GT extraction, prediction parsing, and reward calculation.
Run this to ensure all formats align correctly.
"""

import sys
from datasets import load_dataset
from src.rewards.file_localization.file_localization import (
    extract_ground_truth_from_file_changes,
    multilevel_localization_f1_reward,
)
from src.rewards.file_localization.module_rewards import get_simple_results_from_raw_outputs


def test_gt_extraction():
    """Test ground truth extraction from file_changes."""
    print("="*80)
    print("TEST 1: Ground Truth Extraction")
    print("="*80)
    
    # Load a few examples
    ds = load_dataset("adityasoni17/SWE-Gym-code-search", split="train")
    
    for i in range(3):
        example = ds[i]
        print(f"\n--- Example {i}: {example['instance_id']} ---")
        
        # Extract GT
        gt_data = extract_ground_truth_from_file_changes(example["file_changes"])
        
        print("\nFile Changes (raw):")
        for fc in example["file_changes"]:
            print(f"  File: {fc['file']}")
            for key in ['edited_modules', 'added_modules', 'edited_entities', 'added_entities']:
                val = fc['changes'].get(key)
                if val is not None and len(val) > 0:
                    print(f"    {key}: {list(val)}")
        
        print("\nExtracted GT:")
        print(f"  Files ({len(gt_data['files'])}):")
        for f in sorted(gt_data['files']):
            print(f"    {f}")
        print(f"  Modules ({len(gt_data['modules'])}):")
        for m in sorted(gt_data['modules']):
            print(f"    {m}")
        print(f"  Entities ({len(gt_data['entities'])}):")
        for e in sorted(gt_data['entities']):
            print(f"    {e}")
    
    print("\n✅ GT extraction test complete\n")


def test_prediction_parsing():
    """Test prediction parsing from various output formats."""
    print("="*80)
    print("TEST 2: Prediction Parsing")
    print("="*80)
    
    test_cases = [
        {
            "name": "Format 1: Multiple methods in same class",
            "output": """```
moto/dynamodb/models/dynamo_type.py
class: DynamoType
function: __add__

class: Item
function: update_with_attribute_updates
```````````""",
            "expected": {
                "files": {"moto/dynamodb/models/dynamo_type.py"},
                "modules": {"moto/dynamodb/models/dynamo_type.py:DynamoType",
                           "moto/dynamodb/models/dynamo_type.py:Item"},
                "entities": {"moto/dynamodb/models/dynamo_type.py:DynamoType.__add__",
                            "moto/dynamodb/models/dynamo_type.py:Item.update_with_attribute_updates"}
            }
        },
        {
            "name": "Format 2: Standalone function",
            "output": """```
test_file.py
function: global_function
``````````""",
            "expected": {
                "files": {"test_file.py"},
                "modules": set(),  # No class = no module
                "entities": {"test_file.py:global_function"}
            }
        },
        {
            "name": "Format 3: File only (no functions)",
            "output": """```
test_file.py
`````````""",
            "expected": {
                "files": {"test_file.py"},
                "modules": set(),
                "entities": set()
            }
        },
        {
            "name": "Format 4: Mixed - class methods + standalone function",
            "output": """```
test_file.py
class: MyClass
function: method1
function: method2

other_file.py
function: standalone_func
````````""",
            "expected": {
                "files": {"test_file.py", "other_file.py"},
                "modules": {"test_file.py:MyClass"},
                "entities": {"test_file.py:MyClass.method1",
                            "test_file.py:MyClass.method2",
                            "other_file.py:standalone_func"}
            }
        },
        {
            "name": "Format 5: Multiple files with classes",
            "output": """```
src/models/user.py
class: User
function: save
function: validate

src/views/user_view.py
class: UserView
function: render
```````""",
            "expected": {
                "files": {"src/models/user.py", "src/views/user_view.py"},
                "modules": {"src/models/user.py:User", "src/views/user_view.py:UserView"},
                "entities": {"src/models/user.py:User.save",
                            "src/models/user.py:User.validate",
                            "src/views/user_view.py:UserView.render"}
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print("Input:")
        print(test_case['output'])
        
        # Parse using the function
        results = get_simple_results_from_raw_outputs(test_case['output'])
        
        # Convert to sets like the reward function does
        pred_files = set()
        pred_modules = set()
        pred_entities = set()
        
        for result in results:
            file_path = result.get("file") or ""
            if file_path:
                file_path = file_path.strip()
            if not file_path:
                continue
            
            pred_files.add(file_path)
            
            class_name = result.get("class") or ""
            if class_name:
                class_name = class_name.strip()
            
            func_name = result.get("function") or ""
            if func_name:
                func_name = func_name.strip()
            
            # ONLY add to modules if there's a class
            if class_name:
                module = f"{file_path}:{class_name}"
                pred_modules.add(module)
            
            # Add entity if there's a function
            if func_name:
                if class_name:
                    entity = f"{file_path}:{class_name}.{func_name}"
                else:
                    entity = f"{file_path}:{func_name}"
                pred_entities.add(entity)
        
        print("\nParsed:")
        print(f"  Files: {pred_files}")
        print(f"  Modules: {pred_modules}")
        print(f"  Entities: {pred_entities}")
        
        print("\nExpected:")
        print(f"  Files: {test_case['expected']['files']}")
        print(f"  Modules: {test_case['expected']['modules']}")
        print(f"  Entities: {test_case['expected']['entities']}")
        
        # Verify
        assert pred_files == test_case['expected']['files'], f"Files mismatch! Got {pred_files}, expected {test_case['expected']['files']}"
        assert pred_modules == test_case['expected']['modules'], f"Modules mismatch! Got {pred_modules}, expected {test_case['expected']['modules']}"
        assert pred_entities == test_case['expected']['entities'], f"Entities mismatch! Got {pred_entities}, expected {test_case['expected']['entities']}"
        
        print("\n✅ PASS")
    
    print("\n✅ All prediction parsing tests passed\n")


def test_end_to_end_reward():
    """Test end-to-end reward calculation with real data."""
    print("="*80)
    print("TEST 3: End-to-End Reward Calculation")
    print("="*80)
    
    # Load one example
    ds = load_dataset("adityasoni17/SWE-Gym-code-search", split="train")
    example = ds[0]
    
    print(f"\nInstance: {example['instance_id']}")
    print(f"Problem: {example['problem_statement'][:200]}...")
    
    # Extract GT
    gt_data = extract_ground_truth_from_file_changes(example["file_changes"])
    
    print(f"\nGround Truth:")
    print(f"  Files: {sorted(gt_data['files'])}")
    print(f"  Modules: {sorted(gt_data['modules'])}")
    print(f"  Entities: {sorted(gt_data['entities'])}")
    
    # Create a simulated agent output (perfect prediction)
    # Group entities by file, then by class
    simulated_output = "```\n"
    
    file_to_classes = {}
    for entity in sorted(gt_data['entities']):
        # Parse entity: "file.py:Class.method" or "file.py:function"
        file_part, entity_part = entity.split(":", 1)
        
        if file_part not in file_to_classes:
            file_to_classes[file_part] = {}
        
        if "." in entity_part:
            # Method: "ClassName.method"
            class_name, func_name = entity_part.rsplit(".", 1)
            if class_name not in file_to_classes[file_part]:
                file_to_classes[file_part][class_name] = []
            file_to_classes[file_part][class_name].append(func_name)
        else:
            # Standalone function
            if None not in file_to_classes[file_part]:
                file_to_classes[file_part][None] = []
            file_to_classes[file_part][None].append(entity_part)
    
    # Build output - each class followed immediately by its functions
    for file_path in sorted(file_to_classes.keys()):
        simulated_output += f"{file_path}\n"
        
        classes = file_to_classes[file_path]
        
        # Process standalone functions first (if any)
        if None in classes:
            for func_name in sorted(classes[None]):
                simulated_output += f"function: {func_name}\n"
        
        # Process each class with its methods immediately following
        for class_name in sorted([c for c in classes.keys() if c is not None]):
            simulated_output += f"class: {class_name}\n"
            for func_name in sorted(classes[class_name]):
                simulated_output += f"function: {func_name}\n"
            # Blank line after each class
            if class_name != sorted([c for c in classes.keys() if c is not None])[-1]:
                simulated_output += "\n"
        
        simulated_output += "\n"
    
    simulated_output += "```"
    
    print(f"\nSimulated Agent Output (perfect prediction):")
    print(simulated_output)
    
    # Compute reward using final_message (not messages list)
    reward, metrics = multilevel_localization_f1_reward(
        final_message=simulated_output,
        instance=example
    )
    
    print(f"\nReward Calculation:")
    print(f"  File F1: {metrics['file_f1']:.3f}")
    print(f"  Module F1: {metrics['module_f1']:.3f}")
    print(f"  Entity F1: {metrics['entity_f1']:.3f}")
    print(f"  Total Reward: {reward:.3f}")
    
    # Also show what was parsed
    results = get_simple_results_from_raw_outputs(simulated_output)
    pred_files = set()
    pred_modules = set()
    pred_entities = set()
    
    for result in results:
        file_path = result.get("file") or ""
        if file_path:
            pred_files.add(file_path.strip())
        
        class_name = result.get("class") or ""
        func_name = result.get("function") or ""
        
        if class_name:
            pred_modules.add(f"{file_path}:{class_name.strip()}")
        
        if func_name:
            if class_name:
                pred_entities.add(f"{file_path}:{class_name.strip()}.{func_name.strip()}")
            else:
                pred_entities.add(f"{file_path}:{func_name.strip()}")
    
    print(f"\nPredicted vs GT:")
    print(f"  Pred Files: {sorted(pred_files)}")
    print(f"  GT Files: {sorted(gt_data['files'])}")
    print(f"  Pred Modules: {sorted(pred_modules)}")
    print(f"  GT Modules: {sorted(gt_data['modules'])}")
    print(f"  Pred Entities: {sorted(pred_entities)}")
    print(f"  GT Entities: {sorted(gt_data['entities'])}")
    
    # Verify perfect prediction
    assert metrics['file_f1'] == 1.0, f"File F1 should be 1.0, got {metrics['file_f1']}"
    assert metrics['module_f1'] == 1.0, f"Module F1 should be 1.0, got {metrics['module_f1']}"
    assert metrics['entity_f1'] == 1.0, f"Entity F1 should be 1.0, got {metrics['entity_f1']}"
    
    print("\n✅ Perfect prediction correctly scored!")
    
    # Test partial prediction
    print("\n--- Testing Partial Prediction ---")
    
    # Only predict first entity
    first_entity = sorted(gt_data['entities'])[0]
    file_part, entity_part = first_entity.split(":", 1)
    
    partial_output = f"```\n{file_part}\n"
    
    if "." in entity_part:
        class_name, func_name = entity_part.rsplit(".", 1)
        partial_output += f"class: {class_name}\n"
        partial_output += f"function: {func_name}\n"
    else:
        partial_output += f"function: {entity_part}\n"
    
    partial_output += "```"
    
    print(f"Partial Output (first entity only):")
    print(partial_output)
    
    reward_partial, metrics_partial = multilevel_localization_f1_reward(
        final_message=partial_output,
        instance=example
    )
    
    print(f"\nPartial Prediction Metrics:")
    print(f"  File F1: {metrics_partial['file_f1']:.3f}")
    print(f"  Module F1: {metrics_partial['module_f1']:.3f}")
    print(f"  Entity F1: {metrics_partial['entity_f1']:.3f}")
    print(f"  Total Reward: {reward_partial:.3f}")
    
    assert 0 < reward_partial < 3.0, f"Partial prediction should score between 0 and 3, got {reward_partial}"
    
    print(f"\n✅ Partial prediction correctly scored {reward_partial:.3f}")
    
    print("\n✅ End-to-end test complete\n")


def test_edge_cases():
    """Test edge cases and potential issues."""
    print("="*80)
    print("TEST 4: Edge Cases")
    print("="*80)
    
    # Test 1: Empty output
    print("\n--- Edge Case 1: Empty Output ---")
    results = get_simple_results_from_raw_outputs("")
    assert results == [], "Empty output should return empty list"
    print("✅ Empty output handled correctly")
    
    # Test 2: No backticks
    print("\n--- Edge Case 2: No Backticks ---")
    output_no_backticks = """
test_file.py
class: MyClass
function: method
"""
    results = get_simple_results_from_raw_outputs(output_no_backticks)
    assert len(results) == 1, "Should parse without backticks"
    assert results[0]["file"] == "test_file.py"
    assert results[0]["class"] == "MyClass"
    assert results[0]["function"] == "method"
    print("✅ No backticks handled correctly")
    
    # Test 3: __init__ method
    print("\n--- Edge Case 3: __init__ Method ---")
    output_init = """```
test_file.py
class: MyClass
function: __init__
``````"""
    results = get_simple_results_from_raw_outputs(output_init)
    assert len(results) == 1
    assert results[0]["function"] == "__init__", "Should preserve __init__"
    print("✅ __init__ preserved correctly")
    
    # Test 4: Function with parameters
    print("\n--- Edge Case 4: Function with Parameters ---")
    output_params = """```
test_file.py
class: MyClass
function: my_method(self, arg1, arg2)
`````"""
    results = get_simple_results_from_raw_outputs(output_params)
    assert len(results) == 1
    assert results[0]["function"] == "my_method", "Should strip parameters"
    print("✅ Function parameters stripped correctly")
    
    # Test 5: Case insensitivity
    print("\n--- Edge Case 5: Case Variations ---")
    output_case = """```
test_file.py
Class: MyClass
Function: my_method
````"""
    results = get_simple_results_from_raw_outputs(output_case)
    assert len(results) == 1
    assert results[0]["class"] == "MyClass"
    assert results[0]["function"] == "my_method"
    print("✅ Case variations handled correctly")
    
    # Test 6: Multiple methods per class
    print("\n--- Edge Case 6: Multiple Methods Per Class ---")
    output_multi = """```
test_file.py
class: MyClass
function: method1
function: method2
function: method3
```"""
    results = get_simple_results_from_raw_outputs(output_multi)
    assert len(results) == 3, "Should parse all 3 methods"
    assert all(r["class"] == "MyClass" for r in results), "All should belong to MyClass"
    assert {r["function"] for r in results} == {"method1", "method2", "method3"}
    print("✅ Multiple methods per class handled correctly")
    
    print("\n✅ All edge cases passed\n")


def test_dataset_coverage():
    """Test coverage across the dataset to find potential issues."""
    print("="*80)
    print("TEST 5: Dataset Coverage Analysis")
    print("="*80)
    
    ds = load_dataset("adityasoni17/SWE-Gym-code-search", split="train")
    
    print(f"\nAnalyzing {len(ds)} examples...")
    
    has_init = 0
    has_dunder = 0
    standalone_funcs_in_modules = 0
    max_entities = 0
    multi_file = 0
    file_only = 0
    
    for example in ds[:100]:  # Check first 100
        file_changes = example["file_changes"]
        
        if len(file_changes) > 1:
            multi_file += 1
        
        for fc in file_changes:
            changes = fc["changes"]
            
            all_entities = (changes.get("edited_entities") or []) + \
                          (changes.get("added_entities") or [])
            
            if not all_entities:
                file_only += 1
            
            max_entities = max(max_entities, len(all_entities))
            
            for entity in all_entities:
                if "__init__" in entity:
                    has_init += 1
                if entity.split(":")[-1].startswith("__"):
                    has_dunder += 1
            
            # Check for standalone functions in modules (dataset bug)
            all_modules = (changes.get("edited_modules") or []) + \
                         (changes.get("added_modules") or [])
            
            for module in all_modules:
                # Check if this module is actually a standalone function
                is_standalone = True
                for entity in all_entities:
                    if entity.startswith(module + "."):
                        is_standalone = False
                        break
                
                if is_standalone:
                    standalone_funcs_in_modules += 1
    
    print(f"\nCoverage Statistics (first 100 examples):")
    print(f"  Examples with __init__: {has_init}")
    print(f"  Examples with dunder methods: {has_dunder}")
    print(f"  Max entities in single example: {max_entities}")
    print(f"  Multi-file examples: {multi_file}")
    print(f"  File-only changes (no entities): {file_only}")
    print(f"  Standalone functions mislabeled as modules: {standalone_funcs_in_modules}")
    
    if standalone_funcs_in_modules > 0:
        print(f"\n⚠️  WARNING: Found {standalone_funcs_in_modules} standalone functions in modules!")
        print("    These will be filtered out by extract_ground_truth_from_file_changes()")
    
    print("\n✅ Dataset coverage analysis complete\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE FORMAT VERIFICATION TEST SUITE")
    print("="*80 + "\n")
    
    try:
        test_gt_extraction()
        test_prediction_parsing()
        test_end_to_end_reward()
        test_edge_cases()
        test_dataset_coverage()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        print("\nYour GT extraction, prediction parsing, and reward calculation")
        print("are all correctly aligned and ready for training!")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)