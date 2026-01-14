from typing import Set, List, Dict
from src.rewards import reward


def extract_ground_truth_from_file_changes(file_changes: list) -> dict:
    """
    Extract ground truth from pre-parsed file_changes field.
    
    Filters out standalone functions from modules (dataset bug where
    standalone functions appear in edited_modules).
    """
    files = set()
    modules = set()
    entities = set()
    
    # First pass: collect all entities
    all_entities = set()
    for change in file_changes:
        file_path = change["file"]
        files.add(file_path)
        
        changes = change["changes"]
        
        edited_entities = changes.get("edited_entities") or []
        added_entities = changes.get("added_entities") or []
        
        for entity in list(edited_entities) + list(added_entities):
            entities.add(entity)
            all_entities.add(entity)
    
    # Second pass: collect modules, filtering out standalone functions
    for change in file_changes:
        changes = change["changes"]
        
        edited_modules = changes.get("edited_modules") or []
        added_modules = changes.get("added_modules") or []
        
        for module in list(edited_modules) + list(added_modules):
            # Check if this module is actually used as a class prefix in entities
            # Real class: "file.py:ClassName" should have entities like "file.py:ClassName.method"
            # Fake class (standalone function): "file.py:function" won't have "file.py:function.xxx"
            
            is_real_class = False
            for entity in all_entities:
                if entity.startswith(module + "."):
                    is_real_class = True
                    break
            
            if is_real_class:
                modules.add(module)
            # else: skip it, it's a standalone function mislabeled as module
    
    return {
        "files": files,
        "modules": modules,
        "entities": entities
    }


def compute_f1(pred_set: Set, gt_set: Set) -> float:
    """Generic F1 computation for any level."""
    if not gt_set and not pred_set:
        return 1.0  # Perfect match when both empty
    
    if not gt_set:
        return 0.0  # FP: predicted something when GT is empty
    
    if not pred_set:
        return 0.0  # FN: predicted nothing when GT has items
    
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


@reward("multilevel_localization_f1_reward")
def multilevel_localization_f1_reward(
    final_message: str,
    instance: dict,
    file_level_weight: float = 1.0,
    module_level_weight: float = 1.0,
    entity_level_weight: float = 1.0,
    **kwargs
):
    """
    Compute F1 at file/module/entity levels.
    """
    
    # Extract GT from file_changes
    if "file_changes" not in instance or not instance["file_changes"]:
        return 0.0, {
            "file_f1": 0.0,
            "module_f1": 0.0,
            "entity_f1": 0.0,
            "error": "No file_changes in instance"
        }
    
    gt_data = extract_ground_truth_from_file_changes(instance["file_changes"])
    gt_files = gt_data["files"]
    gt_modules = gt_data["modules"]
    gt_entities = gt_data["entities"]
    
    # Parse predictions
    from src.rewards.file_localization.module_rewards import get_simple_results_from_raw_outputs
    
    results = get_simple_results_from_raw_outputs(final_message)
    
    pred_files = set()
    pred_modules = set()
    pred_entities = set()
    
    for result in results:
        # Handle None values properly
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
        
        # Build module (file:class) - ONLY if class exists
        if class_name:
            module = f"{file_path}:{class_name}"
            pred_modules.add(module)
        
        # Build entity (file:class.func or file:func)
        if func_name:
            if class_name:
                entity = f"{file_path}:{class_name}.{func_name}"
            else:
                entity = f"{file_path}:{func_name}"
            pred_entities.add(entity)
    
    # Compute F1 scores
    file_f1 = compute_f1(pred_files, gt_files) if gt_files else 0.0
    
    # Handle cases where GT might not have modules/entities
    if gt_modules:
        module_f1 = compute_f1(pred_modules, gt_modules)
    else:
        module_f1 = 1.0 if not pred_modules else 0.0
    
    if gt_entities:
        entity_f1 = compute_f1(pred_entities, gt_entities)
    else:
        entity_f1 = 1.0 if not pred_entities else 0.0
    
    # Weighted sum
    reward = (
        file_f1 * file_level_weight +
        module_f1 * module_level_weight +
        entity_f1 * entity_level_weight
    )
    
    return reward, {
        "multilevel_localization_f1_reward": reward,
        "file_f1": file_f1,
        "module_f1": module_f1,
        "entity_f1": entity_f1,
        "gt_files_count": len(gt_files),
        "gt_modules_count": len(gt_modules),
        "gt_entities_count": len(gt_entities),
        "pred_files_count": len(pred_files),
        "pred_modules_count": len(pred_modules),
        "pred_entities_count": len(pred_entities),
    }