import ast

from .module_rewards import get_simple_results_from_raw_outputs, parse_structured_outputs

from src.rewards import reward

def compute_file_f1_score(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    if not true:
        return 0.0 # return 0 reward if ground truth is empty
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0

    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

# def file_localization_f1_reward(final_message, instance):
#     predicted_files = set(ast.literal_eval(final_message.split("<file-list>")[1].split("</file-list>")[0]))
#     # print("Predicted files:", predicted_files)
#     true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
#     # print("True files:", true_files)
#     return compute_file_f1_score(predicted_files, true_files)

@reward("file_localization_f1_reward")
def file_localization_f1_reward(
    final_message: str,
    instance: dict,
    file_level_weight: float=1.0,
    **kwargs
    ):
    all_found_files, all_found_modules, all_found_entities = get_simple_results_from_raw_outputs(final_message)
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
    file_level_score = compute_file_f1_score(all_found_files, true_files)
    weighted_file_score = file_level_weight * file_level_score

    return weighted_file_score, {"file_level_score": file_level_score}

@reward("multilevel_localization_f1_reward")
def multilevel_localization_f1_reward(
    final_message: str,
    instance: dict,
    structured_locations: list[dict] | None = None,
    file_level_weight: float=1.0,
    module_level_weight: float=1.0,
    entity_level_weight: float=1.0,
    **kwargs
    ):

    gt_files = []
    gt_modules = []
    gt_entities = []
    reward = 0

    for change in instance.get("file_changes", []):
        if "file" in change:
            gt_files.append(change["file"])
        if "changes" in change:
            edited_modules = change["changes"].get("edited_modules", [])
            edited_modules = [] if edited_modules is None else edited_modules
            for module in edited_modules:
                gt_modules.append(module)

            edited_entities = change["changes"].get("edited_entities", [])
            edited_entities = [] if edited_entities is None else edited_entities
            for entity in edited_entities:
                gt_entities.append(entity)
    gt_files = set(gt_files)
    gt_modules = set(gt_modules)
    gt_entities = set(gt_entities)

    if structured_locations is not None:
        predicted_files, predicted_modules, predicted_entities = parse_structured_outputs(structured_locations)
    else:
        predicted_files, predicted_modules, predicted_entities = get_simple_results_from_raw_outputs(final_message)

    file_f1_score = compute_file_f1_score(predicted_files, gt_files)
    module_f1_score = compute_file_f1_score(predicted_modules, gt_modules)
    entity_f1_score = compute_file_f1_score(predicted_entities, gt_entities)

    # weight_total = file_level_weight + module_level_weight + entity_level_weight
    # file_level_weight /= weight_total
    # module_level_weight /= weight_total
    # entity_level_weight /= weight_total

    reward = (
        file_f1_score * file_level_weight
    + module_f1_score * module_level_weight
    + entity_f1_score * entity_level_weight
    )

    return reward, {
        "multilevel_localization_f1_reward": reward,
        "file_reward": file_f1_score,
        "module_reward": module_f1_score,
        "entity_reward": entity_f1_score,
        # "prediction": {
        #     "files": list(predicted_files),
        #     "modules": list(predicted_modules),
        #     "entities": list(predicted_entities),
        # },
        # "ground_truth": {
        #     "files": list(gt_files),
        #     "modules": list(gt_modules),
        #     "entities": list(gt_entities),
        # },
    }