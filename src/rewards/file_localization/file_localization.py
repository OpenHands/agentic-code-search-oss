import ast

from .module_rewards import get_simple_results_from_raw_outputs

from src.rewards import reward

def compute_file_f1_score(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    if not pred and not true:
        return 1.0
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
