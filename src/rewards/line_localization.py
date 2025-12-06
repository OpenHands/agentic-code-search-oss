import ast

def iou_score(predicted_lines, true_lines):
    # predicted_lines / true_lines:
    # [(file_path, (start_line, end_line)), ...]
    if not predicted_lines and not true_lines:
        return 1.0

    def to_per_file_intervals(spans):
        per_file = {}
        for file_path, (start, end) in spans:
            if file_path not in per_file:
                per_file[file_path] = []
            per_file[file_path].append((start, end))
        return per_file

    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals)
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def total_length(intervals):
        return sum(end - start + 1 for start, end in intervals)

    def intersection_length(a, b):
        i = 0
        j = 0
        total = 0
        while i < len(a) and j < len(b):
            s1, e1 = a[i]
            s2, e2 = b[j]
            start = max(s1, s2)
            end = min(e1, e2)
            if start <= end:
                total += end - start + 1
            if e1 < e2:
                i += 1
            else:
                j += 1
        return total

    pred_per_file = to_per_file_intervals(predicted_lines)
    true_per_file = to_per_file_intervals(true_lines)

    pred_total = 0
    true_total = 0
    inter_total = 0

    all_files = set(pred_per_file) | set(true_per_file)
    for file_path in all_files:
        pred_intervals = merge_intervals(pred_per_file.get(file_path, []))
        true_intervals = merge_intervals(true_per_file.get(file_path, []))
        pred_total += total_length(pred_intervals)
        true_total += total_length(true_intervals)
        if pred_intervals and true_intervals:
            inter_total += intersection_length(pred_intervals, true_intervals)

    union_total = pred_total + true_total - inter_total
    return inter_total / union_total if union_total > 0 else 0.0

def line_localization_reward(final_message, instance):
    # Expected format: <answer>[(file1, (start_line, end_line)), (file2, (start_line, end_line)), (file2, (start_line, end_line)), ...]</answer>
    pred = ast.literal_eval(final_message.split("<answer>")[1].split("</answer>")[0])
    true = ast.literal_eval(instance["target"])
    return iou_score(pred, true)