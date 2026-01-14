from src.rewards import reward
from typing import List, Dict, Any
import re


def get_tool_name(action_msg: Dict[str, Any]) -> str:
    """Extract tool name from ActionEvent (handles OpenHands SDK format)."""
    tool_name = action_msg.get("tool_name", "")
    if tool_name:
        return tool_name.lower()
    return action_msg.get("action", {}).get("tool", "").lower()


@reward("exploration_constraint")
def exploration_constraint(messages: List[Dict[str, Any]], **kwargs) -> tuple[float, dict]:
    """
    Combined constraint: Must explore before answering.
    
    ⚠️ This is a CONSTRAINT, not a reward.
    Penalties should be strong enough to prevent shortcuts,
    but not so large they dominate the F1 signal.
    
    Current penalties:
    - No tools: -20
    - No semantic: -10
    - Too few turns: -10 per turn
    
    Max penalty: ~-40
    F1 reward range: 0 to 60
    
    This means: Good F1 (40+) can overcome constraint violations,
    but violations significantly hurt the score.
    """
    action_messages = [msg for msg in messages if msg.get("kind") == "ActionEvent"]
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    
    num_turns = len(token_messages)
    num_tools = len(action_messages)
    
    used_semantic = any(
        "semantic" in get_tool_name(msg)
        for msg in action_messages
    )
    
    violations = []
    penalty = 0.0
    
    if num_tools == 0:
        violations.append("no_tools")
        penalty -= 20.0
    
    if not used_semantic:
        violations.append("no_semantic_search")
        penalty -= 10.0
    
    if num_turns < 2:
        violations.append("too_few_turns")
        penalty -= 10.0 * (2 - num_turns)
    
    if penalty < 0:
        return penalty, {
            "exploration_constraint_violated": True,
            "violations": violations,
            "num_turns": num_turns,
            "num_tools": num_tools,
            "used_semantic": used_semantic,
        }
    
    return 0.0, {
        "exploration_constraint_violated": False,
        "num_turns": num_turns,
        "num_tools": num_tools,
        "used_semantic": used_semantic,
    }


@reward("output_format_reward")
def output_format_reward(final_message: str, **kwargs) -> tuple[float, dict]:
    """
    Reward for proper output format.
    
    Range: -15 to +10 (soft constraint)
    
    Checks:
    1. Has triple backticks (optional but encouraged): ±2
    2. Has .py files: +3 or -10
    3. Has proper format (class:/function: or colon notation): +3 or -8
    4. Has multiple entries: +2 or -2
    """
    
    output = final_message.strip()
    
    # Check 1: Empty or too short
    if len(output) < 20:
        return -15.0, {
            "format_score": -15.0,
            "has_backticks": False,
            "has_files": False,
            "has_proper_format": False,
            "num_files": 0,
        }
    
    # Check 2: Has triple backticks
    has_backticks = "```" in output
    
    # Check 3: Extract content
    import re
    backtick_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', output, re.DOTALL)
    if backtick_match:
        content = backtick_match.group(1)
    else:
        content = output
    
    # Check 4: Has .py file paths
    py_files = re.findall(r'\S+\.py', content)
    has_files = len(py_files) > 0
    num_files = len(set(py_files))
    
    # Check 5: Has proper format
    has_labels = bool(re.search(r'(?:^|\n)(?:class|function|method):', content, re.IGNORECASE))
    has_colon_notation = bool(re.search(r'\.py:[A-Za-z_][A-Za-z0-9_.]*', content))
    has_file_only_lines = bool(re.search(r'(?:^|\n)\S+\.py\s*(?:\n|$)', content))
    
    has_proper_format = has_labels or has_colon_notation or (has_files and has_file_only_lines)
    
    # Check 6: Multiple entries
    labeled_entries = len(re.findall(r'(?:^|\n)(?:class|function|method):', content, re.IGNORECASE))
    colon_entries = len(re.findall(r'\.py:[A-Za-z_]', content))
    total_entries = max(labeled_entries, colon_entries, num_files)
    has_multiple_entries = total_entries >= 2
    
    # Calculate score
    violations = []
    score = 0.0
    
    if not has_backticks:
        violations.append("missing_backticks")
        score -= 2.0
    else:
        score += 2.0
    
    if not has_files:
        violations.append("no_python_files")
        score -= 10.0
    else:
        score += 3.0
    
    if not has_proper_format:
        violations.append("improper_format")
        score -= 8.0
    else:
        score += 3.0
    
    if has_files and has_proper_format and not has_multiple_entries:
        if has_labels or has_colon_notation:
            violations.append("single_entry_only")
            score -= 2.0
    elif has_multiple_entries:
        score += 2.0
    
    return score, {
        "format_score": score,
        "has_backticks": has_backticks,
        "has_files": has_files,
        "num_files": num_files,
        "has_proper_format": has_proper_format,
        "has_multiple_entries": has_multiple_entries,
        "total_entries": total_entries,
        "format_violations": violations if violations else None,
    }