from src.rewards import reward


def get_tool_name(action_msg):
    """Extract tool name handling both formats."""
    tool_name = action_msg.get("tool_name", "")
    if tool_name:
        return tool_name.lower()
    return action_msg.get("action", {}).get("tool", "").lower()


@reward("tool_use_reward")
def tool_use_reward(messages, **kwargs) -> float:
    """
    Small bonus for tool usage ratio.
    
    ⚠️ This is a BONUS, not primary signal.
    Range: 0 to ~10 (much smaller than F1's 0-60)
    """
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)
    
    if num_turns == 0:
        return 0.0

    # Returns 0.5 to 3.0 typically
    return 3*num_tool_calls/num_turns


@reward("turn_efficiency")
def turn_efficiency(messages, max_turns=12, **kwargs) -> float:
    """
    Small penalty for going too long.
    
    ⚠️ This is a SOFT constraint.
    Range: 0 to 5 (very small compared to F1)
    """
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)
    
    if num_turns <= 1:
        return 0.0
    
    if (num_tool_calls > 1):
        if num_turns <= max_turns:
            return 5.0
        else:
            return max(0.0, 5.0 - (num_turns - max_turns) * 0.1)

    return 0.0