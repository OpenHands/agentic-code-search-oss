from src.rewards import reward

@reward("tool_use_reward")
def tool_use_reward(messages, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)
    
    if num_turns == 0:
        return 0.0
    
    # if num_tool_calls/num_turns >= 1.0:
    #     return num_tool_calls/num_turns

    return num_tool_calls/num_turns

@reward("turn_efficiency")
def turn_efficiency(messages, max_turns=4, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)
    
    if num_turns == 0:
        return 0.0
    
    if (num_tool_calls >= 1):
        # Decay the reward if more than max_turns are used
        if num_turns <= max_turns:
            return 1.0
        else:
            return max(0.0, 1.0 - (num_turns - max_turns) * 0.1)

    return 0.0