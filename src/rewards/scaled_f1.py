from src.rewards import reward

from src.rewards.file_localization.file_localization import file_localization_f1_reward

@reward("scaled_f1_reward")
def scaled_f1_reward(
    final_message,
    messages,
    instance,
    **kwargs
    ):
    f1_score, info_dict = file_localization_f1_reward(final_message, instance, **kwargs)

    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages) - 1
    if num_turns == 0:
        num_turns = 1  # to avoid division by zero

    num_tool_calls = len(tool_messages)

    avg_tool_calls_per_turn = max(5, num_tool_calls / num_turns)

    info_dict["tool_use_reward"] = avg_tool_calls_per_turn

    reward = f1_score * avg_tool_calls_per_turn
    
    return reward, info_dict