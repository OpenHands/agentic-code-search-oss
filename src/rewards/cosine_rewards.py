import math
from src.rewards import reward

from src.rewards.file_localization.file_localization import (
    multilevel_localization_f1_reward,
    file_localization_f1_reward
    )

@reward("cosine_reward")
def cosine_reward(
    final_message,
    instance,
    messages,
    loc_threshold=1.5,
    use_tool_reward=True,
    use_turn_reward=True,
    use_length_reward=False,
    max_turns=8,
    max_avg_tool_calls=10,
    ideal_avg_tool_calls=5,
    max_length=16384,
    multilevel=True,
    **kwargs
    ):

    try:
        if multilevel:
            loc_reward, reward_dict = multilevel_localization_f1_reward(final_message, instance, **kwargs)
        else:
            loc_reward, reward_dict = file_localization_f1_reward(final_message, instance, **kwargs)

    except Exception as e:
        print(f"Error computing localization reward: {e}")
        loc_reward = 0.0
        reward_dict = {
            "multilevel_localization_f1_reward": 0.0,
            "file_reward": 0.0,
            "module_reward": 0.0,
            "entity_reward": 0.0,
        }

    def _cos_fn(t, T, mu_min, mu_max):
        cos_inner = (math.pi * t) / T
        cos_out = math.cos(cos_inner) + 1
        return mu_min + 0.5 * (mu_max - mu_min) * cos_out

    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    # Don't count the last turn which is the 
    # final answer generation and does not involve tool use
    num_turns = min(1, len(token_messages)-1)
    num_tool_calls = len(tool_messages)
    avg_tool_calls_per_turn = num_tool_calls / num_turns if num_turns > 0 else 0

    reward = 0.0

    # Number of turns
    if use_turn_reward:
        if num_turns > max_turns:
            cosine_turn_reward = 0
        elif loc_reward >= loc_threshold:
            cosine_turn_reward = _cos_fn(num_turns, max_turns, 0.0, 5.0)
        else:
            cosine_turn_reward = _cos_fn(num_turns, max_turns, 0.0, -5.0)
        reward_dict["turn_cosine_reward"] = cosine_turn_reward

        reward += cosine_turn_reward

    # Length of response
    if use_length_reward:
        current_prompt_ids = token_messages[0]["prompt_token_ids"]
        ending_prompt_ids = token_messages[-1]["prompt_token_ids"]
        ending_response_ids = token_messages[-1]["response_token_ids"]
        current_response_ids = ending_prompt_ids + ending_response_ids
        current_response_ids = current_response_ids[len(current_prompt_ids):]

        current_length = len(current_prompt_ids) + len(current_response_ids)

        if current_length > max_length:
            cosine_length_reward = 0
        elif loc_reward >= loc_threshold:
            cosine_length_reward = _cos_fn(current_length, max_length, 0.0, 5.0)
        else:
            cosine_length_reward = _cos_fn(current_length, max_length, 0.0, -5.0)
        reward_dict["length_cosine_reward"] = cosine_length_reward

        reward += cosine_length_reward
    
    # Number of tool calls
    if use_tool_reward:
        if avg_tool_calls_per_turn > max_avg_tool_calls:
            cosine_tool_reward = 0
        elif loc_reward >= loc_threshold:
            # Using 5 as the ideal average number of tool calls per turn
            # Anything more or less than the max score
            if avg_tool_calls_per_turn >= ideal_avg_tool_calls:
                avg_tool_calls_per_turn -= ideal_avg_tool_calls
                cosine_tool_reward = _cos_fn(avg_tool_calls_per_turn, ideal_avg_tool_calls, 1.0, 5.0)
            else:
                cosine_tool_reward = _cos_fn(avg_tool_calls_per_turn, ideal_avg_tool_calls, 5.0, 1.0)
        else:
            # If wrong, encourage to do more calls
            cosine_tool_reward = _cos_fn(avg_tool_calls_per_turn, 10, 0.0, -5.0)
        reward_dict["tool_cosine_reward"] = cosine_tool_reward

        reward += cosine_tool_reward

    return reward, reward_dict
