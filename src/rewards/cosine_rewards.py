import math
from src.rewards import reward

from src.rewards.file_localization.file_localization import multilevel_localization_f1_reward

@reward("cosine_reward")
def turn_cosine_reward(final_message, instance, messages, **kwargs) -> float:

    try:
        loc_reward, reward_dict = multilevel_localization_f1_reward(final_message, instance, **kwargs)
    except Exception as e:
        print(f"Error computing localization reward: {e}")
        loc_reward = 0.0
        reward_dict = {}

    def _cos_fn(t, T, mu_min, mu_max):
        cos_inner = (math.pi * t) / T
        cos_out = math.cos(cos_inner) + 1
        return mu_min + 0.5 * (mu_max - mu_min) * cos_out

    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    tool_messages = [msg for msg in messages if msg["kind"] == "ActionEvent"]
    
    num_turns = len(token_messages)
    num_tool_calls = len(tool_messages)

    # Number of turns
    if num_turns > 10:
        cosine_turn_reward = -10.0
    elif loc_reward >= 0.5:
        cosine_turn_reward = _cos_fn(num_turns, 10, 0.0, 5.0)
    else:
        cosine_turn_reward = _cos_fn(num_turns, 10, 0.0, -5.0)
    reward_dict["turn_cosine_reward"] = cosine_turn_reward
    
    # Number of tool calls
    num_turns = max(1, num_turns)  # avoid division by zero
    avg_tool_calls_per_turn = num_tool_calls / num_turns
    if avg_tool_calls_per_turn > 5:
        cosine_tool_reward = -10.0
    elif loc_reward >= 0.5:
        cosine_tool_reward = _cos_fn(avg_tool_calls_per_turn, 5, 0.0, 5.0)
    else:
        cosine_tool_reward = _cos_fn(avg_tool_calls_per_turn, 5, 0.0, -5.0)
    reward_dict["tool_cosine_reward"] = cosine_tool_reward

    return cosine_turn_reward + cosine_tool_reward, reward_dict
    