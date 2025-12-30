import re
from typing import Tuple

from src.rewards import reward


@reward("format_reward")
def format_reward(
    final_message: str,
    messages=None,
    *,
    tool_pattern: str = r"<(glob|grep|terminal)>",
    eos_hygiene_reward: float = 0.05,
    balanced_think_reward: float = 0.05,
    final_content_no_tool_reward: float = 0.10,
    min_final_content_length: int = 10,
    **kwargs,
) -> Tuple[float, dict]:
    """
    Non-negative format reward. Adds bonuses only (no penalties):
    (1) EOS hygiene (no <|endoftext|> leak), (2) balanced <think> tags,
    (3) final turn has content and no tool calls.
    """
    reward_value = 0.0
    reward_dict = {}

    # (1) Check for <|endoftext|> leak
    if "<|endoftext|>" not in final_message:
        reward_value += eos_hygiene_reward
        reward_dict["eos_hygiene_reward"] = eos_hygiene_reward

    # (2) XML-like tag balance for <think>...</think>
    has_open_think = "<think>" in final_message
    has_close_think = "</think>" in final_message
    # Check if both tags are present and the </think> tag is after the <think> tag
    if has_open_think and has_close_think and final_message.find("</think>") > final_message.find("<think>"):
        reward_value += balanced_think_reward
        reward_dict["balanced_think_reward"] = balanced_think_reward

    # (3) Final answer validity: reward clean hand-off (no tool calls, has content)
    has_tool = bool(re.search(tool_pattern, final_message))
    # Check if the final message has content and is not empty to avoid rewarding empty final messages
    has_content = len(final_message.strip()) > min_final_content_length
    if has_content and not has_tool:
        reward_value += final_content_no_tool_reward
        reward_dict["final_content_no_tool_reward"] = final_content_no_tool_reward

    reward_dict["format_reward"] = reward_value
    return reward_value, reward_dict