"""Utility functions for computing efficiency metrics from trajectory data."""

from typing import Dict, List, Any


def compute_token_metrics(messages: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute token usage metrics from messages.

    Args:
        messages: List of message dictionaries from conversation events

    Returns:
        Dictionary with keys:
        - total_tokens: Total tokens (prompt + response)
        - total_prompt_tokens: Total prompt tokens
        - total_response_tokens: Total response/completion tokens
        - avg_prompt_tokens_per_step: Average prompt tokens per TokenEvent
        - avg_response_tokens_per_step: Average response tokens per TokenEvent
    """
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]

    if not token_messages:
        return {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "avg_prompt_tokens_per_step": 0.0,
            "avg_response_tokens_per_step": 0.0,
        }

    total_prompt_tokens = sum(len(msg.get("prompt_token_ids", [])) for msg in token_messages)
    total_response_tokens = sum(len(msg.get("response_token_ids", [])) for msg in token_messages)

    num_steps = len(token_messages)
    avg_prompt = total_prompt_tokens / num_steps if num_steps > 0 else 0.0
    avg_response = total_response_tokens / num_steps if num_steps > 0 else 0.0

    return {
        "total_tokens": total_prompt_tokens + total_response_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_response_tokens": total_response_tokens,
        "avg_prompt_tokens_per_step": avg_prompt,
        "avg_response_tokens_per_step": avg_response,
    }


def compute_step_count(messages: List[Dict[str, Any]]) -> int:
    """
    Compute the number of steps (TokenEvents) in the trajectory.

    Args:
        messages: List of message dictionaries from conversation events

    Returns:
        Number of TokenEvent messages (agent turns)
    """
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    return len(token_messages)


def compute_tool_call_metrics(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute tool call metrics from messages.

    Args:
        messages: List of message dictionaries from conversation events

    Returns:
        Dictionary with keys:
        - total_tool_calls: Total number of tool invocations
        - avg_tool_calls_per_step: Average tool calls per step
        - tool_call_breakdown: Dict mapping tool names to counts
    """
    # Find all assistant messages with tool calls
    tool_call_count = 0
    tool_breakdown = {}

    for msg in messages:
        # Assistant messages contain tool_calls field
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tool_calls = msg.get("tool_calls", [])
            tool_call_count += len(tool_calls)

            # Track tool types
            for tool_call in tool_calls:
                # Handle both dict and object-like structures
                if isinstance(tool_call, dict):
                    if "function" in tool_call:
                        tool_name = tool_call["function"].get("name", "unknown")
                    else:
                        tool_name = "unknown"
                elif hasattr(tool_call, "function"):
                    tool_name = getattr(tool_call.function, "name", "unknown")
                else:
                    tool_name = "unknown"

                tool_breakdown[tool_name] = tool_breakdown.get(tool_name, 0) + 1

    # Calculate average per step
    num_steps = compute_step_count(messages)
    avg_tool_calls = tool_call_count / num_steps if num_steps > 0 else 0.0

    return {
        "total_tool_calls": tool_call_count,
        "avg_tool_calls_per_step": avg_tool_calls,
        "tool_call_breakdown": tool_breakdown,
    }


def compute_all_efficiency_metrics(
    messages: List[Dict[str, Any]],
    wall_clock_duration: float,
    start_timestamp: str = None,
    end_timestamp: str = None,
) -> Dict[str, Any]:
    """
    Compute all efficiency metrics for a trajectory.

    Args:
        messages: List of message dictionaries from conversation events
        wall_clock_duration: Duration in seconds
        start_timestamp: ISO timestamp of conversation start (optional)
        end_timestamp: ISO timestamp of conversation end (optional)

    Returns:
        Dictionary containing all efficiency metrics
    """
    token_metrics = compute_token_metrics(messages)
    step_count = compute_step_count(messages)
    tool_metrics = compute_tool_call_metrics(messages)

    efficiency_metrics = {
        # Core metrics from issue #26
        "tokens": token_metrics["total_tokens"],
        "steps": step_count,
        "avg_tool_calls_per_step": tool_metrics["avg_tool_calls_per_step"],
        "wall_clock_duration": wall_clock_duration,

        # Extended metrics for richer analysis
        "token_breakdown": {
            "total_prompt_tokens": token_metrics["total_prompt_tokens"],
            "total_response_tokens": token_metrics["total_response_tokens"],
            "avg_prompt_tokens_per_step": token_metrics["avg_prompt_tokens_per_step"],
            "avg_response_tokens_per_step": token_metrics["avg_response_tokens_per_step"],
        },
        "tool_breakdown": {
            "total_tool_calls": tool_metrics["total_tool_calls"],
            "by_tool_type": tool_metrics["tool_call_breakdown"],
        },
    }

    # Add timestamps if provided
    if start_timestamp:
        efficiency_metrics["start_timestamp"] = start_timestamp
    if end_timestamp:
        efficiency_metrics["end_timestamp"] = end_timestamp

    return efficiency_metrics
