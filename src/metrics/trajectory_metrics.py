from typing import Dict, List, Any
from collections import defaultdict

def compute_trajectory_metrics(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute trajectory-level metrics including turns and tool calls.
    
    Args:
        messages: List of event dictionaries from the conversation
        
    Returns:
        Dictionary with trajectory metrics:
            - num_turns: Number of LLM responses (TokenEvents)
            - num_tool_calls: Total number of tool calls (ActionEvents)
            - num_tool_calls_per_turn: Average tool calls per turn
            - tool_calls_by_turn: List of tool call counts for each turn
    """
    if messages == []:
        return {
            "num_turns": 0,
            "num_tool_calls": 0,
            "num_tool_calls_per_turn": 0.0,
            "tool_calls_by_turn": [],
        }

    # Count turns (TokenEvents)
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    num_turns = len(token_messages)
    
    # Count tool calls (ActionEvents) and group by llm_response_id for parallel calls
    action_messages = [msg for msg in messages if msg.get("kind") == "ActionEvent"]
    num_tool_calls = len(action_messages)
    
    # Group tool calls by llm_response_id to understand parallel tool calling
    tool_calls_by_response = defaultdict(int)
    for action in action_messages:
        llm_response_id = action.get("llm_response_id")
        if llm_response_id:
            tool_calls_by_response[llm_response_id] += 1
    
    # Get list of tool call counts per turn
    tool_calls_by_turn = list(tool_calls_by_response.values())
    
    # Compute average tool calls per turn
    num_tool_calls_per_turn = num_tool_calls / num_turns if num_turns > 0 else 0.0
    
    return {
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "num_tool_calls_per_turn": num_tool_calls_per_turn,
        "tool_calls_by_turn": tool_calls_by_turn,
    }