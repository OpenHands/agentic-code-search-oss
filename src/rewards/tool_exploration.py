from src.rewards import reward
from typing import List, Dict, Any
from collections import defaultdict


def group_actions_by_turn(messages: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """
    Group action messages by their LLM response ID (turn).
    Returns dict: {llm_response_id: [action1, action2, ...]}
    """
    actions_by_turn = defaultdict(list)
    for msg in messages:
        if msg.get("kind") == "ActionEvent":
            llm_response_id = msg.get("llm_response_id")
            if llm_response_id:
                actions_by_turn[llm_response_id].append(msg)
    return actions_by_turn


@reward("semantic_search_usage_reward")
def semantic_search_usage_reward(messages: List[Dict[str, Any]], **kwargs) -> tuple[float, dict]:
    """
    Reward using semantic search before giving final answer.
    ACCOUNTS FOR PARALLEL TOOL CALLING.
    
    Returns higher reward if:
    - Semantic search is used at least once (any turn)
    - Used early in the trajectory (exploration phase)
    - Multiple diverse searches (different queries)
    """
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    actions_by_turn = group_actions_by_turn(messages)
    
    if not token_messages:
        return 0.0, {}
    
    # Track semantic searches per turn
    semantic_search_turns = []
    all_queries = []
    
    for turn_idx, token_msg in enumerate(token_messages):
        llm_response_id = token_msg.get("llm_response_id")
        if not llm_response_id:
            continue
        
        turn_actions = actions_by_turn.get(llm_response_id, [])
        
        # Check if ANY action in this turn is semantic search
        for action in turn_actions:
            tool_name = action.get("action", {}).get("tool", "")
            if "semantic" in tool_name.lower():
                query = action.get("action", {}).get("args", {}).get("query", "")
                semantic_search_turns.append(turn_idx)
                all_queries.append(query)
    
    if not semantic_search_turns:
        return -5.0, {
            "semantic_search_turns_count": 0,
            "semantic_search_total_calls": 0,
            "semantic_search_reward": -5.0
        }
    
    # Reward components
    base_reward = 5.0  # For using semantic search at all
    
    # Early usage bonus (used in first 3 turns)
    early_turns = [t for t in semantic_search_turns if t < 3]
    early_bonus = len(early_turns) * 3.0
    
    # Diversity bonus (different query strings)
    unique_queries = len(set(all_queries))
    diversity_bonus = min(unique_queries * 2.0, 6.0)  # Cap at 3 unique queries
    
    # Turn coverage bonus (multiple turns using semantic search, but not too many)
    turns_with_search = len(set(semantic_search_turns))
    if turns_with_search == 1:
        turn_bonus = 0.0
    elif turns_with_search <= 3:
        turn_bonus = turns_with_search * 1.5
    else:
        turn_bonus = 4.5 - (turns_with_search - 3) * 0.5  # Penalty for overuse
    
    total_reward = base_reward + early_bonus + diversity_bonus + turn_bonus
    
    return total_reward, {
        "semantic_search_turns_count": turns_with_search,
        "semantic_search_total_calls": len(all_queries),
        "semantic_search_unique_queries": unique_queries,
        "semantic_search_early_turns": len(early_turns),
        "semantic_search_reward": total_reward,
    }


@reward("bash_exploration_reward")
def bash_exploration_reward(messages: List[Dict[str, Any]], **kwargs) -> tuple[float, dict]:
    """
    Reward bash tool usage for code exploration.
    ACCOUNTS FOR PARALLEL TOOL CALLING.
    
    Rewards:
    - Using bash to search/explore code (rg, grep, find, ls)
    - Reading files (cat, less, head)
    - Penalize: No bash usage or only trivial commands
    """
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    actions_by_turn = group_actions_by_turn(messages)
    
    if not token_messages:
        return 0.0, {}
    
    # Track bash usage per turn
    turns_with_exploration = 0
    turns_with_file_read = 0
    total_exploration_calls = 0
    total_file_read_calls = 0
    
    for token_msg in token_messages:
        llm_response_id = token_msg.get("llm_response_id")
        if not llm_response_id:
            continue
        
        turn_actions = actions_by_turn.get(llm_response_id, [])
        
        turn_has_exploration = False
        turn_has_file_read = False
        
        for action in turn_actions:
            tool_name = action.get("action", {}).get("tool", "")
            if tool_name.lower() != "bash":
                continue
            
            command = action.get("action", {}).get("args", {}).get("command", "")
            
            # Categorize command
            if any(cmd in command for cmd in ["rg", "grep", "find", "ls", "locate"]):
                turn_has_exploration = True
                total_exploration_calls += 1
            elif any(cmd in command for cmd in ["cat", "less", "head", "tail", "bat"]):
                turn_has_file_read = True
                total_file_read_calls += 1
        
        if turn_has_exploration:
            turns_with_exploration += 1
        if turn_has_file_read:
            turns_with_file_read += 1
    
    if turns_with_exploration == 0 and turns_with_file_read == 0:
        return -3.0, {
            "bash_exploration_turns": 0,
            "bash_file_read_turns": 0,
            "bash_exploration_total_calls": 0,
            "bash_file_read_total_calls": 0,
            "bash_exploration_reward": -3.0
        }
    
    # Reward based on TURNS, not total calls (accounts for parallel calling)
    exploration_reward = min(turns_with_exploration * 2.5, 10.0)  # Cap at 4 turns
    file_read_reward = min(turns_with_file_read * 2.0, 8.0)  # Cap at 4 turns
    
    total_reward = exploration_reward + file_read_reward
    
    return total_reward, {
        "bash_exploration_turns": turns_with_exploration,
        "bash_file_read_turns": turns_with_file_read,
        "bash_exploration_total_calls": total_exploration_calls,
        "bash_file_read_total_calls": total_file_read_calls,
        "bash_exploration_reward": total_reward,
    }


@reward("tool_diversity_reward")
def tool_diversity_reward(messages: List[Dict[str, Any]], **kwargs) -> tuple[float, dict]:
    """
    Reward using BOTH semantic search AND bash tools.
    ACCOUNTS FOR PARALLEL TOOL CALLING - checks if both used across trajectory.
    """
    actions_by_turn = group_actions_by_turn(messages)
    
    # Check which tools were used across ALL turns
    has_semantic = False
    has_bash = False
    turns_with_both = 0
    
    for llm_response_id, turn_actions in actions_by_turn.items():
        turn_has_semantic = False
        turn_has_bash = False
        
        for action in turn_actions:
            tool_name = action.get("action", {}).get("tool", "")
            if "semantic" in tool_name.lower():
                has_semantic = True
                turn_has_semantic = True
            elif tool_name.lower() == "bash":
                has_bash = True
                turn_has_bash = True
        
        if turn_has_semantic and turn_has_bash:
            turns_with_both += 1
    
    # Base reward for using both tools at all
    if has_semantic and has_bash:
        base_reward = 8.0
        message = "both_tools_used"
    elif has_semantic or has_bash:
        base_reward = 2.0
        message = "single_tool_used"
    else:
        base_reward = -5.0
        message = "no_tools_used"
    
    # Bonus for turns where BOTH tools used together (parallel calling)
    parallel_bonus = min(turns_with_both * 3.0, 9.0)
    
    total_reward = base_reward + parallel_bonus
    
    return total_reward, {
        "has_semantic_search": has_semantic,
        "has_bash": has_bash,
        "turns_with_both_tools": turns_with_both,
        "tool_diversity_reward": total_reward,
        "tool_diversity_status": message,
    }


@reward("exploration_before_answer_reward")
def exploration_before_answer_reward(
    messages: List[Dict[str, Any]], 
    final_message: str,
    min_exploration_turns: int = 3,
    **kwargs
) -> tuple[float, dict]:
    """
    Ensure model explores (uses tools) for several turns before answering.
    ACCOUNTS FOR PARALLEL TOOL CALLING - counts turns, not individual calls.
    
    Penalize:
    - Answering immediately without exploration
    - Very short trajectories with answer
    
    Reward:
    - Sustained exploration (3+ turns with tools)
    - Then providing answer
    """
    token_messages = [msg for msg in messages if msg.get("kind") == "TokenEvent"]
    actions_by_turn = group_actions_by_turn(messages)
    
    if not token_messages:
        return 0.0, {}
    
    num_turns = len(token_messages)
    
    # Count turns that have at least one tool call
    turns_with_tools = len([
        llm_id for llm_id, actions in actions_by_turn.items()
        if len(actions) > 0
    ])
    
    # Check if final message is a real answer
    has_answer = (
        "<file-list>" in final_message or 
        "```" in final_message or
        "file" in final_message.lower() and "function" in final_message.lower() or
        len(final_message.split("\n")) > 5
    )
    
    if not has_answer:
        return -8.0, {
            "exploration_turns": num_turns,
            "turns_with_tools": turns_with_tools,
            "exploration_before_answer_reward": -8.0,
            "answer_provided": False,
        }
    
    # Calculate exploration quality
    if num_turns < min_exploration_turns:
        # Too short - rushed to answer
        reward = -10.0 + (num_turns * 2.0)
    elif turns_with_tools < min_exploration_turns:
        # Enough turns but not enough tool usage
        reward = -5.0 + (turns_with_tools * 2.0)
    else:
        # Good exploration before answering
        base_reward = 10.0
        
        # Bonus for sustained exploration (but not too long)
        if num_turns <= 10:
            length_bonus = (num_turns - min_exploration_turns) * 1.0
        else:
            length_bonus = (10 - min_exploration_turns) * 1.0 - (num_turns - 10) * 0.5
        
        reward = base_reward + length_bonus
    
    return reward, {
        "exploration_turns": num_turns,
        "turns_with_tools": turns_with_tools,
        "exploration_before_answer_reward": reward,
        "answer_provided": has_answer,
    }


@reward("parallel_tool_calling_bonus")
def parallel_tool_calling_bonus(messages: List[Dict[str, Any]], **kwargs) -> tuple[float, dict]:
    """
    BONUS reward for effective use of parallel tool calling.
    
    Rewards:
    - Turns where multiple tools are called in parallel
    - Especially: semantic_search + bash in same turn (smart exploration)
    """
    actions_by_turn = group_actions_by_turn(messages)
    
    turns_with_parallel = 0
    turns_with_semantic_bash_parallel = 0
    max_parallel = 0
    
    for llm_response_id, turn_actions in actions_by_turn.items():
        num_actions = len(turn_actions)
        
        if num_actions > 1:
            turns_with_parallel += 1
            max_parallel = max(max_parallel, num_actions)
            
            # Check if semantic + bash in same turn
            has_semantic = any("semantic" in a.get("action", {}).get("tool", "").lower() 
                             for a in turn_actions)
            has_bash = any(a.get("action", {}).get("tool", "").lower() == "bash" 
                          for a in turn_actions)
            
            if has_semantic and has_bash:
                turns_with_semantic_bash_parallel += 1
    
    if turns_with_parallel == 0:
        return 0.0, {
            "turns_with_parallel_tools": 0,
            "turns_with_semantic_bash_parallel": 0,
            "max_parallel_tools": 0,
            "parallel_tool_calling_bonus": 0.0,
        }
    
    # Reward parallel tool use
    base_parallel_reward = min(turns_with_parallel * 2.0, 8.0)
    
    # Extra bonus for semantic+bash parallel (very efficient)
    smart_parallel_bonus = turns_with_semantic_bash_parallel * 3.0
    
    total_reward = base_parallel_reward + smart_parallel_bonus
    
    return total_reward, {
        "turns_with_parallel_tools": turns_with_parallel,
        "turns_with_semantic_bash_parallel": turns_with_semantic_bash_parallel,
        "max_parallel_tools": max_parallel,
        "parallel_tool_calling_bonus": total_reward,
    }