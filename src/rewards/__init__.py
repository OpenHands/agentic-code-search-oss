# from .result_tool_check import result_tool_check
# from .result_tool_precision import result_tool_precision
# from .result_tool_recall import result_tool_recall
# from .result_tool_f1 import result_tool_f1

# __all__ = [
#     "result_tool_check",
#     "result_tool_precision",
#     "result_tool_recall",
#     "result_tool_f1",
# ]

# Make a registry of available reward functions that can added to by using a decorator
REWARD_REGISTRY = {
}

def get_reward_function(reward_name: str):
    """Get a reward function by name from the registry."""
    if reward_name not in REWARD_REGISTRY:
        raise ValueError(f"Reward function '{reward_name}' not found in registry.")
    return REWARD_REGISTRY[reward_name]

def reward(name: str):
    """Decorator to register a new reward function."""
    def decorator(func):
        REWARD_REGISTRY[name] = func
        return func
    return decorator