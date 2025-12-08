from src.rewards import reward

@reward("multiturn_reward")
def multiturn_reward(messages, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    if len(token_messages) > 1:
        return 1.0
    return 0.0