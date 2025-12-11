from src.rewards import reward

@reward("multiturn_reward")
def multiturn_reward(messages, minimal_turns=1, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    if len(token_messages) > minimal_turns:
        return 1.0
    return 0.0