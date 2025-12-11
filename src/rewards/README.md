# Adding Rewards

You can easily add rewards by registering it

```
from src.rewards import reward

@reward("multiturn_reward")
def multiturn_reward(messages, minimal_turns=1, **kwargs) -> float:
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    if len(token_messages) > minimal_turns:
        return 1.0
    return 0.0
```

## Main Arguments

There will be 3 main arguments that will be fed to any reward function

```
input_args = {
    "final_message": final_message, # Final prediction from model
    "messages": messages, # Trajectory messages
    "instance": instance, # Input data
}
```

You should always include `**kwargs` in your function so that unused arguments will pass through.


## Making a yaml configuration

To list what rewards you want to use, you can list them in a new config.yaml

```
reward:
  - fn: multiturn_reward
    args:
        minimal_turns: 2.0
  - fn: file_localization_f1_reward
```

Note that `fn` will point towards your registered name in `@reward`. If you want to define a parameter specific to a reward function, you can ste that in the `args` just like the example above.


## Running with Custom Configuration

To run with your custom yaml

```
bash scripts/run_async_training.sh \
    -m Qwen/Qwen3-8B \
    ... \
    -o "+generator.reward=custom_config.yaml"
```