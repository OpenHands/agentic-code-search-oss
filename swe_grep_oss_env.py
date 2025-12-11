import json
import logging
import traceback
from typing import Literal

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

import src.tools as tools
from src.constants import DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOOL_CALLS
from src.prompts.system_prompt import SYSTEM_PROMPT
from src.utils.get_instance import get_instance_path
from src.utils.parse_patch import parse_patch

logger = logging.getLogger("swe-grep-oss")


class SWEGrepEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Only add bash tool - no result tool needed with XML output
        self.add_tool(tools.bash, args_to_skip=["cwd"])

    async def is_completed(
        self, messages: vf.types.Messages, state: vf.types.State, **kwargs
    ) -> bool:
        max_turns_reached = await self.max_turns_reached(state)
        prompt_too_long = await self.prompt_too_long(state)

        # Check if the last message contains <files> XML tags
        has_files_tag = False
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.get("role") == "assistant":
                content = last_message.get("content", "")
                if isinstance(content, str) and "<files>" in content and "</files>" in content:
                    has_files_tag = True

        if has_files_tag or max_turns_reached or prompt_too_long:
            return True

        return False

    async def env_response(
        self, messages: vf.types.Messages, state: vf.types.State, **kwargs
    ) -> tuple[vf.types.Messages, vf.types.State]:
        assert isinstance(messages, list)

        tool_messages = []
        tool_calls = messages[-1].get("tool_calls", [])
        for tool_call in tool_calls:
            tool_name: str = tool_call.get("function", {}).get("name", "")
            tool_call_id: str = tool_call.get("id", "")

            arguments_str = tool_call.get("function", {}).get("arguments", "")

            try:
                tool_args = json.loads(arguments_str)

                # Handle double-encoded JSON (when json.loads returns a string instead of dict)
                if isinstance(tool_args, str):
                    self.logger.warning(
                        f"Double-encoded JSON detected, attempting to parse again: {tool_args[:100]}"
                    )
                    tool_args = json.loads(tool_args)

                # Final check: must be a dict
                if not isinstance(tool_args, dict):
                    raise TypeError(f"Expected dict, got {type(tool_args).__name__}")

            except (json.JSONDecodeError, TypeError) as e:
                self.logger.error(f"Failed to parse tool arguments: {e}")
                self.logger.error(f"Raw arguments: {repr(arguments_str)}")
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": f"Error: Invalid tool arguments - {str(e)}",
                        "tool_call_id": tool_call_id,
                    }
                )
                continue

            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            tool_message: vf.types.Message = await self.call_tool(
                tool_name, tool_args, tool_call_id
            )
            tool_messages.append(tool_message)
        return tool_messages, state

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: vf.types.Messages,
        completion: vf.types.Messages | None = None,
        answer: str = "",
        state: vf.types.State = {},
        task: str = "default",
        info: vf.types.Info | None = None,
        example_id: int = 0,
        sampling_args: vf.types.SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[vf.types.Messages, vf.types.State]:
        try:
            return await super().rollout(
                client,
                model,
                prompt,
                completion,
                answer,
                state,
                task,
                info,
                example_id,
                sampling_args,
                **kwargs,
            )
        except Exception as e:
            import traceback

            self.logger.error(f"Error in rollout: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise to see the actual error

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.types.Messages,
        state: vf.types.State,
        **kwargs,
    ) -> dict:
        try:
            if tool_name == "bash":
                repo_path = get_instance_path(
                    {
                        "repo": state["info"]["repo"],
                        "instance_id": state["info"]["instance_id"],
                    }
                )
                updated_tool_args = dict(tool_args)
                updated_tool_args["cwd"] = repo_path
                return updated_tool_args
        except Exception:
            # Add detailed logging
            self.logger.error(f"update_tool_args called with tool_name={tool_name}")
            self.logger.error(f"tool_args type: {type(tool_args)}")
            self.logger.error(f"tool_args value: {repr(tool_args)}")
            self.logger.error(f"messages: {messages}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise to see the actual error

        return tool_args


def load_environment(
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    mode: Literal["train", "test", "full", "rl"] = "rl",
    **kwargs,
):
    """
    Load and configure the SWE-Grep environment.

    Args:
        max_tokens: Maximum tokens for model responses
        max_tool_calls: Maximum number of tool calls allowed
        mode: Dataset mode - "train" (80%), "test" (20%), "full" (100%), or "rl" (train+eval split)
        **kwargs: Additional arguments passed to SWEGrepEnv

    Returns:
        SWEGrepEnv instance configured with the specified dataset
    """

    # Load and prepare dataset
    full_dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    full_dataset = full_dataset.shuffle(seed=42)

    # Transform dataset with metadata and prompts
    def transform_row(row):
        return {
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
                "max_tokens": max_tokens,
                "max_tool_calls": max_tool_calls,
            },
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["problem_statement"]},
            ],
            "answer": json.dumps(parse_patch(row["patch"])),
        }

    full_dataset = full_dataset.map(transform_row)

    # Split dataset for train/eval modes
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # XML parser to extract files from <files> tags
    parser = vf.XMLParser(["files"], answer_field="files")

    # Reward function: F1 score between predicted and actual files
    def file_localization_reward(completion: str, answer: str, **kwargs) -> float:
        """
        Calculate F1 score between predicted files and actual files from patch.
        """

        # Helper function to normalize file paths
        def normalize_path(path: str) -> str:
            """Remove leading './' from file paths for consistent comparison."""
            path = path.strip()
            if path.startswith("./"):
                path = path[2:]
            return path

        # Parse the model's response
        predicted_files_str = parser.parse_answer(completion)
        if predicted_files_str is None:
            return 0.0

        try:
            # Try to parse as JSON array
            if predicted_files_str.strip().startswith("["):
                predicted_files = json.loads(predicted_files_str)
            else:
                # Split by newlines and filter empty
                predicted_files = [f.strip() for f in predicted_files_str.split("\n") if f.strip()]
        except:
            predicted_files = []

        # Parse the ground truth answer
        try:
            actual_files = json.loads(answer) if isinstance(answer, str) else answer
        except:
            actual_files = []

        # Normalize paths and convert to sets for comparison
        predicted_set = set(normalize_path(f) for f in predicted_files)
        actual_set = set(normalize_path(f) for f in actual_files)

        # Calculate F1 score
        if len(predicted_set) == 0 and len(actual_set) == 0:
            return 1.0
        if len(predicted_set) == 0 or len(actual_set) == 0:
            return 0.0

        true_positives = len(predicted_set & actual_set)
        precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0.0
        recall = true_positives / len(actual_set) if len(actual_set) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    # Reward that countr total no of turns with atleast 1 tool call
    def turns_with_tool_calls(completion: vf.types.Messages) -> float:
        """
        Count the total number of turns with atleast 1 tool call.
        """
        if not isinstance(completion, list):
            return 0.0

        count = 0
        for message in completion:
            if isinstance(message, dict) and "tool_calls" in message:
                tool_calls = message.get("tool_calls", [])
                if len(tool_calls) > 0:
                    count += 1

        if count == 0:
            return 0

        return 1

    # Reward to increase number of tool calls per turn
    def tool_call_count_per_turn(completion: vf.types.Messages) -> float:
        """
        Count the number of tool calls per turn.
        """
        if not isinstance(completion, list):
            return 0.0

        counts = []
        for message in completion:
            if isinstance(message, dict) and "tool_calls" in message:
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    counts.append(len(tool_calls))

        if len(counts) == 0:
            return 0.0

        avg_count = sum(counts) / len(counts)

        # clip it at 5 tool calls per turn
        avg_count = min(5.0, avg_count) / 5.0

        return avg_count

    # Define rubric with single F1 reward
    rubric = vf.Rubric(
        funcs=[file_localization_reward, turns_with_tool_calls, tool_call_count_per_turn],
        weights=[1.0, 1.0, 1.0],
    )

    # Common environment configuration
    env_config = {
        "parser": parser,
        "rubric": rubric,
        "max_turns": 8,
        **kwargs,
    }

    # Select dataset(s) based on mode
    if mode == "full":
        env_config["dataset"] = full_dataset
    elif mode == "train":
        env_config["dataset"] = train_dataset
    elif mode == "test":
        env_config["dataset"] = eval_dataset
    else:  # mode == "rl" (default)
        env_config["dataset"] = train_dataset
        env_config["eval_dataset"] = eval_dataset

    return SWEGrepEnv(**env_config)
