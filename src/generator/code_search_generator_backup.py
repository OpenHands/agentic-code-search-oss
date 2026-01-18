import copy
import json
import asyncio
from pyexpat.errors import messages
from socket import timeout
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
from omegaconf import DictConfig
import traceback
import ray
import requests
from pathlib import Path
import os
import ast
import time
from datetime import datetime
import numpy as np
from collections import defaultdict

import re
import signal
from contextlib import contextmanager

import gcsfs
import fsspec

from skyrl_train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    GeneratorOutput,
    GeneratorInput,
)
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    encode_messages_subset,
)
from openhands.tools.preset.default import get_default_agent

from openhands.sdk.conversation.response_utils import get_agent_final_response
from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools
from openhands.tools.preset.planning import get_planning_tools
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool
from openhands.tools.terminal import TerminalTool
from openhands.sdk.tool import Tool, register_tool
from openhands.sdk import (
    Agent,
    LLM,
    Event,
    Conversation,
    RemoteConversation,
    LLMConvertibleEvent,
    get_logger,
)
from openhands.sdk.event import ActionEvent
from src.tools.localization_finish import LocalizationFinishAction, LocalizationFinishTool
from src.prompts.prompt_builder import get_instruction
from src.utils.instance import clone_instance
from src.agent.agent import CustomAgent

from src.rewards import get_reward_function
from src.tools import TOOL_REGISTRY

from src.metrics.efficiency_metrics import compute_all_efficiency_metrics
from src.metrics.trajectory_metrics import compute_trajectory_metrics

import logging
import signal

logger = get_logger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)

file_path = os.path.dirname(__file__)

def get_structured_locations(events: List[Event]) -> Optional[List[Dict[str, Any]]]:
    """Extract structured locations from LocalizationFinishAction in events.
    Args:
        events: List of conversation events to search through.
    Returns:
        List of location dicts with 'file', 'class', 'function' keys, or None if not found.
    """
    # Find the last LocalizationFinishAction
    cnt = [1 for event in events if isinstance(event, ActionEvent) and event.source == "agent" and isinstance(event.action, LocalizationFinishAction)]
    cnt = sum(cnt)
    if cnt != 1: # the localization finish tool must be called exactly once.
        return None
    for event in reversed(events):
        if (
            isinstance(event, ActionEvent)
            and event.source == "agent"
            and isinstance(event.action, LocalizationFinishAction)
        ):
            # Extract structured locations from the action
            locations = []
            for loc in event.action.locations:
                locations.append({
                    "file": loc.file,
                    "class_name": loc.class_name,
                    "function_name": loc.function_name,
                })
            return locations
    return None

@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    litellm_base_url: dict,
    generator_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: Union[TrajectoryID, Any],
    global_step: int,
    training_phase: Union[TrainingPhase, Any],
):

    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance.get("base_commit", None)
    if "use_patch" in instance and instance["use_patch"]:
        patch = instance["patch"]
    else:
        patch = None
    
    # Avoid collisions in /tmp testbed directories
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(f"/scratch/workspace/{uuid_str}/")
    status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace, patch)

    if training_phase == "eval":
        temperature = 0.6
    else:
        temperature = 1.0

    final_message = ""
    structured_locations = None
    messages = []

    # for tool_name in generator_cfg.tools:
    #     if tool_name in TOOL_REGISTRY:
    #         register_tool(tool_name, TOOL_REGISTRY[tool_name])
    #     else:
    #         raise ValueError(f"Tool {tool_name} does not exist in the registry")

    # tools = [
    #     Tool(name=tool_name) for tool_name in generator_cfg.tools
    # ]

    register_tool(LocalizationFinishTool.name, LocalizationFinishTool)
    tools = [
        # Tool(name=GlobTool.name),
        # Tool(name=GrepTool.name),
        Tool(name=TerminalTool.name),
        Tool(name="localization_finish"),
    ]

    # Get prompt paths from config (path-independent)
    prompts_base_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    system_prompt_path = os.path.join(prompts_base_dir, generator_cfg.prompts.system_prompt)
    user_prompt_path = os.path.join(prompts_base_dir, generator_cfg.prompts.user_prompt)

    assert os.path.exists(system_prompt_path), f"System prompt file {system_prompt_path} does not exist"
    assert os.path.exists(user_prompt_path), f"User prompt file {user_prompt_path} does not exist"

    agent = CustomAgent(
        llm=LLM(
            usage_id="agent",
            model=litellm_model_name,
            base_url=litellm_base_url,
            api_key="sk-xxx",
            temperature=temperature,
            litellm_extra_body={
                "return_token_ids": True,
                "include_stop_str_in_output": False,
                "chat_template_kwargs": {
                    "add_generation_prompt": True,
                    # "enable_thinking": True
                }
            }
        ),
        tools=tools,
        # security_analyzer=None,
        system_prompt_filename=system_prompt_path
    )

    conversation = Conversation(
        agent=agent,
        max_iteration_per_run=generator_cfg.max_turns,
        visualizer=None,
        workspace=str(working_dir),
    )
    input_message = get_instruction(instance, user_prompt_path, str(working_dir))

    # Capture start time
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    try:
        conversation.send_message(input_message)
        logger.info("Conversation Starting")
        conversation.run()
        messages = list(map(lambda event: event.model_dump(), conversation.state.events))
        final_message = get_agent_final_response(conversation.state.events)
        structured_locations = get_structured_locations(conversation.state.events)
    except Exception as e:
        logger.error(f"Error during conversation: {str(e)}", exc_info=True)
        try:
            messages = list(map(lambda event: event.model_dump(), conversation.state.events))
            final_message = get_agent_final_response(conversation.state.events)
            structured_locations = get_structured_locations(conversation.state.events)
        except Exception as e:
            logger.error(f"Error during final message extraction in err'ed rollout: {str(e)}", exc_info=True)
            messages = []
            final_message = ""
    finally:
        # Capture end time
        try:
            if workspace.exists():
                os.system(f"rm -rf {str(workspace)}")
                logger.info(f"Removed workspace {str(workspace)}")
            conversation.close()
        except Exception as _:
            pass
        logger.info("Conversation Finished")
        end_time = time.time()
        end_timestamp = datetime.now().isoformat()
        wall_clock_duration = end_time - start_time

        additional_attr = {
            "wall_clock_duration": wall_clock_duration,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp
        }

    # NOTE: Hard-coded final message to ensure all rollouts that don't call the custom finish tool have reward == 0
    return messages, "", structured_locations, additional_attr


class CodeSearchGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
        step_wise: bool = False,
    ):
        # Call parent constructor first
        super().__init__(
            generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name
        )
        # logger.info(f"Engine design: {inference_engine_client.engines[0].inference_engine_actor.openai_serving_chat}")

        self.http_endpoint_host = generator_cfg.get(
            "http_endpoint_host", "127.0.0.1"
        )
        self.http_endpoint_port = generator_cfg.get(
            "http_endpoint_port", 8000
        )
        self.base_url = f"http://{self.http_endpoint_host}:{self.http_endpoint_port}/v1/"
        logger.info(f"Using CodeSearchGenerator with model {model_name} at {self.base_url}")
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # self.litellm_model_name = "openai/" + self.model_name
        self.litellm_model_name = "openai/" + self.model_name

        # if self.generator_cfg.chat_template.name_or_path is not None:
        #     raise NotImplementedError(
        #         "OpenhandsGenerator doesn't support custom chat template"
        #     )

        self.step_wise = step_wise
        self.max_train_length = generator_cfg.get(
            "max_train_length", 100000
        )

    def sanity_check_last_step(self, token_messages):
        # Checks if the tool call formatting is correct in the last step from the detokenized response str of the last turn
        if len(token_messages) == 0:
            return False
        response_token_ids = token_messages[-1]["response_token_ids"]
        last_response_str: str = self.tokenizer.decode(response_token_ids, skip_special_tokens=False)
        # First sanity check -- verify if there is exactly one <tool_call> and one </tool_call> in response (if there are multiple tool calls give 0 reward regardless of correctness)
        cnt_tool_call = last_response_str.count("<tool_call>")
        cnt_tool_end = last_response_str.count("</tool_call>")
        if cnt_tool_call != 1 or cnt_tool_end != 1:
            return False
        # Second sanity check -- verify if the <|im_end|> is present exactly once
        elif last_response_str.count("<|im_end|>") != 1:
            return False
        # Third sanity check -- verify if there is no non-whitespace text after </tool_call> and before <|im_end|>
        else:
            portion = last_response_str.split("</tool_call>")[1].split("<|im_end|>")[0]
            if portion.strip() != "":
                return False
        return True

    async def code_search_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]], Optional[Dict[str, Any]]]:
        # sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        instance = env_extras
        error = None
        try:
            messages, final_message, structured_locations, additional_attr = await init_and_run.remote(
                instance,
                self.litellm_model_name,
                self.base_url,
                self.generator_cfg,
                "swe-gym",
                sampling_params,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )
        except Exception as e:
            logger.error(f"Critical Error in conversation: {str(e)}", exc_info=True)
            # TODO properly handle this
            error = str(e) + "\n" + traceback.format_exc()
            messages = []
            final_message = ""
            structured_locations = None
            additional_attr = {
                "wall_clock_duration": 0.0,
                "start_timestamp": None,
                "end_timestamp": None
            }

        # print("=" * 100)
        # print("Conversation finished. Got the following LLM messages:")
        # for i, message in enumerate(messages):
        #     print(f"Message {i}: {str(message)[:100]}")
        # print("Final message:", final_message)

        # Run sanity check before computing the reward so that the logged metrics reflect the actual reward received in training
        token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
        trajectory_exhausted_steps = structured_locations is None and len(token_messages) >= self.generator_cfg.max_turns

        # The agent called the custom finish tool but there were some sanity check issues like calling the tool multiple times, have extra text after ending tool-call, calling this tool in parallel with other tools etc. Give 0 reward in such cases
        if structured_locations is not None and self.sanity_check_last_step(token_messages) == False:
            # If sanity check fails, set structured_locations to None so that reward fns that depend on it give 0 reward
            structured_locations = None
            final_message = ""

        # Reward Manager
        reward = 0
        reward_dict = {}

        for reward_fn_args in self.generator_cfg.reward:
            try:
                input_args = {
                    "final_message": final_message,
                    "messages": messages,
                    "instance": instance,
                    "structured_locations": structured_locations
                }

                reward_fn = get_reward_function(reward_fn_args["fn"])

                input_args = {
                    **input_args, 
                    **reward_fn_args.get("args", {})
                    }

                reward_outputs = reward_fn(**input_args)
                if isinstance(reward_outputs, tuple):
                    reward_value, reward_items = reward_outputs
                else:
                    reward_value = reward_outputs
                    reward_items = {reward_fn_args["fn"]: reward_value}
            except Exception as e:
                logger.error(f"Error in computing reward {reward_fn_args['fn']}: {e}", exc_info=True)
                reward_value = 0.0
                reward_items = {reward_fn_args["fn"]: reward_value}

            reward += reward_value

            reward_dict = {
                **reward_dict,
                **reward_items,
            }

        # Compute Trajectory Metrics
        efficiency_metrics = compute_all_efficiency_metrics(
            messages=messages,
            **additional_attr,
        )

        trajectory_metrics = compute_trajectory_metrics(messages)

        metrics_dict = {
            **efficiency_metrics,
            **trajectory_metrics
        }

        print(f"Total reward: {reward}\nReward details: {reward_dict}\nTrajectory metrics: {metrics_dict}")

        token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
        rollout_list = []
        num_steps = len(token_messages)
        if len(token_messages) > 0:
            if self.step_wise:
                for idx, message in enumerate(token_messages):
                    current_prompt_ids = message["prompt_token_ids"]
                    current_response_ids = message["response_token_ids"]

                    rollout_list.append(
                        (
                            current_response_ids,
                            reward,
                            "complete",
                            [1]*len(current_response_ids),
                            current_prompt_ids,
                            None,
                            trajectory_metrics
                        )
                    )
            else:
                # Max Sequence for training
                max_train_len = self.max_train_length

                current_prompt_ids = token_messages[0]["prompt_token_ids"]
                ending_prompt_ids = token_messages[-1]["prompt_token_ids"]
                ending_response_ids = token_messages[-1]["response_token_ids"]
                current_response_ids = ending_prompt_ids + ending_response_ids
                current_response_ids = current_response_ids[len(current_prompt_ids):]

                max_response_len = max_train_len - len(current_prompt_ids)
                # make mask of 0 for everything inside <|im_start|> 
                # and assistant and 1 elsewhere 
                start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                end_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
                end_of_turn_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                mask = []
                found_role_switch = False
                inside = False
                idx = 0
                while idx < len(current_response_ids):
                    token_id = current_response_ids[idx]
                    if not inside:
                        mask.append(1)
                        idx += 1
                        if token_id == end_of_turn_token_id:
                            inside = True
                    else:
                        if token_id == start_token_id:
                            inside = True
                            mask.append(0)
                            idx += 1
                        elif token_id == end_token_id and found_role_switch:
                            inside = False
                            mask.append(0)
                            mask.append(0)
                            idx += 2
                        else:
                            mask.append(0)
                            idx += 1

                        if token_id == start_token_id:
                            found_role_switch = True
                        else:
                            found_role_switch = False

                # mask zero out everything beyond max_response_len
                # Don't truncate the response, just mask out the loss
                if len(current_response_ids) > max_response_len:
                    for i in range(max_response_len, len(current_response_ids)):
                        mask[i] = 0
                
                # mask loss completely from trajectories that exhausted all steps without calling the custom finish tool
                if trajectory_exhausted_steps:
                    logger.info("Trajectory exhausted all steps without calling the custom finish tool. Masking out loss from this rollout.")
                    for i in range(len(mask)):
                        mask[i] = 0 

                # If the trajectory did not call the custom finish tool within max turns, it did not complete successfully, mask its loss
                # if num_steps >= self.generator_cfg.max_turns and (structured_locations is None or len(structured_locations) == 0):
                #     for i in range(len(mask)):
                #         mask[i] = 0 # mask loss completely from trajectories that did not complete within max_turns 

                rollout_list.append(
                    (
                        current_response_ids,
                        reward,
                        "complete",
                        mask,
                        current_prompt_ids,
                        None,
                        trajectory_metrics
                    )
                )

        else:
            # Ideally the code should not reach here
            response_ids = [151643]
            stop_reason = "error"
            loss_mask = [1]
            initial_input_ids = [151643]
            trajectory_metrics = {}  # Empty metrics for error case
            rollout_list.append(
                (response_ids, reward, stop_reason, loss_mask, initial_input_ids, None, trajectory_metrics)
            )

        # Add "/" at the end of traj_dir if not present
        if not self.generator_cfg.traj_dir.endswith("/"):
            self.generator_cfg.traj_dir += "/"

        path = self.generator_cfg.traj_dir + f"step_{batch_metadata.global_step}/{batch_metadata.training_phase}/"
        # Check if traj_dir is a gcs path
        if path.startswith("gs://"):
            use_gcs = True
            fs = gcsfs.GCSFileSystem()
        else:
            use_gcs = False
            fs = fsspec.filesystem("file")
            # Pre-create directory to avoid race conditions with parallel workers
            os.makedirs(path, exist_ok=True)
        
        instance_id = env_extras["instance_id"]

        if error is not None:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.error"
            filename_path = path + filename
            print(f"Saving error to {filename_path}")
            if use_gcs == False:
                os.makedirs(os.path.dirname(filename_path), exist_ok=True)
            with fs.open(filename_path, "w", auto_mkdir=True) as f:
                f.write(error)
        else:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
            filename_path = path + filename

            if use_gcs == False:
                os.makedirs(os.path.dirname(filename_path), exist_ok=True)

            # get everything between ```` with regex
            try:
                raw_final_message = structured_locations if structured_locations is not None else {}
            except Exception as e:
                raw_final_message = ""
            matches = re.findall(r"```(.*?)```", final_message, re.DOTALL)
            parsed_final_message = matches[-1] if matches else final_message

            result_dict = {
                "instance_id": instance_id,
                "target": env_extras["target"],
                "total_reward": reward,
                "reward_dict": reward_dict,
                "parsed_final_message": parsed_final_message,
                "raw_final_message": raw_final_message,
                "messages": messages,
                "metrics_dict": metrics_dict,
            }

            print(f"Saving trajectory to {filename_path}")
            with fs.open(filename_path, "w", auto_mkdir=True) as f:
                json.dump(result_dict, f, indent=2) #, sort_keys=True, ensure_ascii=False)

        return [rollout_list, reward_dict, metrics_dict]

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.backend, self.generator_cfg.sampling_params
        )

        task_rollouts = []
        for i in range(len(prompts)):
            rollout = self.code_search_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            
            task_rollouts.append(rollout)

        collected_task_rollouts = await asyncio.gather(*task_rollouts)

        all_outputs = [rollout[0] for rollout in collected_task_rollouts]
        rewards_dict = [rollout[1] for rollout in collected_task_rollouts]
        metrics_dict = [rollout[2] for rollout in collected_task_rollouts]

        responses = sum([[output[0] for output in step_outputs] for step_outputs in all_outputs], [])
        rewards = sum([[output[1] for output in step_outputs] for step_outputs in all_outputs], [])
        stop_reasons = sum([[output[2] for output in step_outputs] for step_outputs in all_outputs], [])
        loss_masks = sum([[output[3] for output in step_outputs] for step_outputs in all_outputs], [])
        prompt_token_ids = sum([[output[4] for output in step_outputs] for step_outputs in all_outputs], [])

        out_trajectory_ids = []
        is_last_step = []
        for i in range(len(all_outputs)):
            step_outputs = all_outputs[i]
            for step_id in range(len(step_outputs)):
                out_trajectory_id = copy.deepcopy(trajectory_ids[i])
                out_trajectory_id.step = step_id
                out_trajectory_ids.append(out_trajectory_id.instance_id)
                is_last_step.append(step_id == len(step_outputs) - 1)

        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        tracked_metrics = {}

        # Aggregate Rewards and Metrics
        for tracker_name, tracker_dict in zip(
            ["reward", "metrics"], [rewards_dict, metrics_dict]
        ):
            for tracker_dict_item in tracker_dict:
                for k, v in tracker_dict_item.items():
                    # Check if v is numeric
                    if not isinstance(v, (int, float)):
                        continue
                    if f"{tracker_name}/{k}" not in tracked_metrics:
                        tracked_metrics[f"{tracker_name}/{k}"] = []
                    tracked_metrics[f"{tracker_name}/{k}"].append(v)
        
        # Average all tracked metrics
        for k, v in tracked_metrics.items():
            tracked_metrics[k] = sum(v) / len(v)

        generator_output: GeneratorOutput = {
            "trajectory_ids": out_trajectory_ids,
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            "is_last_step": is_last_step,
            **tracked_metrics,
        }

        return generator_output