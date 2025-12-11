import json
import asyncio
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

import signal
from contextlib import contextmanager

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
from openhands.sdk import (
    Agent,
    LLM,
    Event,
    Conversation,
    RemoteConversation,
    LLMConvertibleEvent,
    get_logger,
)

from src.prompts.prompt_builder import get_instruction
from src.utils.instance import clone_instance

from src.rewards import get_reward_function

from src.metrics.efficiency_metrics import compute_all_efficiency_metrics

import logging
import signal

logger = get_logger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.ERROR)

file_path = os.path.dirname(__file__)

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
    commit_id = instance["base_commit"]
    
    # Avoid collisions in /tmp testbed directories
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(f"/scratch/lsutawik/tmp/testbed/{uuid_str}/")
    status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)

    if training_phase == "eval":
        temperature = 0.6
    else:
        temperature = 1.0

    final_message = ""
    messages = []

    agent = Agent(
        llm=LLM(
            service_id="agent",
            model=litellm_model_name,
            base_url=f"http://localhost:8080/v1/",
            api_key="sk-xxx",
            temperature=temperature,
            litellm_extra_body={
                "return_token_ids": True,
                "include_stop_str_in_output": True,
            }
        ),
        tools=get_planning_tools(),
        security_analyzer=None,
    )

    conversation = Conversation(
        agent=agent,
        max_iteration_per_run=8,
        visualizer=None,
        workspace=str(working_dir),
    )
    # prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "templates", "file_localization.j2")
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "templates", "file_module.j2")
    input_message = get_instruction(instance, prompt_path, str(working_dir))
    conversation.send_message(input_message)

    logger.info("Conversation Starting")

    # Capture start time
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    conversation.run()

    messages = list(map(lambda event: event.model_dump(), conversation.state.events))
    final_message = get_agent_final_response(conversation.state.events)

    conversation.close()
    logger.info("Conversation Finished")

    # Capture end time
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    wall_clock_duration = end_time - start_time

    additional_attr = {
        "wall_clock_duration": wall_clock_duration,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp
    }

    return messages, final_message, additional_attr


class CodeSearchGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        # Call parent constructor first
        super().__init__(
            generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name
        )

        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # self.litellm_model_name = "openai/" + self.model_name
        self.litellm_model_name = "litellm_proxy/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError(
                "OpenhandsGenerator doesn't support custom chat template"
            )

    async def code_search_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:
        # sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        instance = env_extras
        error = None
        try:

            messages, final_message, additional_attr = await init_and_run.remote(
                instance,
                self.litellm_model_name,
                # sweagent_config,
                self.base_url,
                self.generator_cfg,
                # env_extras["data_source"],
                "swe-gym",
                sampling_params,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )
        except Exception as e:
            logger.error(f"Error in starting conversation: {e}", exc_info=True)
            # TODO properly handle this
            error = str(e) + "\n" + traceback.format_exc()
            messages = []
            final_message = ""
            wall_clock_duration = 0.0
            start_timestamp = None
            end_timestamp = None

        # print("=" * 100)
        # print("Conversation finished. Got the following LLM messages:")
        # for i, message in enumerate(messages):
        #     print(f"Message {i}: {str(message)[:100]}")
        # print("Final message:", final_message)

        # Reward Manager
        reward = 0
        reward_dict = {}

        for reward_fn_args in self.generator_cfg.reward:
            input_args = {
                "final_message": final_message,
                "messages": messages,
                "instance": instance,
            }

            reward_fn = get_reward_function(reward_fn_args["fn"])

            input_args = {
                **input_args, 
                **reward_fn_args.get("args", {})
                }

            try:
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

        print(f"Reward details: {reward_dict}, Total reward: {reward}")

        # Compute efficiency metrics
        if messages != []:
            efficiency_metrics = compute_all_efficiency_metrics(
                messages=messages,
                **additional_attr,
            )
        else:
            efficiency_metrics = {
                "tokens": 0,
                "steps": 0,
                "avg_tool_calls_per_step": 0.0,
                "wall_clock_duration": additional_attr["wall_clock_duration"],
            }

        print(f"Efficiency metrics: {efficiency_metrics}")

        #     # print("=" * 100)
        #     # print("Conversation finished. Got the following LLM messages:")
        #     # for i, message in enumerate(messages):
        #     #     print(f"Message {i}: {str(message)[:200]}")

        token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
        rollout_list = []
        gamma = 0.9
        num_steps = len(token_messages)
        if len(token_messages) > 0:
            for idx, message in enumerate(token_messages):
                current_prompt_ids = message["prompt_token_ids"]
                current_response_ids = message["response_token_ids"]
                step_reward = reward * gamma**(num_steps - idx - 1)

                rollout_list.append(
                    (
                        current_response_ids,
                        step_reward,
                        "complete",
                        [1]*len(current_response_ids),
                        current_prompt_ids,
                        None
                    )
                )

        else:
            response_ids = [151643]
            stop_reason = "error"
            loss_mask = [1]
            initial_input_ids = [151643]
            rollout_list.append(
                (response_ids, reward, stop_reason, loss_mask, initial_input_ids, None)
            )


        #     stop_reason = "complete"
        #     prompt_ids_list = []
        #     response_ids_list = []
        #     trajectory_ids_list = []
        #     loss_mask = []
        #     initial_input_len = 0
        #     past_trajectory_len = 0
        #     for idx, message in enumerate(token_messages):
        #         current_prompt_ids = message["prompt_token_ids"]
        #         current_response_ids = message["response_token_ids"]

        #         prompt_ids_list.append(current_prompt_ids)
        #         response_ids_list.append(current_response_ids)
        #         trajectory_ids_list.append(current_prompt_ids + current_response_ids)

        #         if idx == 0:
        #             initial_input_ids = current_prompt_ids
        #             initial_input_len = len(initial_input_ids)
        #             loss_mask = [1] * len(current_response_ids)
        #             continue

        #         past_trajectory_len = len(trajectory_ids_list[idx-1])
        #         past_response_len = len(response_ids_list[idx-1])
        #         current_prompt_len = len(current_prompt_ids)
        #         current_response_len = len(current_response_ids)

        #         # print("idx:", idx)
        #         # print("initial_input_ids_len:", initial_input_len)
        #         # print("past_trajectory_len:", past_trajectory_len)
        #         # print("past_response_len:", past_response_len)
        #         # print("current_prompt_len:", current_prompt_len)
        #         # print("current_response_len:", current_response_len)

        #         # past_prompt_len = len(prompt_ids_list[idx-1]) if idx > 0 else 0
        #         past_response_observation_ids = current_prompt_ids[past_trajectory_len:]
        #         past_response_observation_len = len(past_response_observation_ids)
        #         # print("past_response_observation_len:", past_response_observation_len)
        #         loss_mask.extend([0] * past_response_observation_len)
        #         loss_mask.extend([1] * current_response_len)
            
        #     response_ids = current_prompt_ids[initial_input_len:] + current_response_ids
        #     assert len(response_ids) == len(loss_mask), f"Response ids length {len(response_ids)} != loss mask length {len(loss_mask)}"

        path = Path(self.generator_cfg.traj_dir) / f"step_{batch_metadata.global_step}" / batch_metadata.training_phase
        path.mkdir(parents=True, exist_ok=True)
        instance_id = env_extras["instance_id"]

        if error is not None:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.error"
            filename_path = path / filename
            print(f"Saving error to {filename_path}")
            with open(filename_path, "w") as f:
                f.write(error)
        else:
            filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
            filename_path = path / filename

            result_dict = {
                "target": env_extras["target"],
                "reward_dict": reward_dict,
                "final_message": final_message,
                "messages": messages,
                "efficiency_metrics": efficiency_metrics,
            }

            print(f"Saving trajectory to {filename_path}")
            with open(filename_path, "w") as f:
                json.dump(result_dict, f, indent=2) #, sort_keys=True, ensure_ascii=False)

        # return (response_ids, reward, stop_reason, loss_mask, initial_input_ids, None)
        return [rollout_list[-1], reward_dict]

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

        reward_dict = [rollout[1] for rollout in collected_task_rollouts]
        all_outputs = [rollout[0] for rollout in collected_task_rollouts]

        # Filter out the `None` entries, which means that trajectory generation failed
        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [
            output[4] for output in all_outputs if output[0] is not None
        ]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        # This is supposedly where I can add custom wandb logs
        reward_metrics = {}
        for reward_dict_item in reward_dict:
            for k, v in reward_dict_item.items():
                if f"reward/{k}" not in reward_metrics:
                    reward_metrics[f"reward/{k}"] = []
                reward_metrics[f"reward/{k}"].append(v)

        # Average the reward metrics over the batch
        for k, v in reward_metrics.items():
            reward_metrics[k] = sum(v) / len(v)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            **reward_metrics,
        }

        return generator_output
