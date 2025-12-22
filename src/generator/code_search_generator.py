import copy
import json
import asyncio
from pyexpat.errors import messages
from socket import timeout
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
from omegaconf import DictConfig, OmegaConf
import traceback
import ray
import requests
from pathlib import Path
import os
import ast
import shutil
from pydantic import SecretStr
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
from openhands.tools.terminal import TerminalTool
from openhands.sdk.tool import Tool
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
from src.agent.agent import CustomAgent

from src.rewards import get_reward_function

from src.metrics.efficiency_metrics import compute_all_efficiency_metrics
from src.metrics.trajectory_metrics import compute_trajectory_metrics

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
    semantic_search_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: Union[TrajectoryID, Any],
    global_step: int,
    training_phase: Union[TrainingPhase, Any],
):
    import os
    import shutil
    
    instance_id = instance["instance_id"]
    repo_name = instance["repo"]
    commit_id = instance["base_commit"]
    
    worker_id = os.getpid()
    workspace = Path(f"/data/user_data/sanidhyv/tmp/testbed_{worker_id}/")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Track what we created for cleanup
    created_workspace = False
    created_index = False
    index_path = None
    
    try:
        status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)
        created_workspace = True
        print(f"[Worker {worker_id}] working_dir: {working_dir}")

        if training_phase == "eval":
            temperature = 0.6
        else:
            temperature = 1.0

        final_message = ""
        messages = []

        # Configure semantic search if enabled
        use_semantic_search = semantic_search_cfg.get("enabled", False)
        mcp_config = None
        agent_context = None

        print(f"[Episode {instance_id}] Semantic search config: {OmegaConf.to_yaml(semantic_search_cfg)}")
        print(f"[Episode {instance_id}] Semantic search enabled: {use_semantic_search}")

        if use_semantic_search:
            try:
                from openhands.sdk.context.skills import Skill
                from openhands.sdk import AgentContext

                # Get base path
                base_path = Path(generator_cfg.get("base_path", "/data/user_data/sanidhyv/agentic-code-search-oss"))
                skill_path = base_path / ".openhands" / "skills" / "semantic-search.md"
                
                print(f"[Episode {instance_id}] Looking for skill at: {skill_path}")
                
                if not skill_path.exists():
                    print(f"[Episode {instance_id}] ERROR: Skill file not found at {skill_path}, semantic search disabled")
                    use_semantic_search = False
                else:
                    print(f"[Episode {instance_id}] Loading skill from {skill_path}")
                    skill = Skill.load(str(skill_path))
                    print(f"[Episode {instance_id}] Skill loaded successfully")
                    
                    wrapper_path = base_path / "scripts/run_mcp_server_training.sh"
                    
                    print(f"[Episode {instance_id}] Looking for MCP wrapper at: {wrapper_path}")
                    
                    if not wrapper_path.exists():
                        print(f"[Episode {instance_id}] ERROR: MCP wrapper not found at {wrapper_path}")
                        use_semantic_search = False
                    else:
                        import stat
                        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC)
                        print(f"[Episode {instance_id}] MCP wrapper found and made executable")
                        
                        # Track index location for cleanup
                        from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
                        repo_commit_hash = get_repo_commit_hash(repo_name, commit_id)
                        index_path = Path(f"/data/user_data/sanidhyv/tmp/embedding_cache/{repo_commit_hash}")
                        
                        print(f"[Episode {instance_id}] Index will be stored at: {index_path}")
                        
                        mcp_config = {
                            "mcpServers": {
                                "semantic-code-search": {
                                    "command": "bash",
                                    "args": [str(wrapper_path)],
                                    "env": {
                                        "WORKSPACE_PATH": str(working_dir),
                                        "RAY_ADDRESS": os.environ.get("RAY_ADDRESS", "auto"),
                                        "PYTHONPATH": str(base_path),
                                    }
                                }
                            }
                        }
                        
                        print(f"[Episode {instance_id}] MCP config created: {mcp_config}")
                        
                        agent_context = AgentContext(skills=[skill])
                        print(f"[Episode {instance_id}] Agent context created with semantic search skill")
                        
                        # Mark that we'll create an index
                        if not index_path.exists():
                            created_index = True
                            print(f"[Episode {instance_id}] Will create new index (marked for cleanup)")
                        else:
                            print(f"[Episode {instance_id}] Index already exists (will not cleanup)")
                            
                        print(f"[Episode {instance_id}] ✓ Semantic search fully configured")
                        
            except Exception as e:
                print(f"[Episode {instance_id}] ERROR setting up semantic search: {e}")
                traceback.print_exc()
                use_semantic_search = False
                mcp_config = None
                agent_context = None

        # Agent creation - use CustomAgent with optional semantic search
        agent_kwargs = {
            "llm": LLM(
                usage_id="agent",
                model=litellm_model_name,
                base_url=litellm_base_url,
                api_key="sk-xxx",
                temperature=temperature,
                litellm_extra_body={
                    "return_token_ids": True,
                    "include_stop_str_in_output": True,
                }
            ),
            "tools": [Tool(name=TerminalTool.name)],
            "security_analyzer": None,
            "system_prompt_filename": os.path.join(os.path.dirname(__file__), "..", "prompts", "templates", "system_prompt.j2")
        }
        
        if use_semantic_search and mcp_config is not None and agent_context is not None:
            agent_kwargs["agent_context"] = agent_context
            agent_kwargs["mcp_config"] = mcp_config
            print(f"[Episode {instance_id}] Agent will be created WITH semantic search (mcp_config and agent_context)")
        else:
            print(f"[Episode {instance_id}] Agent will be created WITHOUT semantic search")
        
        print(f"[Episode {instance_id}] Creating agent with kwargs keys: {list(agent_kwargs.keys())}")
        agent = CustomAgent(**agent_kwargs)
        print(f"[Episode {instance_id}] Agent created successfully")

        conversation = Conversation(
            agent=agent,
            max_iteration_per_run=10,
            visualizer=None,
            workspace=str(working_dir),
        )
                
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "templates", "file_module.j2")
        input_message = get_instruction(instance, prompt_path, str(working_dir))
        
        # Truncate input if too long
        from transformers import AutoTokenizer
        MAX_INPUT_TOKENS = 12000
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                litellm_model_name.replace("litellm_proxy/", ""),
                trust_remote_code=True,
                cache_dir="/data/user_data/sanidhyv/.cache/huggingface"
            )
            input_tokens = tokenizer.encode(input_message)
            
            if len(input_tokens) > MAX_INPUT_TOKENS:
                print(f"[Episode {instance_id}] Input too long ({len(input_tokens)} tokens), truncating")
                input_tokens = input_tokens[:500] + input_tokens[-(MAX_INPUT_TOKENS-500):]
                input_message = tokenizer.decode(input_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"[Episode {instance_id}] Could not check input length: {e}")
        
        conversation.send_message(input_message)
        print(f"[Episode {instance_id}] Starting conversation...")
        logger.info("Conversation Starting")

        # Capture start time
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()

        conversation.run()

        messages = list(map(lambda event: event.model_dump(), conversation.state.events))
        final_message = get_agent_final_response(conversation.state.events)

        conversation.close()
        logger.info("Conversation Finished")
        print(f"[Episode {instance_id}] Conversation finished")

        # Capture end time
        end_time = time.time()
        end_timestamp = datetime.now().isoformat()
        wall_clock_duration = end_time - start_time

        additional_attr = {
            "wall_clock_duration": wall_clock_duration,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
        }

        return messages, final_message, additional_attr, use_semantic_search

    except Exception as e:
        print(f"[Episode {instance_id}] ERROR in init_and_run: {e}")
        traceback.print_exc()
        raise
        
    finally:
        # Cleanup workspace and index
        print(f"[Worker {worker_id}] Cleaning up episode {instance_id}")
        
        # 1. Clean up workspace (repo clone)
        if created_workspace:
            try:
                shutil.rmtree(workspace, ignore_errors=True)
                print(f"[Worker {worker_id}] Removed workspace: {workspace}")
            except Exception as e:
                print(f"[Worker {worker_id}] Warning: Could not remove workspace: {e}")
        
        # 2. Clean up semantic search index (if we created it)
        if created_index and index_path and index_path.exists():
            try:
                shutil.rmtree(index_path, ignore_errors=True)
                print(f"[Worker {worker_id}] Removed index: {index_path}")
            except Exception as e:
                print(f"[Worker {worker_id}] Warning: Could not remove index: {e}")
           
    
                
class CodeSearchGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        semantic_search_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        # Call parent constructor first
        super().__init__(
            generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name
        )

        self.http_endpoint_host = generator_cfg.get(
            "http_endpoint_host", "127.0.0.1"
        )
        self.http_endpoint_port = generator_cfg.get(
            "http_endpoint_port", 8000
        )
        self.base_url = f"http://{self.http_endpoint_host}:{self.http_endpoint_port}/v1/"
        logger.info(f"Using CodeSearchGenerator with model {model_name} at {self.base_url}")
        self.generator_cfg = generator_cfg
        self.semantic_search_cfg = semantic_search_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        # self.litellm_model_name = "openai/" + self.model_name
        self.litellm_model_name = "litellm_proxy/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError(
                "OpenhandsGenerator doesn't support custom chat template"
            )
            
        print(f"CodeSearchGenerator initialized with semantic_search enabled: {self.semantic_search_cfg.get('enabled', False)}")

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

            messages, final_message, additional_attr, used_semantic_search = await init_and_run.remote(
                instance,
                self.litellm_model_name,
                self.base_url,
                self.generator_cfg,
                self.semantic_search_cfg,
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
            additional_attr = {
                "wall_clock_duration": 0.0,
                "start_timestamp": None,
                "end_timestamp": None,
            }
            used_semantic_search = False

        # print("=" * 100)
        # print("Conversation finished. Got the following LLM messages:")
        # for i, message in enumerate(messages):
        #     print(f"Message {i}: {str(message)[:100]}")
        # print("Final message:", final_message)

        # Reward Manager
        reward = 0
        reward_dict = {}

        for reward_fn_args in self.generator_cfg.reward:
            try:
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

        if final_message == "":
            reward = -10.0

        print(f"Reward details: {reward_dict}, Total reward: {reward}")

        # Compute Trajectory Metrics
        # Filter out non-efficiency keys before passing to compute_all_efficiency_metrics
        efficiency_keys = {'wall_clock_duration', 'start_timestamp', 'end_timestamp'}
        efficiency_attr = {k: v for k, v in additional_attr.items() if k in efficiency_keys}
        
        efficiency_metrics = compute_all_efficiency_metrics(
            messages=messages,
            **efficiency_attr,
        )

        trajectory_metrics = compute_trajectory_metrics(messages)

        metrics_dict = {
            **efficiency_metrics,
            **trajectory_metrics,
            "used_semantic_search": used_semantic_search,
        }

        print(f"Trajectory metrics: {metrics_dict}")

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
                        None,
                        trajectory_metrics
                    )
                )

        else:
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
            raw_final_message = final_message
            matches = re.findall(r"```(.*?)```", final_message, re.DOTALL)
            parsed_final_message = matches[0] if matches else final_message

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
            # "trajectory_ids": out_trajectory_ids,
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
            # "is_last_step": is_last_step,
            **tracked_metrics,
        }

        return generator_output