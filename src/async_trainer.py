import numpy as np
import torch

from loguru import logger
from typing import List

from skyrl_train.utils import ppo_utils, trainer_utils

from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.generators.base import GeneratorOutput
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer, GeneratedOutputGroup


def patched_concatenate_generator_outputs(generator_outputs: List[GeneratorOutput]) -> GeneratorOutput:
    """
    Concatenate the generator outputs of multiple batches.

    We only aggregate rollout metrics the can deduced by responses and rewards, but not
    those that use `env_metrics` or `env_classes`.
    """
    assert len(generator_outputs) > 0
    has_rollout_logprobs = [output.get("rollout_logprobs") is not None for output in generator_outputs]
    if any(has_rollout_logprobs) and not all(has_rollout_logprobs):
        raise ValueError(
            "generator outputs are expected to all have null rollout_logprobs or all non-null, but received a mix"
        )
    result: GeneratorOutput = {
        "prompt_token_ids": sum([output["prompt_token_ids"] for output in generator_outputs], []),
        "response_ids": sum([output["response_ids"] for output in generator_outputs], []),
        "rewards": sum([output["rewards"] for output in generator_outputs], []),
        "loss_masks": sum([output["loss_masks"] for output in generator_outputs], []),
        "stop_reasons": (
            sum([output["stop_reasons"] for output in generator_outputs], [])
            if "stop_reasons" in generator_outputs[0] and generator_outputs[0]["stop_reasons"] is not None
            else None
        ),
        "rollout_logprobs": (
            sum([output["rollout_logprobs"] for output in generator_outputs], [])
            if generator_outputs[0]["rollout_logprobs"] is not None
            else None
        ),
        "trajectory_ids": sum([output["trajectory_ids"] for output in generator_outputs], []),
        "is_last_step": sum([output["is_last_step"] for output in generator_outputs], []),
    }

    # propagate additional keys with list values as-is
    additional_keys = [
        key for key in generator_outputs[0] if key not in result and isinstance(generator_outputs[0][key], (int, float))
    ]
    additional_result = {}
    if len(additional_keys):
        logger.info(f"Attempting to concatenate values for additional keys {additional_keys}")
    for key in additional_keys:
        try:
            # result[key] = sum([generator_output[key] for generator_output in generator_outputs], [])
            additional_result[key] = np.mean([generator_output[key] for generator_output in generator_outputs]).item()
        except Exception as e:
            logger.error(f"Error in aggregating key {key}: {e}", exc_info=True)

    # Re-aggregate rollout metrics
    rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
    result["rollout_metrics"] = {**rollout_metrics, **additional_result}

    # Validate the generator output using the number of prompts
    # Import here to avoid circular dependency.
    from skyrl_train.utils.trainer_utils import validate_generator_output

    # print("trajectory_ids", result["trajectory_ids"])
    # print("rewards", result["rewards"])
    # print("is_last_step", result["is_last_step"])

    num_prompts = len(result["prompt_token_ids"])
    validate_generator_output(num_prompts, result)

    return result

class CustomFullyAsyncRayPPOTrainer(FullyAsyncRayPPOTrainer):

    def convert_generation_group_mini_batch_to_training_input(
        self, cur_generation_group_mini_batch: List[GeneratedOutputGroup]
    ) -> TrainingInputBatch:
        """Given a mini-batch of generated groups, concatenate them into a single GeneratorOutput, then convert to a TrainingInputBatch."""
        generator_outputs = []
        uids = []
        stalenesses = []
        staleness_violation_count = 0
        group_size = len(cur_generation_group_mini_batch[0].generator_output["response_ids"])
        for cur_generated_output_group in cur_generation_group_mini_batch:
            cur_staleness = self.global_step - cur_generated_output_group.global_step_when_scheduled
            stalenesses.append(cur_staleness)
            generator_outputs.append(cur_generated_output_group.generator_output)
            uids.extend([cur_generated_output_group.uid] * group_size)

            # Check staleness violation.
            if cur_staleness > self.max_staleness_steps:
                # TODO(Charlie): should we drop, drop and resample, or just log?
                logger.warning(
                    "Staleness control violated despite using AsyncStalenessManager: "
                    f"cur_staleness={cur_staleness}, max_staleness_steps={self.max_staleness_steps}.\n"
                    "If this happens too often, consider increasing max_staleness_steps, adjusting "
                    "trainer.fully_async.num_parallel_generation_workers, or adjusting generation-training GPU allocation.\n"
                    "See https://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html#async-staleness-manager for more details."
                )
                staleness_violation_count += 1

        generator_output = patched_concatenate_generator_outputs(generator_outputs)
        assert generator_output["rollout_metrics"] is not None, "Rollout metrics should be non-null."
        self.all_metrics.update(generator_output["rollout_metrics"])

        # Log staleness statistics for this step
        self.all_metrics.update(
            {
                "async/staleness_mean": sum(stalenesses) / len(stalenesses),
                "async/staleness_max": max(stalenesses),
                "async/staleness_min": min(stalenesses),
                "async/staleness_ratio": sum(1 for s in stalenesses if s > 0) / len(stalenesses),
                "async/staleness_violation_count": staleness_violation_count,
            }
        )

        # Convert rewards to per-token form and compute reward metrics before training conversion
        uids = generator_output["trajectory_ids"]
        step_wise_training = self.cfg.trainer.step_wise_training
        self.cfg.trainer.step_wise_training = False
        generator_output = self.postprocess_generator_output(generator_output, uids)

        # print example just for debugging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Example generated: {vis}")

        # return self.convert_to_training_input(generator_output, uids)
        training_input = self.convert_to_training_input(generator_output, uids)
        self.cfg.trainer.step_wise_training = step_wise_training
        return training_input

    # @torch.no_grad()
    # def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
    #     """Calculate advantages and returns for the data batch.

    #     Expects:
    #         - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
    #         - `["response_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
    #         - `["loss_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
    #         - `["values"]`: Float[torch.Tensor, "batch_size seqlen"]
    #         - `["rewards"]`: Float[torch.Tensor, "batch_size seqlen"]
    #         - `.metadata["uids"]`: List[str]

    #     Adds:
    #         - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
    #         - `["returns"]`: Float[torch.Tensor, "batch_size seqlen"]
    #     """
    #     token_level_rewards = data["rewards"]

    #     if self.cfg.trainer.step_wise_training:
    #         is_last_step = data["is_last_step"].bool()
    #         print("is_last_step", is_last_step)
    #         response_mask = data["response_mask"]
    #         index = np.array(data.metadata["uids"])
    #         print("index", index)
    #         adv_estimator = self.cfg.trainer.algorithm.advantage_estimator
    #         config = self.cfg.trainer.algorithm
    #         values = data["values"]
    #         gamma = self.cfg.trainer.algorithm.gamma
    #         lambd = self.cfg.trainer.algorithm.lambd
    #         grpo_norm_by_std = self.cfg.trainer.algorithm.grpo_norm_by_std
    #         last_step_rewards = token_level_rewards[is_last_step]
    #         # compatible with any advantage estimator
    #         last_step_advantages, last_step_returns = ppo_utils.compute_advantages_and_returns(
    #             token_level_rewards=last_step_rewards,
    #             response_mask=response_mask[is_last_step],
    #             index=index[is_last_step.cpu().numpy()],
    #             adv_estimator=adv_estimator,
    #             values=values[is_last_step] if values is not None else None,
    #             config=config,
    #             gamma=gamma,
    #             lambd=lambd,
    #             grpo_norm_by_std=grpo_norm_by_std,
    #         )
    #         traj_ids = (
    #             torch.cat([torch.tensor([False], device=is_last_step.device), is_last_step[:-1]]).int().cumsum(dim=0)
    #         )
    #         print(f"traj_ids: {traj_ids}")
    #         num_groups = traj_ids[-1].item() + 1
    #         assert num_groups == len(
    #             last_step_advantages
    #         ), f"number of groups {num_groups} doesn't match the number of trajectories as given by `is_last_step` {len(last_step_advantages)}. The `is_last_step` tensor is likely malformed"
    #         advantages = last_step_advantages[traj_ids]
    #         returns = last_step_returns[traj_ids]
    #     else:
    #         advantages, returns = ppo_utils.compute_advantages_and_returns(
    #             token_level_rewards=token_level_rewards,
    #             response_mask=data["response_mask"],
    #             index=data.metadata["uids"],
    #             adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
    #             config=self.cfg.trainer.algorithm,
    #             values=data["values"],
    #             gamma=self.cfg.trainer.algorithm.gamma,
    #             lambd=self.cfg.trainer.algorithm.lambd,
    #             grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
    #         )
    #     data["returns"] = returns
    #     data["advantages"] = advantages

    #     # remove padding while calculating metrics
    #     pad_size = data.metadata.get("pad_size", 0)
    #     num_samples = len(token_level_rewards)

    #     return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
    #     if self.cfg.trainer.step_wise_training:
    #         avg_rewards: float = return_sums[data["is_last_step"][: num_samples - pad_size]].mean().item()
    #     else:
    #         avg_rewards: float = return_sums.mean().item()

    #     avg_response_length = data.metadata["avg_response_length"]
    #     data = data.to("cpu")

    #     valid_advantages = torch.masked_select(
    #         data["advantages"][: num_samples - pad_size, ...], data["response_mask"][: num_samples - pad_size].bool()
    #     )
    #     avg_advantages: float = valid_advantages.mean().item()
    #     avg_advantages_abs: float = valid_advantages.abs().mean().item()

    #     if "metrics" not in data.metadata:
    #         data.metadata["metrics"] = {}
    #     data.metadata["metrics"].update(
    #         {
    #             "avg_final_rewards": avg_rewards,
    #             "avg_response_length": avg_response_length,
    #             "avg_advantages": avg_advantages,
    #             "avg_advantages_abs": avg_advantages_abs,
    #         }
    #     )

    #     logger.info(f"avg_final_rewards: {avg_rewards}, avg_response_length: {avg_response_length}")
    #     self.all_metrics.update(
    #         {
    #             "loss/avg_final_rewards": avg_rewards,
    #             "loss/avg_raw_advantages": avg_advantages,
    #             "loss/avg_raw_advantages_abs": avg_advantages_abs,
    #         }
    #     )
    #     return data