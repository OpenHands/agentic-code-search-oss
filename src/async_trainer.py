import numpy as np
import torch
import asyncio

from loguru import logger
from typing import List
from skyrl_train.utils import Timer
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
            additional_result[key] = np.mean([generator_output[key] for generator_output in generator_outputs]).item()
        except Exception as e:
            logger.error(f"Error in aggregating key {key}: {e}", exc_info=True)

    # Re-aggregate rollout metrics
    rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
    result["rollout_metrics"] = {**rollout_metrics, **additional_result}

    # Validate the generator output using the number of prompts
    from skyrl_train.utils.trainer_utils import validate_generator_output

    num_prompts = len(result["prompt_token_ids"])
    validate_generator_output(num_prompts, result)

    return result


class CustomFullyAsyncRayPPOTrainer(FullyAsyncRayPPOTrainer):
    """
    Custom async trainer for batched training.
    
    Key changes:
    1. Prevents automatic epoch looping (for batched training control)
    2. Fixes TrajectoryID serialization for step-wise training
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag to control whether to loop epochs or stop after one
        self._single_epoch_mode = False

    def enable_single_epoch_mode(self):
        """Enable single-epoch mode for batched training."""
        self._single_epoch_mode = True
        logger.info("[CustomAsyncTrainer] Single-epoch mode enabled")

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

        # ✅ FIX: Convert TrajectoryID objects to strings for use as dict keys
        trajectory_ids = generator_output["trajectory_ids"]
        
        # Convert TrajectoryID to string if needed
        if trajectory_ids and hasattr(trajectory_ids[0], '__dict__'):
            # TrajectoryID objects - convert to strings
            uids_hashable = [str(tid) for tid in trajectory_ids]
        else:
            # Already strings or ints
            uids_hashable = trajectory_ids
        
        step_wise_training = self.cfg.trainer.step_wise_training
        self.cfg.trainer.step_wise_training = False
        
        generator_output = self.postprocess_generator_output(generator_output, uids_hashable)

        # print example just for debugging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Example generated: {vis}")

        training_input = self.convert_to_training_input(generator_output, uids_hashable)
        self.cfg.trainer.step_wise_training = step_wise_training
        return training_input

    async def train(self):
        """
        Override train() to support single-epoch mode for batched training.
        
        When single_epoch_mode is enabled, this will:
        1. Run exactly one epoch (even if cfg.trainer.epochs > 1)
        2. NOT reset the dataloader at epoch end
        3. NOT validate staleness manager at epoch end
        4. Return cleanly after one epoch completes
        """
        self.global_step = 0

        # Load checkpoint state if resumption is enabled
        if self.resume_mode != trainer_utils.ResumeMode.NONE:
            with Timer("load_checkpoints"):
                self.global_step, _, loaded_consumed_data_uids_set = self.load_checkpoints()
                logger.info(f"Resumed training from global_step {self.global_step}")
                if self.global_step > 0:
                    self.async_train_dataloader.load_state_from_checkpoint(loaded_consumed_data_uids_set)
                    self._staleness_manager.load_state_from_checkpoint(self.global_step + 1)
                    expected_consumed_in_epoch = self.mini_batch_size * (self.global_step % self.num_steps_per_epoch)
                    assert len(loaded_consumed_data_uids_set) == expected_consumed_in_epoch, (
                        "Unexpected number of consumed data UIDs. Got: "
                        f"{len(loaded_consumed_data_uids_set)} != {expected_consumed_in_epoch}"
                    )

        # Initialize weight sync state
        with Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        # sync weights to inference engines
        with Timer("sync_weights_to_inference_engines"):
            await self.async_sync_policy_weights_to_inference_engines()

        # Eval before training
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self.tracker.log(eval_metrics, step=self.global_step)

        # main training loop
        from tqdm import tqdm
        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Step Progress")
        start_epoch = self.global_step // self.num_steps_per_epoch
        self.global_step += 1  # start training at global_step 1
        
        # ✅ KEY CHANGE: Limit epochs to 1 in single-epoch mode
        end_epoch = start_epoch + 1 if self._single_epoch_mode else self.cfg.trainer.epochs
        
        for epoch in range(start_epoch, end_epoch):
            logger.info(f"[CustomAsyncTrainer] Starting epoch {epoch} (single_epoch_mode={self._single_epoch_mode})")
            
            # 0. Per-epoch prologue
            generation_output_group_buffer = asyncio.Queue[GeneratedOutputGroup](
                maxsize=self.mini_batch_size * (self.max_staleness_steps + 1)
            )

            generator_tasks = [
                asyncio.create_task(self._run_generate_for_a_group_loop(generation_output_group_buffer))
                for _ in range(self.num_parallel_generation_workers)
            ]

            for _ in range(self.global_step, (1 + epoch) * self.num_steps_per_epoch + 1):
                with Timer("step", self.all_timings):
                    # 1. Wait until we have enough groups buffered
                    cur_generation_group_mini_batch: List[GeneratedOutputGroup] = []
                    with Timer("wait_for_generation_buffer", self.all_timings):
                        buffer_pbar = tqdm(
                            total=self.mini_batch_size,
                            initial=0,
                            desc="Generation Buffer Progress",
                            position=1,
                        )
                        while len(cur_generation_group_mini_batch) < self.mini_batch_size:
                            cur_generation_group_mini_batch.append(await generation_output_group_buffer.get())
                            buffer_pbar.update(1)
                            buffer_pbar.set_postfix({"buffer qsize": generation_output_group_buffer.qsize()})
                        buffer_pbar.close()

                    # 2. Post-process and convert to training format
                    with Timer("convert_to_training_input", self.all_timings):
                        training_input = await asyncio.to_thread(
                            self.convert_generation_group_mini_batch_to_training_input, cur_generation_group_mini_batch
                        )

                    # 3. Run training
                    with Timer("run_training", self.all_timings):
                        status = await self._run_training(training_input)
                        await self.async_train_dataloader.mark_consumed_uids(
                            [g.uid for g in cur_generation_group_mini_batch]
                        )

                    # 4. Sync weights
                    with Timer("sync_weights", self.all_timings):
                        await self.inference_engine_client.pause_generation()
                        await self.async_sync_policy_weights_to_inference_engines()
                        await self.inference_engine_client.resume_generation()

                # 5. Logging
                logger.info(status)
                self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
                self.tracker.log(self.all_metrics, step=self.global_step)
                self.all_metrics = {}
                pbar.update(1)

                # 6. Eval and checkpointing
                if self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                ):
                    with Timer("eval", self.all_timings):
                        eval_metrics = await self.eval()
                        self.all_metrics.update(eval_metrics)
                if self.cfg.trainer.ckpt_interval > 0 and self.global_step % self.cfg.trainer.ckpt_interval == 0:
                    with Timer("save_checkpoints", self.all_timings):
                        await asyncio.to_thread(self.save_checkpoints)
                if self.cfg.trainer.hf_save_interval > 0 and self.global_step % self.cfg.trainer.hf_save_interval == 0:
                    with Timer("save_hf_model", self.all_timings):
                        await asyncio.to_thread(self.save_models)
                self.tracker.log({"timing/" + k: v for k, v in self.all_timings.items()}, step=self.global_step)
                self.all_timings = {}
                self.global_step += 1

                # 7. Notify capacity change
                await self._staleness_manager.notify_capacity_change(self.global_step)
                
                # ✅ SKIP UID validation in single-epoch mode (will be handled between batches)
                if not self._single_epoch_mode:
                    expected_consumed_in_epoch = self.mini_batch_size * ((self.global_step - 1) % self.num_steps_per_epoch)
                    actual_consumed_in_epoch = len(self.async_train_dataloader.get_consumed_uids_list())
                    assert actual_consumed_in_epoch == expected_consumed_in_epoch, (
                        "Unexpected number of consumed data UIDs. Got: "
                        f"{actual_consumed_in_epoch} != {expected_consumed_in_epoch}"
                    )

            # 8. Per-epoch epilogue
            if self.cfg.trainer.update_ref_every_epoch and self.ref_model is not None:
                with Timer("update_ref_with_policy", self.all_timings):
                    await asyncio.to_thread(self.update_ref_with_policy)

            # Cancel generator tasks
            for t in generator_tasks:
                t.cancel()
            try:
                await asyncio.gather(*generator_tasks, return_exceptions=True)
            except Exception:
                pass

            # ✅ KEY CHANGE: Skip reset/validation in single-epoch mode
            if not self._single_epoch_mode:
                assert all(t.done() for t in generator_tasks), "Generator tasks must be done"
                assert generation_output_group_buffer.qsize() == 0, "Generation buffer should be empty"
                await self.async_train_dataloader.reset_at_epoch_end()
                await self._staleness_manager.validate_state_at_epoch_end(self.global_step)
            else:
                logger.info(f"[CustomAsyncTrainer] Epoch {epoch} complete in single-epoch mode, skipping reset/validation")
                # Just ensure tasks are done
                assert all(t.done() for t in generator_tasks), "Generator tasks must be done"

        pbar.close()
        
        # Final checkpointing
        if self.cfg.trainer.ckpt_interval > 0:
            with Timer("save_checkpoints", self.all_timings):
                await asyncio.to_thread(self.save_checkpoints)
                logger.info("Saved final checkpoint.")
        if self.cfg.trainer.hf_save_interval > 0:
            with Timer("save_hf_model", self.all_timings):
                await asyncio.to_thread(self.save_models)
                logger.info("Saved final model.")
        
        logger.info(f"[CustomAsyncTrainer] Training complete (single_epoch_mode={self._single_epoch_mode})")