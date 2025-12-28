import numpy as np
import asyncio
from loguru import logger
from typing import List

from skyrl_train.utils import Timer
from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.generators.base import GeneratorOutput
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer, GeneratedOutputGroup


def patched_concatenate_generator_outputs(generator_outputs: List[GeneratorOutput]) -> GeneratorOutput:
    """Concatenate generator outputs with proper metric aggregation."""
    assert len(generator_outputs) > 0
    has_rollout_logprobs = [output.get("rollout_logprobs") is not None for output in generator_outputs]
    if any(has_rollout_logprobs) and not all(has_rollout_logprobs):
        raise ValueError("Mixed rollout_logprobs state")
    
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

    # Aggregate additional metrics
    additional_keys = [
        key for key in generator_outputs[0] 
        if key not in result and isinstance(generator_outputs[0][key], (int, float))
    ]
    additional_result = {}
    for key in additional_keys:
        try:
            additional_result[key] = np.mean([g[key] for g in generator_outputs]).item()
        except Exception as e:
            logger.error(f"Error aggregating {key}: {e}")

    # Re-aggregate rollout metrics
    rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
    result["rollout_metrics"] = {**rollout_metrics, **additional_result}

    # Validate
    from skyrl_train.utils.trainer_utils import validate_generator_output
    validate_generator_output(len(result["prompt_token_ids"]), result)

    return result


class CustomFullyAsyncRayPPOTrainer(FullyAsyncRayPPOTrainer):
    """
    Custom async trainer for batched training.
    
    Changes:
    1. Fixes TrajectoryID serialization
    2. Properly handles epoch logic for batched mode
    3. Each batch is epoch 0, global_step continues across batches
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batched_mode = False

    def enable_batched_mode(self):
        """Enable batched mode - each batch is independent epoch."""
        self._batched_mode = True
        logger.info("[CustomAsyncTrainer] Batched mode enabled")

    def convert_generation_group_mini_batch_to_training_input(
        self, cur_generation_group_mini_batch: List[GeneratedOutputGroup]
    ) -> TrainingInputBatch:
        """Convert generation groups to training input - with TrajectoryID fix."""
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

            if cur_staleness > self.max_staleness_steps:
                logger.warning(f"Staleness violation: {cur_staleness} > {self.max_staleness_steps}")
                staleness_violation_count += 1

        generator_output = patched_concatenate_generator_outputs(generator_outputs)
        assert generator_output["rollout_metrics"] is not None
        self.all_metrics.update(generator_output["rollout_metrics"])

        # Log staleness stats
        self.all_metrics.update({
            "async/staleness_mean": sum(stalenesses) / len(stalenesses),
            "async/staleness_max": max(stalenesses),
            "async/staleness_min": min(stalenesses),
            "async/staleness_ratio": sum(1 for s in stalenesses if s > 0) / len(stalenesses),
            "async/staleness_violation_count": staleness_violation_count,
        })

        # ✅ FIX: Convert TrajectoryID objects to strings
        trajectory_ids = generator_output["trajectory_ids"]
        if trajectory_ids and hasattr(trajectory_ids[0], '__dict__'):
            uids_hashable = [str(tid) for tid in trajectory_ids]
        else:
            uids_hashable = trajectory_ids
        
        # Temporarily disable step_wise_training for postprocessing
        step_wise_training = self.cfg.trainer.step_wise_training
        self.cfg.trainer.step_wise_training = False
        
        generator_output = self.postprocess_generator_output(generator_output, uids_hashable)
        
        # Example logging
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Example generated: {vis}")

        training_input = self.convert_to_training_input(generator_output, uids_hashable)
        
        # Restore step_wise_training
        self.cfg.trainer.step_wise_training = step_wise_training
        
        return training_input

    async def train(self):
        """
        Override train() to handle batched mode.
        
        In batched mode:
        - Each batch is epoch 0 (fresh start)
        - global_step continues across batches (from checkpoint)
        - No dataset state loaded (UIDs, staleness)
        """
        # ✅ FIX: In batched mode, we need to track initial global_step separately
        initial_global_step = 0

        # Load checkpoint state if resumption is enabled
        if self.resume_mode != trainer_utils.ResumeMode.NONE:
            with Timer("load_checkpoints"):
                loaded_step, _, loaded_consumed_data_uids_set = self.load_checkpoints()
                
                if self._batched_mode:
                    # ✅ BATCHED MODE: Only use model weights + global_step
                    logger.info(f"[CustomAsyncTrainer] Batched mode - loaded checkpoint at global_step={loaded_step}")
                    logger.info(f"[CustomAsyncTrainer] Continuing training from this step")
                    logger.info(f"[CustomAsyncTrainer] NOT loading dataset state (new batch, new dataset)")
                    initial_global_step = loaded_step
                    # Don't load consumed UIDs or staleness state
                    
                else:
                    # ✅ NORMAL MODE: Full checkpoint restore
                    logger.info(f"[CustomAsyncTrainer] Normal mode - loading full checkpoint state")
                    initial_global_step = loaded_step
                    
                    if loaded_step > 0:
                        self.async_train_dataloader.load_state_from_checkpoint(loaded_consumed_data_uids_set)
                        self._staleness_manager.load_state_from_checkpoint(loaded_step + 1)
                        
                        expected_consumed = self.mini_batch_size * (loaded_step % self.num_steps_per_epoch)
                        assert len(loaded_consumed_data_uids_set) == expected_consumed, (
                            f"UID mismatch: {len(loaded_consumed_data_uids_set)} != {expected_consumed}"
                        )

        # ✅ Set global_step to starting point
        self.global_step = initial_global_step

        # Initialize weight sync
        with Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        # Sync weights to inference engines
        with Timer("sync_weights_to_inference_engines"):
            await self.async_sync_policy_weights_to_inference_engines()

        # Eval before training
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self.tracker.log(eval_metrics, step=self.global_step)

        # ✅ FIX: Batched mode always uses epoch 0, normal mode uses calculated epoch
        from tqdm import tqdm
        
        if self._batched_mode:
            # Each batch is epoch 0
            start_epoch = 0
            end_epoch = 1
            # Calculate how many steps to run in THIS batch
            steps_in_batch = self.num_steps_per_epoch
            end_step = self.global_step + steps_in_batch
            
            logger.info(f"[CustomAsyncTrainer] Batched mode: epoch 0")
            logger.info(f"[CustomAsyncTrainer] Will train from step {self.global_step} to {end_step}")
            logger.info(f"[CustomAsyncTrainer] Steps in this batch: {steps_in_batch}")
        else:
            # Normal mode: calculate epoch from global_step
            start_epoch = self.global_step // self.num_steps_per_epoch
            end_epoch = self.cfg.trainer.epochs
            end_step = end_epoch * self.num_steps_per_epoch
            
            logger.info(f"[CustomAsyncTrainer] Normal mode: epochs {start_epoch} to {end_epoch}")
        
        # Progress bar
        pbar = tqdm(
            total=end_step - self.global_step,
            initial=0,
            desc="Training Step Progress"
        )
        
        # ✅ Increment global_step to start training
        self.global_step += 1
        
        for epoch in range(start_epoch, end_epoch):
            logger.info(f"[CustomAsyncTrainer] Starting epoch {epoch}")
            
            # Per-epoch prologue
            generation_output_group_buffer = asyncio.Queue[GeneratedOutputGroup](
                maxsize=self.mini_batch_size * (self.max_staleness_steps + 1)
            )

            generator_tasks = [
                asyncio.create_task(self._run_generate_for_a_group_loop(generation_output_group_buffer))
                for _ in range(self.num_parallel_generation_workers)
            ]

            # ✅ FIX: Loop until we've done the required steps for THIS batch
            while self.global_step <= end_step:
                with Timer("step", self.all_timings):
                    # 1. Wait for generation buffer
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

                    # 2. Convert to training input
                    with Timer("convert_to_training_input", self.all_timings):
                        training_input = await asyncio.to_thread(
                            self.convert_generation_group_mini_batch_to_training_input,
                            cur_generation_group_mini_batch
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
                
                # 7. Notify capacity change
                await self._staleness_manager.notify_capacity_change(self.global_step)
                
                # ✅ Increment for next step
                self.global_step += 1

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

            # Validation
            assert all(t.done() for t in generator_tasks), "Generator tasks must be done"
            
            if not self._batched_mode:
                assert generation_output_group_buffer.qsize() == 0, "Buffer should be empty"
                await self.async_train_dataloader.reset_at_epoch_end()
                await self._staleness_manager.validate_state_at_epoch_end(self.global_step - 1)

        pbar.close()
        
        # Final checkpointing
        if self.cfg.trainer.ckpt_interval > 0:
            with Timer("save_checkpoints", self.all_timings):
                await asyncio.to_thread(self.save_checkpoints)
                logger.info("[CustomAsyncTrainer] Saved final checkpoint")
                
        if self.cfg.trainer.hf_save_interval > 0:
            with Timer("save_hf_model", self.all_timings):
                await asyncio.to_thread(self.save_models)
                logger.info("[CustomAsyncTrainer] Saved final model")
        
        logger.info(f"[CustomAsyncTrainer] Training complete")


# Import for type checking
from skyrl_train.utils import trainer_utils