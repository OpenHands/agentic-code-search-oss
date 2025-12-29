import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray

import asyncio

# from src.tools import tool_exists
from src.generator.code_search_generator import CodeSearchGenerator
from src.async_trainer import CustomFullyAsyncRayPPOTrainer as FullyAsyncRayPPOTrainer
# from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer


class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        generator = CodeSearchGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
            step_wise=cfg.trainer.get("step_wise_training", False),
        )
        return generator

class AsyncCodeSearchPPOExp(CodeSearchPPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def run(self):
        trainer = self._setup_trainer()
        # Start the async training loop
        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    if cfg.get("run_async_trainer", False):
        print("Running async trainer")
        exp = AsyncCodeSearchPPOExp(cfg)
    else:
        print("Running sync trainer")
        exp = CodeSearchPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    # check cfg.generator.exp_config if it exists or not
    if hasattr(cfg.generator, "exp_config"):
        # Open yaml file and print its contents
        with open(cfg.generator.exp_config, "r") as f:
            exp_cfg = OmegaConf.load(f)

        with open_dict(cfg):
            cfg.generator.reward = exp_cfg.reward
            cfg.generator.tools = exp_cfg.tools
            # Parse prompts if they exist in the exp config
            if hasattr(exp_cfg, "prompts"):
                cfg.generator.prompts = exp_cfg.prompts
    else:
        with open_dict(cfg):
            cfg.generator.reward = [
                {"fn": "multilevel_localization_f1_reward"},
            ]
            cfg.generator.tools = [
                "terminal",
            ]

    # # Check if the tool exists in the registry
    # for tool in cfg.generator.tools:
    #     if not tool_exists(tool):
    #         raise ValueError(f"Tool {tool} does not exist in the registry")
    
    # Set default prompts if not specified
    if not hasattr(cfg.generator, "prompts"):
        with open_dict(cfg):
            cfg.generator.prompts = {
                "system_prompt": "templates/system_prompt.j2",
                "user_prompt": "templates/file_module_parallel_tools.j2"
            }
    
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
