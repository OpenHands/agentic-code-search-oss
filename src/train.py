import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray

import asyncio

from src.generator.code_search_generator import CodeSearchGenerator
from src.async_trainer import CustomFullyAsyncRayPPOTrainer as FullyAsyncRayPPOTrainer
# from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer


class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        semantic_search_cfg = cfg.get('semantic_search', OmegaConf.create({
            'enabled': True,
            'embedding_model': 'jinaai/jina-code-embeddings-0.5b',
            'reranker_model': None,
            'device': 'cuda',
            'max_indices': 50
        }))
        generator = CodeSearchGenerator(
            generator_cfg=cfg.generator,
            semantic_search_cfg=semantic_search_cfg,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
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

    # Check cfg.generator.reward if it exists or not
    if hasattr(cfg.generator, "reward"):
        # Open yaml file and print its contents
        with open(cfg.generator.reward, "r") as f:
            reward_cfg = OmegaConf.load(f)
        cfg.generator.reward = reward_cfg.reward
    else:
        with open_dict(cfg):
            cfg.generator.reward = [
                {"fn": "multiturn_reward"},
                {"fn": "file_localization_f1_reward"},
            ]

    semantic_search_cfg = cfg.get('semantic_search', OmegaConf.create({'enabled': False}))
    
    if semantic_search_cfg.enabled:
        print("Initializing Semantic Search")

        # Check if indices are pre-computed
        from pathlib import Path
        cache_dir = Path("/data/user_data/sanidhyv/.cache") / "swebench_indices"
        if not cache_dir.exists() or len(list(cache_dir.iterdir())) == 0:
            print("⚠️  WARNING: No pre-computed indices found!")
            print("   Run pre-indexing first: python preindex_swebench.py")
        else:
            num_indices = len(list(cache_dir.iterdir()))
            print(f"Found {num_indices} pre-computed indices in {cache_dir}")

        # Initialize shared embedding service
        from src.services.embedding_service import get_embedding_service
        
        device = semantic_search_cfg.device
        max_indices = semantic_search_cfg.max_indices

        print(f"\nInitializing embedding service:")
        print(f"  - Device: {device}")
        print(f"  - Embedding model: {semantic_search_cfg.embedding_model}")
        print(f"  - Reranker model: {semantic_search_cfg.reranker_model}")
        print(f"  - LRU cache: max {max_indices} indices")
       
        embedding_service = get_embedding_service(
            device=device,
            max_indices=max_indices,
        )

        stats = ray.get(embedding_service.get_cache_stats.remote())
        print(f"Embedding service ready!")
        print(f"  - Cache: {stats['loaded_indices']}/{stats['max_indices']} indices loaded")
    if ray.is_initialized():
        ray.shutdown()
    from skyrl_train.utils import prepare_runtime_environment
    from skyrl_train.utils.ppo_utils import sync_registries
    import os

    # Prepare environment variables
    env_vars = prepare_runtime_environment(cfg)

    # Initialize Ray with packages installed via pip
    ray.init(
        runtime_env={
            "env_vars": env_vars
        }
    )

    # Sync registries
    sync_registries()
    ray.get(skyrl_entrypoint.remote(cfg))

if __name__ == "__main__":
    main()
