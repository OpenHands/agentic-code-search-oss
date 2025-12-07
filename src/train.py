import hydra
from omegaconf import DictConfig, OmegaConf
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.utils import initialize_ray
import ray

import asyncio

from src.generator.code_search_generator import CodeSearchGenerator
from src.async_trainer import AsyncRayPPOTrainer


class CodeSearchPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        generator = CodeSearchGenerator(
            generator_cfg=cfg.generator,
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

    initialize_ray(cfg)

    # Initialize embedding service if semantic search enabled
    if cfg.generator.get("use_semantic_search", False):
        print("\n" + "="*80)
        print("Initializing Semantic Search")
        print("="*80)

        # Check if indices are pre-computed
        from pathlib import Path
        cache_dir = Path.home() / ".cache" / "swebench_indices"
        if not cache_dir.exists() or len(list(cache_dir.iterdir())) == 0:
            print("⚠️  WARNING: No pre-computed indices found!")
            print("   Run pre-indexing first: python preindex_swebench.py")
            print("   Training will fail if semantic search is used without indices.")
        else:
            num_indices = len(list(cache_dir.iterdir()))
            print(f"✓ Found {num_indices} pre-computed indices in {cache_dir}")

        # Initialize shared embedding service (Ray actor)
        from src.services.embedding_service import get_embedding_service
        device = cfg.get("embedding_device", "cpu")
        max_indices = cfg.get("max_indices", 50)  # LRU cache size
        max_cache_size_gb = cfg.get("max_cache_size_gb", None)  # Disk space limit

        print(f"\nInitializing embedding service on {device}...")
        print(f"  - LRU cache: max {max_indices} indices")
        if max_cache_size_gb:
            print(f"  - Disk limit: {max_cache_size_gb:.1f} GB")

        embedding_service = get_embedding_service(
            device=device,
            max_indices=max_indices,
            max_cache_size_gb=max_cache_size_gb,
        )

        # Wait for initialization to complete
        stats = ray.get(embedding_service.get_cache_stats.remote())
        print(f"✓ Embedding service ready!")
        print(f"  - Device: {stats['device']}")
        print(f"  - Embedding model: {stats['embedding_model']}")
        print(f"  - Cache: {stats['loaded_indices']}/{stats['max_indices']} indices loaded")
        print(f"  - Disk usage: {stats['total_cache_size_gb']:.2f} GB" + (f" / {stats['max_cache_size_gb']:.1f} GB" if stats['max_cache_size_gb'] else ""))
        print(f"  - Reranker model: {stats['reranker_model']}")
        print("="*80 + "\n")

    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()