#!/usr/bin/env python3
"""
Safely convert FSDP checkpoint to HuggingFace format.

Handles DTensor (Distributed Tensors) from FSDP2 checkpoints.
"""

import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def init_distributed():
    """Initialize PyTorch distributed for DTensor handling."""
    if not dist.is_initialized():
        # Set environment variables for single-process distributed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        # Initialize process group
        dist.init_process_group(
            backend='gloo',  # Use gloo for CPU
            init_method='env://',
            world_size=1,
            rank=0
        )
        print("✓ Initialized PyTorch distributed (required for DTensor)")


def convert_dtensor_to_tensor(obj):
    """Recursively convert DTensor to regular Tensor."""
    if isinstance(obj, dict):
        return {k: convert_dtensor_to_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dtensor_to_tensor(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_dtensor_to_tensor(item) for item in obj)
    elif hasattr(obj, '_local_tensor'):
        # This is a DTensor - extract local tensor
        return obj._local_tensor
    elif torch.is_tensor(obj):
        # Already a regular tensor
        return obj
    else:
        return obj


def load_fsdp_shards(policy_dir: Path):
    """Load and merge FSDP sharded checkpoints."""
    print("="*80)
    print("Step 1: Loading FSDP sharded weights")
    print("="*80)
    print(f"Loading from: {policy_dir}")
    print()
    
    # Initialize distributed for DTensor
    init_distributed()
    
    # Find all model shards
    model_shards = sorted(policy_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_shards:
        raise FileNotFoundError(f"No model shards found in {policy_dir}")
    
    print(f"Found {len(model_shards)} shards:")
    for shard in model_shards:
        size_mb = shard.stat().st_size / 1024 / 1024
        print(f"  {shard.name}: {size_mb:.2f} MB")
    print()
    
    # Load all shards
    print("Loading shards into memory...")
    all_state_dicts = []
    for shard_path in model_shards:
        print(f"  Loading {shard_path.name}...")
        shard = torch.load(shard_path, map_location="cpu")
        
        # Convert DTensors to regular tensors
        shard = convert_dtensor_to_tensor(shard)
        
        all_state_dicts.append(shard)
    
    print(f"✓ Loaded {len(all_state_dicts)} shards")
    print()
    
    # Merge shards
    print("Merging sharded state dicts...")
    merged_state = OrderedDict()
    
    # Get all unique keys
    all_keys = set()
    for sd in all_state_dicts:
        all_keys.update(sd.keys())
    
    print(f"Processing {len(all_keys)} unique parameter keys...")
    
    merge_count = 0
    replicate_count = 0
    single_count = 0
    
    for key in sorted(all_keys):
        # Collect this parameter from all shards
        param_shards = []
        for sd in all_state_dicts:
            if key in sd:
                param_shards.append(sd[key])
        
        if len(param_shards) == 1:
            # Only in one shard - use as is
            merged_state[key] = param_shards[0]
            single_count += 1
        elif len(param_shards) == len(all_state_dicts):
            # In all shards - check if replicated or sharded
            first_param = param_shards[0]
            
            # Check if all are identical (replicated)
            try:
                if all(torch.equal(first_param, p) for p in param_shards[1:]):
                    # Replicated - use first
                    merged_state[key] = first_param
                    replicate_count += 1
                    continue
            except Exception:
                # torch.equal failed, assume sharded
                pass
            
            # Sharded - need to concatenate
            # Try to infer the sharding dimension
            shapes = [p.shape for p in param_shards]
            
            # Find dimension that differs
            shard_dim = None
            if len(shapes[0]) > 0:
                for dim in range(len(shapes[0])):
                    if not all(s[dim] == shapes[0][dim] for s in shapes):
                        shard_dim = dim
                        break
            
            if shard_dim is not None:
                # Concatenate along shard dimension
                try:
                    merged_state[key] = torch.cat(param_shards, dim=shard_dim)
                    merge_count += 1
                except Exception as e:
                    # Fallback: try dim 0
                    try:
                        merged_state[key] = torch.cat(param_shards, dim=0)
                        merge_count += 1
                    except Exception:
                        print(f"  WARNING: Could not merge {key}, using first shard")
                        merged_state[key] = first_param
            else:
                # Couldn't determine sharding - try dim 0
                try:
                    merged_state[key] = torch.cat(param_shards, dim=0)
                    merge_count += 1
                except Exception as e:
                    print(f"  WARNING: Could not merge {key}, using first shard")
                    merged_state[key] = first_param
        else:
            # Parameter not in all shards - use first available
            merged_state[key] = param_shards[0]
            single_count += 1
    
    print(f"✓ Merged into {len(merged_state)} parameters")
    print(f"  - Replicated: {replicate_count}")
    print(f"  - Concatenated: {merge_count}")
    print(f"  - Single shard: {single_count}")
    print()
    
    return merged_state


def clean_state_dict_keys(state_dict):
    """Remove training wrapper prefixes but keep model. prefix."""
    print("="*80)
    print("Step 2: Cleaning state dict keys")
    print("="*80)
    
    cleaned = OrderedDict()
    prefix_counts = {}
    
    for key, value in state_dict.items():
        original_key = key
        
        # Remove training wrapper prefixes but keep 'model.'
        for prefix in [
            "_fsdp_wrapped_module.",
            "_forward_module.",
            "_checkpoint_wrapped_module.",
        ]:
            if key.startswith(prefix):
                key = key[len(prefix):]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        # Special handling for 'module.' prefix
        # Only remove if it's not followed by 'model.'
        if key.startswith("module.") and not key.startswith("module.model."):
            key = key[len("module."):]
            prefix_counts["module."] = prefix_counts.get("module.", 0) + 1
        
        cleaned[key] = value
    
    if prefix_counts:
        print("Removed prefixes:")
        for prefix, count in prefix_counts.items():
            print(f"  {prefix}: {count} occurrences")
    else:
        print("No prefixes to remove")
    
    print()
    print("Sample cleaned keys:")
    for i, key in enumerate(list(cleaned.keys())[:5]):
        print(f"  {key}")
    
    # Check if keys have 'model.' prefix
    has_model_prefix = any(k.startswith("model.") for k in cleaned.keys())
    print(f"\nKeys have 'model.' prefix: {has_model_prefix}")
    
    if not has_model_prefix:
        print("⚠️  WARNING: Keys don't have 'model.' prefix, adding it...")
        # Add model. prefix to all keys
        prefixed = OrderedDict()
        for key, value in cleaned.items():
            if not key.startswith("model."):
                prefixed[f"model.{key}"] = value
            else:
                prefixed[key] = value
        cleaned = prefixed
        
        print("Sample keys after adding prefix:")
        for i, key in enumerate(list(cleaned.keys())[:5]):
            print(f"  {key}")
    
    print()
    return cleaned


def detect_base_model(ckpt_dir: Path) -> str:
    """Try to detect base model from checkpoint."""
    # Check huggingface config
    hf_config = ckpt_dir / "huggingface" / "config.json"
    if hf_config.exists():
        import json
        with open(hf_config) as f:
            config = json.load(f)
        base_model = config.get("_name_or_path")
        if base_model:
            return base_model
    
    # Default fallback
    return "Qwen/Qwen2.5-Coder-7B-Instruct"


def save_as_huggingface(state_dict, output_dir: Path, base_model: str, tokenizer_dir: Path = None):
    """Save the state dict as a HuggingFace model."""
    print("="*80)
    print("Step 3: Creating HuggingFace model")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base model: {base_model}")
    print(f"Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print(f"✓ Base model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
    print()
    
    print("Loading fine-tuned weights...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  {len(missing_keys)} missing keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"    - {key}")
        else:
            for key in missing_keys[:5]:
                print(f"    - {key}")
            print(f"    ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"⚠️  {len(unexpected_keys)} unexpected keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"    - {key}")
        else:
            for key in unexpected_keys[:5]:
                print(f"    - {key}")
            print(f"    ... and {len(unexpected_keys) - 5} more")
    
    if not missing_keys and not unexpected_keys:
        print("✓ All keys matched perfectly!")
    
    print()
    print("="*80)
    print("Step 4: Saving HuggingFace model")
    print("="*80)
    
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    print("✓ Model saved")
    print()
    
    print("="*80)
    print("Step 5: Saving tokenizer")
    print("="*80)
    
    if tokenizer_dir and tokenizer_dir.exists():
        print(f"Loading tokenizer from: {tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        print(f"Loading tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    tokenizer.save_pretrained(output_dir)
    print("✓ Tokenizer saved")
    print()
    
    print("="*80)
    print("Conversion Complete!")
    print("="*80)
    
    # List output files
    output_files = sorted(output_dir.glob("*"))
    print(f"\nCreated {len(output_files)} files:")
    
    total_size = 0
    for f in output_files:
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  {f.name:45s} {size_mb:10.2f} MB")
    
    print(f"\n  {'Total size:':45s} {total_size:10.2f} MB")
    print()
    print(f"✓ Model saved to: {output_dir}")
    print()
    print("To use with vLLM:")
    print(f'  MODEL="{output_dir}"')
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (parent of global_step_XX)"
    )
    parser.add_argument(
        "--global-step",
        type=int,
        required=True,
        help="Global step to load (e.g., 44)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    try:
        ckpt_dir = Path(args.ckpt_dir)
        output_dir = Path(args.output_dir)
        global_step = args.global_step
        
        # Construct policy directory path
        policy_dir = ckpt_dir / f"global_step_{global_step}" / "policy"
        
        if not policy_dir.exists():
            raise FileNotFoundError(f"Policy directory not found: {policy_dir}")
        
        print("="*80)
        print("FSDP to HuggingFace Conversion")
        print("="*80)
        print(f"Checkpoint: {ckpt_dir}")
        print(f"Global step: {global_step}")
        print(f"Output: {output_dir}")
        print("="*80)
        print()
        
        # Load and merge FSDP shards
        state_dict = load_fsdp_shards(policy_dir)
        
        # Clean keys
        cleaned_state = clean_state_dict_keys(state_dict)
        
        # Determine base model
        base_model = args.base_model
        if base_model is None:
            print("Auto-detecting base model...")
            base_model = detect_base_model(policy_dir)
            print(f"Detected: {base_model}")
            print()
        
        # Check for tokenizer in checkpoint
        tokenizer_dir = policy_dir / "huggingface"
        if not tokenizer_dir.exists():
            tokenizer_dir = None
        
        # Save as HuggingFace
        save_as_huggingface(cleaned_state, output_dir, base_model, tokenizer_dir)
        
    finally:
        # Cleanup distributed
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()