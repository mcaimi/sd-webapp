#!/usr/bin/env python
"""
Merge Operations

Stable Diffusion Checkpoint Merging machinery

- SD1.5
- SDXL
"""

try:
    import torch
    import json
    import time
    from libs.globals.vars import MergeMethod
    from libs.stablediffusion.funcs import merge_tensors
    from libs.shared.exceptions import CheckpointLoadError, MergeError
    from pathlib import PosixPath
    from safetensors.torch import load_file, save_file
    from typing import List, Callable, Optional, Union, Tuple
    from dataclasses import dataclass
except ImportError as e:
    print(f"Caught Exception: {e}")
    raise e


@dataclass
class MergeConfig:
    """Configuration for merge operations"""

    method: MergeMethod = MergeMethod.LINEAR
    alpha: float = 0.5
    device: str = "cpu"
    preserve_metadata: bool = True
    chunk_size: Optional[int] = None  # For memory-efficient processing
    progress_callback: Optional[Callable[[float], None]] = None


# load checkpoint weight and return a dict with metadata
def load_checkpoint_dict(
    checkpoint_file: Union[str, PosixPath], device: str = "cpu"
) -> Tuple[dict, dict]:
    """Load checkpoint and return weights dict and metadata"""
    if isinstance(checkpoint_file, str):
        checkpoint_file = PosixPath(checkpoint_file)

    if not checkpoint_file.exists():
        raise CheckpointLoadError(f"Checkpoint file not found: {checkpoint_file}")

    if not checkpoint_file.name.endswith(".safetensors"):
        raise CheckpointLoadError(
            f"Only safetensors format supported, got: {checkpoint_file.suffix}"
        )

    try:
        print(f"Loading weights from checkpoint {checkpoint_file}...")
        weights = load_file(checkpoint_file, device=device)

        # Try to load metadata
        metadata = {}
        metadata_file = checkpoint_file.with_suffix(".json")
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        return weights, metadata
    except Exception as e:
        raise CheckpointLoadError(
            f"Failed to load checkpoint {checkpoint_file}: {str(e)}"
        )


#  checkpoint structure validation (keys/layers)
def validate_checkpoints(base_ckpt: dict, *other_ckpts: dict) -> Tuple[bool, List[str]]:
    """Validate that checkpoints are compatible for merging"""
    errors = []
    base_keys = set(base_ckpt.keys())

    for i, ckpt in enumerate(other_ckpts):
        ckpt_keys = set(ckpt.keys())
        missing_in_base = ckpt_keys - base_keys
        missing_in_ckpt = base_keys - ckpt_keys

        if missing_in_base:
            errors.append(
                f"Checkpoint {i + 1} has extra keys: {list(missing_in_base)[:5]}..."
            )
        if missing_in_ckpt:
            errors.append(
                f"Checkpoint {i + 1} missing keys: {list(missing_in_ckpt)[:5]}..."
            )

        # Check tensor shapes
        for key in base_keys.intersection(ckpt_keys):
            if base_ckpt[key].shape != ckpt[key].shape:
                errors.append(
                    f"Shape mismatch for key '{key}': {base_ckpt[key].shape} vs {ckpt[key].shape}"
                )

    return len(errors) == 0, errors


# Enhanced merge function with multiple methods
def merge_checkpoints(
    base_ckpt: dict,
    target_ckpt: dict,
    config: MergeConfig,
    base_reference: Optional[dict] = None,
) -> dict:
    """Advanced checkpoint merging with multiple algorithms"""
    # Validate inputs
    if isinstance(config.alpha, str):
        config.alpha = float(config.alpha)

    is_valid, errors = validate_checkpoints(base_ckpt, target_ckpt)
    if not is_valid:
        raise MergeError(f"Checkpoint validation failed: {errors}")

    merged_dict = {}
    total_keys = len(base_ckpt.keys())

    for i, key in enumerate(base_ckpt.keys()):
        # Progress callback
        if config.progress_callback:
            config.progress_callback(i / total_keys)

        # Prepare tensors
        tensor_a = base_ckpt[key]
        tensor_b = target_ckpt[key]
        layer_alpha = config.alpha

        # Additional arguments for some merge methods
        merge_kwargs = {}
        if (
            config.method in [MergeMethod.ADDITIVE, MergeMethod.SUBTRACT]
        ) and base_reference:
            merge_kwargs["base_tensor"] = base_reference[key]

        # Perform merge
        merged_dict[key] = merge_tensors(
            tensor_a, tensor_b, config.method, layer_alpha, **merge_kwargs
        )

    # finish processing
    if config.progress_callback:
        config.progress_callback(1.0)

    # return merged checkpoint dictionary
    return merged_dict


# checkpoint saving with metadata support
def save_checkpoint(
    final_ckpt: dict,
    filename: Union[str, PosixPath],
    metadata: Optional[dict] = None,
    backup_existing: bool = True,
) -> bool:
    """Save checkpoint with optional metadata and backup"""
    if isinstance(filename, str):
        filename = PosixPath(filename)

    try:
        # Create backup if file exists
        if backup_existing and filename.exists():
            backup_path = filename.with_suffix(
                f".backup_{int(time.time())}.safetensors"
            )
            filename.rename(backup_path)
            print(f"Created backup: {backup_path}")

        # Save weights
        print(f"Saving checkpoint to {filename}...")
        save_file(final_ckpt, filename)

        # Save metadata if provided
        if metadata:
            metadata_file = filename.with_suffix(".json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_file}")

        return True
    except Exception as e:
        print(f"Failed to save checkpoint: {str(e)}")
        return False


# Batch merge functionality
def batch_merge_checkpoints(
    base_checkpoint: Union[str, PosixPath],
    checkpoint_list: List[Union[str, PosixPath]],
    output_dir: Union[str, PosixPath],
    config: MergeConfig,
) -> List[str]:
    """Merge base checkpoint with multiple other checkpoints"""
    if isinstance(output_dir, str):
        output_dir = PosixPath(output_dir)

    output_dir.mkdir(exist_ok=True)

    # Load base checkpoint
    base_weights, base_metadata = load_checkpoint_dict(base_checkpoint, config.device)

    results = []

    for i, target_checkpoint in enumerate(checkpoint_list):
        try:
            print(f"\nMerging {i + 1}/{len(checkpoint_list)}: {target_checkpoint}")

            # Load target checkpoint
            target_weights, target_metadata = load_checkpoint_dict(
                target_checkpoint, config.device
            )

            # Perform merge
            merged_weights = merge_checkpoints(base_weights, target_weights, config)

            # Create output filename
            target_path = PosixPath(target_checkpoint)
            output_filename = output_dir / f"merged_{target_path.stem}.safetensors"

            # Merge metadata
            merged_metadata = {
                "base_model": str(base_checkpoint),
                "target_model": str(target_checkpoint),
                "merge_method": config.method.value,
                "merge_alpha": config.alpha,
                "timestamp": time.time(),
                "base_metadata": base_metadata,
                "target_metadata": target_metadata,
            }

            # Save result
            if save_checkpoint(merged_weights, output_filename, merged_metadata):
                results.append(str(output_filename))
                print(f"✓ Saved: {output_filename}")
            else:
                print(f"✗ Failed to save: {output_filename}")

        except Exception as e:
            print(f"✗ Error merging {target_checkpoint}: {str(e)}")

    return results


# Memory-efficient merge for large models
def merge_checkpoints_chunked(
    base_ckpt: dict, target_ckpt: dict, config: MergeConfig, chunk_size: int = 100
) -> dict:
    """Memory-efficient merge processing in chunks"""
    keys = list(base_ckpt.keys())
    merged_dict = {}

    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i : i + chunk_size]
        print(
            f"Processing chunk {i // chunk_size + 1}/{(len(keys) + chunk_size - 1) // chunk_size}"
        )

        # Create chunk dictionaries
        base_chunk = {k: base_ckpt[k] for k in chunk_keys}
        target_chunk = {k: target_ckpt[k] for k in chunk_keys}

        # Merge chunk
        chunk_config = MergeConfig(
            method=config.method,
            alpha=config.alpha,
            device=config.device,
        )

        merged_chunk = merge_checkpoints(base_chunk, target_chunk, chunk_config)
        merged_dict.update(merged_chunk)

        # Clear memory
        del base_chunk, target_chunk, merged_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return merged_dict

