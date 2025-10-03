#!/usr/bin/env python
"""
Advanced Stable Diffusion XL Model Merging Library

Supports multiple merge algorithms:
    - Linear interpolation
    - Spherical linear interpolation (SLERP)
    - Additive merging
    - Subtractive merging

Features:
    - Simple 2-model merging
    - Compatibility checks
    - Batch merging of multiple models
    - Multi-step merging
"""

# load libs
try:
    import torch
    import time
    from libs.shared.utils import get_gpu
    from libs.globals.vars import MergeMethod
    from pathlib import PosixPath
    from typing import Dict, List, Optional, Union
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# import mergeops
try:
    from libs.stablediffusion.mergeops import (
        MergeConfig,
        load_checkpoint_dict,
        save_checkpoint,
        merge_checkpoints,
    )
except ImportError as e:
    print(f"Caught exception: {e}")
    raise e


# Integration with SDXL Pipeline
class SDXLMergePipeline:
    """Advanced merging pipeline integrated with SDXLPipelineGenerator"""

    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else get_gpu()
        self.loaded_checkpoints = {}  # Cache for loaded checkpoints

    def load_and_cache_checkpoint(
        self, checkpoint_path: Union[str, PosixPath], cache_key: Optional[str] = None
    ) -> str:
        """Load and cache a checkpoint for future merging operations"""
        if isinstance(checkpoint_path, str):
            checkpoint_path = PosixPath(checkpoint_path)

        cache_key = cache_key or checkpoint_path.stem

        if cache_key not in self.loaded_checkpoints:
            weights, metadata = load_checkpoint_dict(checkpoint_path, self.device)
            self.loaded_checkpoints[cache_key] = {
                "weights": weights,
                "metadata": metadata,
                "path": checkpoint_path,
            }
            print(f"Cached checkpoint: {cache_key}")

        return cache_key

    def create_merge_recipe(self, base_model: str, merge_steps: List[Dict]) -> Dict:
        """Create a merge recipe for complex multi-step merging"""
        return {
            "base_model": base_model,
            "steps": merge_steps,
            "created_at": time.time(),
        }

    def execute_merge_recipe(
        self, recipe: dict, output_path: Union[str, PosixPath]
    ) -> bool:
        """Execute a complex merge recipe with multiple steps"""
        try:
            # Load base model
            base_key = self.load_and_cache_checkpoint(recipe["base_model"])
            current_weights = self.loaded_checkpoints[base_key]["weights"].copy()

            recipe_metadata = {
                "recipe": recipe,
                "execution_timestamp": time.time(),
                "steps_executed": [],
            }

            # Execute each merge step
            for i, step in enumerate(recipe["steps"]):
                print(f"Executing merge step {i + 1}/{len(recipe['steps'])}")

                target_key = self.load_and_cache_checkpoint(step["target_model"])
                target_weights = self.loaded_checkpoints[target_key]["weights"]

                # Create merge config
                config = MergeConfig(
                    method=MergeMethod(step.get("method", "linear")),
                    alpha=step.get("alpha", 0.5),
                    device=self.device,
                    progress_callback=lambda p: print(f"  Progress: {p * 100:.1f}%")
                    if step.get("show_progress", False)
                    else None,
                )

                # Perform merge
                base_reference = None
                if config.method in [MergeMethod.ADDITIVE, MergeMethod.SUBTRACT]:
                    base_reference = self.loaded_checkpoints[base_key]["weights"]

                current_weights = merge_checkpoints(
                    current_weights, target_weights, config, base_reference
                )

                # Record step execution
                recipe_metadata["steps_executed"].append(
                    {
                        "step_index": i,
                        "target_model": str(step["target_model"]),
                        "method": step.get("method", "linear"),
                        "alpha": step.get("alpha", 0.5),
                        "completed_at": time.time(),
                    }
                )

            # Save final result
            return save_checkpoint(current_weights, output_path, recipe_metadata)

        except Exception as e:
            print(f"Failed to execute merge recipe: {str(e)}")
            return False

    def merge_for_pipeline_generator(
        self,
        base_model: Union[str, PosixPath],
        target_model: Union[str, PosixPath],
        config: MergeConfig,
        output_path: Union[str, PosixPath],
    ) -> Optional[str]:
        """Merge models specifically for use with SDXLPipelineGenerator"""
        try:
            # Load checkpoints
            base_weights, base_metadata = load_checkpoint_dict(
                base_model, config.device
            )
            target_weights, target_metadata = load_checkpoint_dict(
                target_model, config.device
            )

            # Perform merge
            print(
                f"Merging {base_model} with {target_model} using {config.method.value}"
            )
            merged_weights = merge_checkpoints(base_weights, target_weights, config)

            # Create comprehensive metadata
            merged_metadata = {
                "merge_info": {
                    "base_model": str(base_model),
                    "target_model": str(target_model),
                    "method": config.method.value,
                    "alpha": config.alpha,
                    "timestamp": time.time(),
                },
                "model_info": {
                    "architecture": "sdxl",
                    "base_resolution": 1024,
                    "compatible_with": "SDXLPipelineGenerator",
                    "supported_resolutions": [
                        "1024x1024",
                        "1152x896",
                        "896x1152",
                        "1216x832",
                        "832x1216",
                        "1344x768",
                        "768x1344",
                        "1536x640",
                        "640x1536",
                    ],
                },
                "base_metadata": base_metadata,
                "target_metadata": target_metadata,
            }

            # Save merged model
            if save_checkpoint(merged_weights, output_path, merged_metadata):
                print(f"✓ Merged model saved: {output_path}")
                return str(output_path)
            else:
                print("✗ Failed to save merged model")
                return None

        except Exception as e:
            print(f"✗ Merge failed: {str(e)}")
            return None

    def clear_cache(self):
        """Clear cached checkpoints to free memory"""
        self.loaded_checkpoints.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Checkpoint cache cleared")

    def get_cache_info(self) -> Dict:
        """Get information about cached checkpoints"""
        return {
            "cached_models": list(self.loaded_checkpoints.keys()),
            "cache_size": len(self.loaded_checkpoints),
            "device": self.device,
        }
