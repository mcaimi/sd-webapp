#!/usr/bin/env python
"""
File Persistence Operations

Handles saving generated images and metadata to disk.
Separated from display logic for better separation of concerns.
"""

import json
import logging
from io import BytesIO
from typing import Dict, Any, Optional, Tuple

from libs.shared.config import get_app_config
from libs.shared.utils import random_string

logger = logging.getLogger(__name__)


def build_generation_metadata(
    model_metadata: Dict,
    lora_metadata: Dict,
    output_parameters: Dict,
    scheduler_config: Optional[Dict],
) -> Dict[str, Any]:
    """
    Build generation JSON metadata.

    Args:
        model_metadata: Model metadata
        lora_metadata: LoRA metadata
        output_parameters: Generation parameters
        scheduler_config: Scheduler configuration

    Returns:
        Complete metadata dictionary
    """
    return {
        "model_name": model_metadata.get("model_checkpoint")
        or model_metadata.get("name"),
        "loras": [
            {
                "lora_name": lora.name
                if hasattr(lora, "name")
                else lora_metadata.get(l, {}).get("name"),
                "merge_strength": lora.merge_strength
                if hasattr(lora, "merge_strength")
                else lora_metadata.get(l, {}).get("merge_strength"),
            }
            for l, lora in (
                lora_metadata.items() if isinstance(lora_metadata, dict) else []
            )
        ],
        "output_parameters": output_parameters,
        "scheduler_config": scheduler_config,
    }


def save_generation_output(
    output_image,
    output_parameters: Dict,
    model_metadata: Dict,
    lora_metadata: Dict,
    scheduler_config: Optional[Dict],
    seed: int,
    model_type: str = "sd15",
    prefix: str = "",
) -> Tuple[str, str]:
    """
    Save generated image and metadata to disk.

    Args:
        output_image: Generated image (numpy array)
        output_parameters: Generation parameters
        model_metadata: Model metadata
        lora_metadata: LoRA metadata
        scheduler_config: Scheduler configuration
        seed: Generation seed
        model_type: Either 'sd15' or 'sdxl'
        prefix: Optional prefix for filenames

    Returns:
        Tuple of (image_filename, json_filename)
    """
    from torchvision import transforms as tvT

    config = get_app_config()
    inference_uuid = random_string()

    # Build filename prefix
    file_prefix = f"{model_type}"
    if prefix:
        file_prefix = f"{file_prefix}_{prefix}"

    # Create generation JSON
    gen_json = build_generation_metadata(
        model_metadata, lora_metadata, output_parameters, scheduler_config
    )

    # Save JSON
    json_filename = f"{file_prefix}_{seed}_{inference_uuid}.json"
    json_path = config.get_output_path(json_filename, "json")
    logger.debug("Saving generation JSON: %s", json_path)
    with open(json_path, "w") as f:
        json.dump(gen_json, f)

    # Save image
    png_filename = f"{file_prefix}_{seed}_{inference_uuid}.png"
    png_path = config.get_output_path(png_filename, "images")
    logger.debug("Saving generated image: %s", png_path)

    pil_img = tvT.ToPILImage()
    png_bytes = BytesIO()
    pil_img(output_image).save(png_bytes, format="PNG")

    with open(png_path, "wb") as f:
        f.write(png_bytes.getvalue())

    return png_filename, json_filename
