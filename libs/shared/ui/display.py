#!/usr/bin/env python
"""
Display-Only UI Components

Provides Streamlit UI for displaying results without file I/O side effects.
"""

import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

from libs.shared.api_client import SDAPIClient


def reset_streamlit_cache() -> None:
    """Clear Streamlit's resource cache."""
    st.cache_resource.clear()


def build_generation_metadata(
    model_name: Optional[str],
    lora_metadata: Dict,
    output_parameters: Dict,
    scheduler_config: Optional[Dict],
) -> Dict[str, Any]:
    """
    Build generation JSON metadata.

    The resulting dict can be loaded back via GenerationMetadata to
    restore generation parameters.

    Args:
        model_name: Checkpoint used for generation
        lora_metadata: LoRA metadata, {key: {"name", "merge_strength", ...}}
        output_parameters: Generation metadata returned by the API
        scheduler_config: Scheduler configuration

    Returns:
        Complete metadata dictionary
    """
    return {
        "model_name": model_name,
        "loras": [
            {
                "lora_name": lora.get("name"),
                "merge_strength": lora.get("merge_strength"),
            }
            for lora in (lora_metadata or {}).values()
        ],
        "output_parameters": output_parameters,
        "scheduler_config": scheduler_config,
    }


def display_download_buttons(
    image: Image.Image, gen_json: Dict[str, Any], file_stem: str
) -> None:
    """
    Render PNG and JSON download buttons for a generated image.

    Downloads don't trigger a rerun, so displayed results stay on screen.

    Args:
        image: Generated image
        gen_json: Generation metadata
        file_stem: Filename without extension (also used as widget key)
    """
    png_bytes = BytesIO()
    image.save(png_bytes, format="PNG")

    png_col, json_col = st.columns(2)
    png_col.download_button(
        "Image",
        data=png_bytes.getvalue(),
        file_name=f"{file_stem}.png",
        mime="image/png",
        icon=":material/download:",
        on_click="ignore",
        key=f"dl_png_{file_stem}",
        width="stretch",
    )
    json_col.download_button(
        "Parameters",
        data=json.dumps(gen_json, indent=2),
        file_name=f"{file_stem}.json",
        mime="application/json",
        icon=":material/data_object:",
        on_click="ignore",
        key=f"dl_json_{file_stem}",
        width="stretch",
    )


def display_generation_results(
    generated_pixmaps: List[Tuple],
    lora_metadata: Dict,
    model_type: str = "sd15",
    prefix: str = "",
) -> None:
    """
    Display generated images with download buttons.

    Images are persisted by the API backend (when storage.persist_to_backend
    is enabled there); nothing is written to local disk.

    Args:
        generated_pixmaps: List of (image, params, scheduler_config, seed, model_name) tuples
        lora_metadata: LoRA metadata dictionary
        model_type: Model family, used in download filenames
        prefix: Optional prefix for download filenames
    """
    if not generated_pixmaps:
        return

    st.success(
        f"Generation success! Inference produced {len(generated_pixmaps)} images:"
    )

    file_prefix = f"{model_type}_{prefix}" if prefix else model_type

    with st.container():
        for i, element in enumerate(generated_pixmaps):
            output_image, output_parameters, scheduler_config, gen_seed, model_name = element

            img_out, parms_out = st.columns(
                [2, 1], border=True, vertical_alignment="center"
            )

            with img_out:
                st.image(output_image, output_format="PNG")

            gen_json = build_generation_metadata(
                model_name, lora_metadata, output_parameters, scheduler_config
            )

            with parms_out:
                st.caption(f"{model_name} · seed {gen_seed}")
                st.json(gen_json, expanded=False)
                display_download_buttons(
                    output_image, gen_json, f"{file_prefix}_{gen_seed}_{i}"
                )


def display_device_info(api_client: SDAPIClient) -> None:
    """Display current device information in sidebar from API."""
    try:
        system_info = api_client.get_system_info()
        device = system_info.get("device", "unknown")
        gpu_name = system_info.get("gpu_name")

        if gpu_name:
            st.markdown(f"**Device: {device} - {gpu_name}**")
        else:
            st.markdown(f"**Device: {device}**")
    except Exception as e:
        st.warning(f"Could not fetch device info: {e}")
