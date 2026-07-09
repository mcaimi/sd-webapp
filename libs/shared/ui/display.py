#!/usr/bin/env python
"""
Display-Only UI Components

Provides Streamlit UI for displaying results without file I/O side effects.
"""

from typing import List, Tuple, Dict

import streamlit as st

from libs.shared.utils import get_gpu
from libs.shared.persistence import save_generation_output, build_generation_metadata


def reset_streamlit_cache() -> None:
    """Clear Streamlit's resource cache."""
    st.cache_resource.clear()


def display_generation_results(
    generated_pixmaps: List[Tuple],
    model_metadata: Dict,
    lora_metadata: Dict,
    model_type: str = "sd15",
    prefix: str = "",
) -> None:
    """
    Display generated images and save to disk.

    Args:
        generated_pixmaps: List of (image, params, scheduler_config, seed) tuples
        model_metadata: Model metadata dictionary
        lora_metadata: LoRA metadata dictionary
        model_type: Either 'sd15' or 'sdxl'
        prefix: Optional prefix for saved files
    """
    if not generated_pixmaps:
        return

    st.success(
        f"Generation success! Inference produced {len(generated_pixmaps)} images:"
    )

    with st.container():
        for element in generated_pixmaps:
            output_image, output_parameters, scheduler_config, gen_seed = element

            # Display results
            img_out, parms_out = st.columns(
                [2, 1], border=True, vertical_alignment="center"
            )

            with img_out:
                st.image(output_image, output_format="PNG")

            # Build generation JSON for display
            gen_json = build_generation_metadata(
                model_metadata, lora_metadata, output_parameters, scheduler_config
            )

            with parms_out:
                st.json(gen_json, expanded=False)

            # Save to disk
            save_generation_output(
                output_image=output_image,
                output_parameters=output_parameters,
                model_metadata=model_metadata,
                lora_metadata=lora_metadata,
                scheduler_config=scheduler_config,
                seed=gen_seed,
                model_type=model_type,
                prefix=prefix,
            )


def display_device_info() -> None:
    """Display current device information in sidebar."""
    device, dtype = get_gpu()
    st.markdown(f"**Device: {device} ({dtype})**")
