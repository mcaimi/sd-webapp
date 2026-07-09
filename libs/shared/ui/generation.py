#!/usr/bin/env python
"""
Generation Settings UI Components

Provides Streamlit UI for generation parameters and prompts.
"""

from typing import Tuple

import streamlit as st

from libs.shared.models import GenerationSettings
from libs.globals.vars import schedulers


def create_generation_settings_ui(
    defaults: GenerationSettings,
    show_batch_size: bool = True,
    show_dimensions: bool = True,
) -> GenerationSettings:
    """
    Create the generation settings UI expander.

    Args:
        defaults: Default values for settings
        show_batch_size: Whether to show batch size control
        show_dimensions: Whether to show width/height controls

    Returns:
        Updated GenerationSettings
    """
    with st.expander("Generation Settings..."):
        guidance = st.slider(
            "Guidance Scale",
            value=defaults.guidance_scale,
            min_value=0.0,
            max_value=50.0,
            step=0.1,
        )

        width = defaults.width
        height = defaults.height
        if show_dimensions:
            with st.container(border=True):
                w, h = st.columns([1, 1])
                width = w.number_input("Image Width", value=defaults.width)
                height = h.number_input("Image Height", value=defaults.height)

        inference_steps = st.number_input(
            "Inference Steps",
            value=defaults.inference_steps,
        )

        batch_size = defaults.batch_size
        if show_batch_size:
            with st.container(border=True):
                batch_size = st.number_input("Batch Size", min_value=1, value=1)

        with st.container(border=True):
            sched, seedbox = st.columns([1, 1])
            scheduler_type = sched.selectbox(
                "Noise Scheduler",
                options=list(schedulers.keys()),
                index=defaults.scheduler_index,
            )
            seed = seedbox.number_input(
                "Random Seed",
                min_value=-1,
                max_value=None,
                value=defaults.seed,
                step=1,
                help="Generation Seed. -1 Means Random Seed",
            )

        scheduler_index = list(schedulers.keys()).index(scheduler_type)

    return GenerationSettings(
        positive_prompt=defaults.positive_prompt,
        negative_prompt=defaults.negative_prompt,
        guidance_scale=guidance,
        inference_steps=inference_steps,
        width=width,
        height=height,
        seed=seed,
        scheduler_index=scheduler_index,
        batch_size=batch_size,
    )


def create_prompt_inputs(
    positive_default: str = "",
    negative_default: str = "",
) -> Tuple[str, str]:
    """
    Create prompt input text areas.

    Args:
        positive_default: Default positive prompt
        negative_default: Default negative prompt

    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    positive_prompt = st.text_area(
        "Positive Prompt",
        value=positive_default,
        placeholder="Write here what you want in the image",
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        value=negative_default,
        placeholder="Write here what you don't want in the image",
    )
    return positive_prompt, negative_prompt
