#!/usr/bin/env python
"""
Stable Diffusion 1.5 Generation Page

Provides text-to-image generation using SD1.5 models with:
- Model and LoRA selection
- Custom VAE support
- Batch generation
- Model comparison mode
"""

import json

import streamlit as st

from libs.shared.config import get_app_config
from libs.shared.utils import enumerate_models, read_safetensors_header, random_string
from libs.shared.ui_components import (
    GenerationSettings,
    create_prompt_inputs,
    display_generation_results,
    reset_streamlit_cache,
)
from libs.stablediffusion.sd15 import SD15PipelineGenerator, load_custom_vae
from libs.stablediffusion.metadata import GenerationMetadata
from libs.globals.vars import schedulers, RANDOM_BIT_LENGTH
from libs.stablediffusion.funcs import get_random_seed
from libs.shared.utils import get_gpu


# Configuration
MODEL_TYPE = "sd15"
config = get_app_config()

# Load header
st.html("assets/sd15_header.html")


def reset_model_cache() -> None:
    """Clear cached models when selection changes."""
    reset_streamlit_cache()


@st.cache_resource
def load_sd15_model(model_path):
    """Load and cache an SD1.5 model."""
    return SD15PipelineGenerator(model_checkpoint=model_path)


# === SIDEBAR ===
with st.sidebar:
    # Model selection
    model_options = enumerate_models(config.checkpoints_sd15_path)
    selected_model = st.selectbox(
        label="Select SD15 Model",
        options=list(model_options.keys()),
        index=0,
        on_change=reset_model_cache,
    )

    # Load model metadata
    with st.spinner("Loading Model Metadata..."):
        try:
            model_metadata = {
                "model_checkpoint": selected_model,
                "model_path": model_options.get(selected_model).absolute() if selected_model else None,
                "metadata": read_safetensors_header(model_options.get(selected_model)) if selected_model else {},
            }
        except Exception:
            model_metadata = {
                "model_checkpoint": "no model available",
                "model_path": None,
                "metadata": {},
            }

    # LoRA selection
    lora_options = enumerate_models(config.loras_sd15_path)
    selected_lora = st.multiselect(
        label="Select SD15 Lora",
        options=list(lora_options.keys()),
        max_selections=5,
        default=[],
        on_change=reset_model_cache,
    )

    # LoRA metadata and strength sliders
    with st.spinner("Loading Lora Metadata..."):
        lora_metadata = {}
        for i, lora_name in enumerate(selected_lora):
            try:
                lora_path = lora_options.get(lora_name)
                lora_metadata[f"lora_{i}"] = {
                    "name": lora_name,
                    "lora_path": lora_path.absolute() if lora_path else None,
                    "merge_strength": st.slider(
                        label=f"{lora_name} merge strength",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        step=0.1,
                        key=f"lora_strength_{i}",
                    ),
                    "metadata": read_safetensors_header(lora_path) if lora_path else {},
                }
            except Exception:
                lora_metadata[f"lora_{i}"] = {
                    "name": "not available",
                    "lora_path": None,
                    "merge_strength": 0,
                    "metadata": {},
                }

    # VAE selection
    override_vae = st.checkbox("Override VAE", value=False, on_change=reset_model_cache)
    vae_metadata = None
    if override_vae:
        vae_options = enumerate_models(config.vae_sd15_path)
        selected_vae = st.selectbox(
            label="Select SD15 VAE",
            options=list(vae_options.keys()),
            index=0,
            on_change=reset_model_cache,
        )

        with st.spinner("Loading VAE Metadata..."):
            try:
                vae_path = vae_options.get(selected_vae)
                vae_metadata = {
                    "vae_checkpoint": selected_vae,
                    "vae_path": vae_path.absolute() if vae_path else None,
                    "metadata": read_safetensors_header(vae_path) if vae_path else {},
                }
            except Exception:
                vae_metadata = {
                    "vae_checkpoint": "no available vae",
                    "vae_path": None,
                    "metadata": {},
                }

    st.markdown(f"**Device: {get_gpu()}**")


# === MAIN PAGE ===
st.markdown("### **Stable Diffusion Generation Page, v1.5**")
st.markdown("*Generate images using Stable Diffusion 1.5 models.*")

# Default settings
defaults = GenerationSettings(
    width=512,
    height=768,
    inference_steps=50,
    guidance_scale=7.0,
    seed=-1,
    scheduler_index=0,
)

# Load previous generation parameters
prev_gen_file = st.file_uploader(
    "Load Inference Parameters from file...",
    accept_multiple_files=False,
    type="json",
)
if prev_gen_file is not None:
    prev_metadata = GenerationMetadata(json.load(prev_gen_file))
    defaults.positive_prompt = prev_metadata.prompt or ""
    defaults.negative_prompt = prev_metadata.negative_prompt or ""
    defaults.guidance_scale = prev_metadata.guidance_scale or defaults.guidance_scale
    defaults.inference_steps = prev_metadata.num_inference_steps or defaults.inference_steps
    defaults.width = prev_metadata.width or defaults.width
    defaults.height = prev_metadata.height or defaults.height
    defaults.seed = prev_metadata.seed or defaults.seed
    try:
        defaults.scheduler_index = list(schedulers.keys()).index(prev_metadata.scheduler)
    except (ValueError, AttributeError):
        pass

# Prompt inputs
positive_prompt, negative_prompt = create_prompt_inputs(
    defaults.positive_prompt,
    defaults.negative_prompt,
)

# Generation settings
with st.expander("Generation Settings..."):
    guidance = st.slider(
        "Guidance Scale", value=defaults.guidance_scale, min_value=0.0, max_value=50.0, step=0.1
    )
    with st.container(border=True):
        w, h = st.columns([1, 1])
        width = w.number_input("Image Width", value=defaults.width)
        height = h.number_input("Image Height", value=defaults.height)

    inference_steps = st.number_input("Inference Steps", value=defaults.inference_steps)

    with st.container(border=True):
        batch_size = st.number_input("Batch Size", min_value=1, value=1)

    with st.container(border=True):
        sched, seedbox = st.columns([1, 1])
        scheduler_type = sched.selectbox("Noise Scheduler", options=schedulers, index=defaults.scheduler_index)
        seed = seedbox.number_input(
            "Random Seed", min_value=-1, max_value=None, value=defaults.seed, step=1,
            help="Generation Seed. -1 Means Random Seed",
        )

# === TABS ===
image_gen_tab, model_comparison_tab = st.tabs(["Image Generation", "Model Comparison"])

with image_gen_tab:
    gen_info_col, gen_btn_col = st.columns([2, 1])

    with gen_info_col:
        st.markdown(f"**Generate Images using model {model_metadata.get('model_checkpoint')}**.")

    with gen_btn_col:
        submit_button = st.button("Generate", type="primary")

    if submit_button:
        with st.spinner(f"Loading Stable Diffusion Model {model_metadata.get('model_checkpoint')}..."):
            sd_generator = load_sd15_model(model_metadata.get("model_path"))
            sd_generator.loadSDPipeline()

        with st.spinner("Merging LoRA Adapters..."):
            sd_generator.addLorasToPipeline(loras=lora_metadata)

        if override_vae and vae_metadata:
            with st.spinner(f"Loading VAE {vae_metadata.get('vae_path')}..."):
                sd_generator.pipeline.vae = load_custom_vae(vae_metadata.get("vae_path"))

        with st.spinner(f"Moving pipeline to device: {sd_generator.accelerator}"):
            sd_generator.pipeToConfiguredDevice()

        # Run inference
        generated_pixmaps = []
        if batch_size > 1 and seed > 0:
            st.warning(f"Seed {seed} is constant and batch size is {batch_size}: Images will be identical.")

        for i in range(batch_size):
            gen_seed = seed if seed > 0 else get_random_seed(RANDOM_BIT_LENGTH)
            with st.spinner(f"Generating image {i + 1}/{batch_size}..."):
                output_image, output_parameters = sd_generator.forward(
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    steps=inference_steps,
                    scheduler_type=scheduler_type,
                    width=width,
                    height=height,
                    seed=gen_seed,
                    cfg=guidance,
                )
                scheduler_config = sd_generator.getSchedulerConfig()
            generated_pixmaps.append((output_image, output_parameters, scheduler_config, gen_seed))

with model_comparison_tab:
    comp_info_col, comp_btn_col = st.columns([2, 1])

    with comp_info_col:
        st.markdown("**Generate images with consistent parameters across selected models.**")
        target_models = st.multiselect(
            label="Select target models",
            max_selections=6,
            help="Generate image using the same settings across different models",
            options=list(model_options.keys()),
            default=[],
        )

    with comp_btn_col:
        gen_button = st.button("Generate over Models", type="primary", disabled=(len(target_models) == 0))

    if gen_button:
        generated_pixmaps = []
        gen_seed = seed if seed > 0 else get_random_seed(RANDOM_BIT_LENGTH)

        for i, model in enumerate(target_models):
            model_metadata["model_checkpoint"] = model
            ckpt = model_options.get(model)
            pipeline = load_sd15_model(ckpt.absolute())
            pipeline.loadSDPipeline()

            with st.spinner("Merging LoRA Adapters..."):
                pipeline.addLorasToPipeline(loras=lora_metadata)

            if override_vae and vae_metadata:
                with st.spinner(f"Loading VAE {vae_metadata.get('vae_path')}..."):
                    pipeline.pipeline.vae = load_custom_vae(vae_metadata.get("vae_path"))

            with st.spinner(f"Moving pipeline to device: {pipeline.accelerator}"):
                pipeline.pipeToConfiguredDevice()

            with st.spinner(f"Generating image {i + 1} with {model}..."):
                output_image, output_parameters = pipeline.forward(
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    steps=inference_steps,
                    scheduler_type=scheduler_type,
                    width=width,
                    height=height,
                    seed=gen_seed,
                    cfg=guidance,
                )
                scheduler_config = pipeline.getSchedulerConfig()
            generated_pixmaps.append((output_image, output_parameters, scheduler_config, gen_seed))
            reset_model_cache()

# === OUTPUT SECTION ===
try:
    if generated_pixmaps:
        display_generation_results(
            generated_pixmaps=generated_pixmaps,
            model_metadata=model_metadata,
            lora_metadata=lora_metadata,
            model_type=MODEL_TYPE,
        )
except NameError:
    st.info("Select generation method and perform inference.")
