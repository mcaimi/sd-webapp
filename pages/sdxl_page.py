#!/usr/bin/env python
"""
Stable Diffusion XL Generation Page

Provides text-to-image generation using SDXL models via API backend with:
- Model and LoRA selection
- Custom VAE support
- Batch generation
- Model comparison mode
"""

import json

import streamlit as st

from libs.shared.models import GenerationSettings
from libs.shared.ui.generation import create_prompt_inputs, wait_for_job_with_progress
from libs.shared.ui.display import display_generation_results
from libs.shared import api_cache
from libs.shared.utils import GenerationMetadata
from libs.globals.vars import RANDOM_BIT_LENGTH
from libs.shared.utils import get_random_seed


# Configuration
MODEL_TYPE = "sdxl"
api_client = api_cache.get_api_client()

# Load header
st.html("assets/sdxl_header.html")


# === SIDEBAR ===
with st.sidebar:
    # Check API health
    try:
        health = api_cache.health_check()
        st.success(f"API: {health['status']}")
    except Exception as e:
        st.error(f"API unavailable: {e}")

    # Model selection
    try:
        models_response = api_cache.list_models(MODEL_TYPE, "checkpoints")
        model_options = {m["name"]: m for m in models_response["models"]}
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        model_options = {}

    selected_model = st.selectbox(
        label="Select SDXL Model",
        options=list(model_options.keys()),
        index=0,
    )

    # Load model metadata
    model_metadata = {"model_checkpoint": selected_model, "metadata": {}}
    if selected_model:
        with st.spinner("Loading Model Metadata..."):
            try:
                metadata_response = api_cache.get_model_metadata(
                    MODEL_TYPE, "checkpoints", selected_model
                )
                model_metadata["metadata"] = metadata_response.get("metadata", {})
            except Exception as e:
                st.warning(f"Could not load metadata: {e}")

    # LoRA selection
    try:
        loras_response = api_cache.list_models(MODEL_TYPE, "loras")
        lora_options = {m["name"]: m for m in loras_response["models"]}
    except Exception:
        lora_options = {}

    selected_lora = st.multiselect(
        label="Select SDXL Lora",
        options=list(lora_options.keys()),
        max_selections=5,
        default=[],
    )

    # LoRA metadata and strength sliders
    lora_metadata = {}
    lora_configs = []
    for i, lora_name in enumerate(selected_lora):
        strength = st.slider(
            label=f"{lora_name} merge strength",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            key=f"lora_strength_{i}",
        )
        lora_configs.append({"name": lora_name, "strength": strength})

        try:
            lora_meta = api_cache.get_model_metadata(MODEL_TYPE, "loras", lora_name)
            lora_metadata[f"lora_{i}"] = {
                "name": lora_name,
                "merge_strength": strength,
                "metadata": lora_meta.get("metadata", {}),
            }
        except Exception:
            lora_metadata[f"lora_{i}"] = {
                "name": lora_name,
                "merge_strength": strength,
                "metadata": {},
            }

    # VAE selection
    override_vae = st.checkbox("Override VAE", value=False)
    vae_metadata = None
    custom_vae = None
    if override_vae:
        try:
            vae_response = api_cache.list_models(MODEL_TYPE, "vaes")
            vae_options = {m["name"]: m for m in vae_response["models"]}
        except Exception:
            vae_options = {}

        selected_vae = st.selectbox(
            label="Select SDXL VAE",
            options=list(vae_options.keys()),
            index=0,
        )
        custom_vae = selected_vae

        if selected_vae:
            with st.spinner("Loading VAE Metadata..."):
                try:
                    vae_meta = api_cache.get_model_metadata(
                        MODEL_TYPE, "vaes", selected_vae
                    )
                    vae_metadata = {
                        "vae_checkpoint": selected_vae,
                        "metadata": vae_meta.get("metadata", {}),
                    }
                except Exception:
                    vae_metadata = {
                        "vae_checkpoint": selected_vae,
                        "metadata": {},
                    }

    # System info
    try:
        sys_info = api_cache.get_system_info()
        st.markdown(f"**Device: {sys_info['device']}**")
    except Exception:
        st.markdown("**Device: Unknown**")


# === MAIN PAGE ===
st.markdown("### **Stable Diffusion Generation Page, XL**")
st.markdown("*Generate images using Stable Diffusion XL models.*")

# Fetch available schedulers from API
try:
    available_schedulers = api_cache.list_schedulers()
except Exception as e:
    st.error(f"Failed to fetch schedulers: {e}")
    available_schedulers = ["DPM++ 2M"]  # Fallback default

# Default settings (SDXL uses larger default resolution)
defaults = GenerationSettings(
    width=832,
    height=1216,
    inference_steps=20,
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
prev_metadata = None
if prev_gen_file is not None:
    try:
        prev_metadata = GenerationMetadata(json.load(prev_gen_file))
    except (ValueError, json.JSONDecodeError) as e:
        st.error(f"Invalid parameters file: {e}")
if prev_metadata is not None:
    defaults.positive_prompt = prev_metadata.prompt or ""
    defaults.negative_prompt = prev_metadata.negative_prompt or ""
    defaults.guidance_scale = prev_metadata.guidance_scale or defaults.guidance_scale
    defaults.inference_steps = (
        prev_metadata.num_inference_steps or defaults.inference_steps
    )
    defaults.width = prev_metadata.width or defaults.width
    defaults.height = prev_metadata.height or defaults.height
    defaults.seed = prev_metadata.seed or defaults.seed
    try:
        defaults.scheduler_index = available_schedulers.index(
            prev_metadata.scheduler
        )
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
        "Guidance Scale",
        value=defaults.guidance_scale,
        min_value=0.0,
        max_value=50.0,
        step=0.1,
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
        scheduler_type = sched.selectbox(
            "Noise Scheduler",
            options=available_schedulers,
            index=min(defaults.scheduler_index, len(available_schedulers) - 1),
        )
        seed = seedbox.number_input(
            "Random Seed",
            min_value=-1,
            max_value=None,
            value=defaults.seed,
            step=1,
            help="Generation Seed. -1 Means Random Seed",
        )

# === TABS ===
RESULTS_KEY = f"{MODEL_TYPE}_generated_pixmaps"
image_gen_tab, model_comparison_tab = st.tabs(["Image Generation", "Model Comparison"])



def wait_for_image(job_id: str, label: str):
    """Poll a generation job, showing progress, and return (image, result)."""
    result = wait_for_job_with_progress(api_client, job_id, label)["result"]
    return api_client.decode_image_from_result(result), result

with image_gen_tab:
    gen_info_col, gen_btn_col = st.columns([2, 1])

    with gen_info_col:
        st.markdown(
            f"**Generate Images using model {model_metadata.get('model_checkpoint')}**."
        )

    with gen_btn_col:
        submit_button = st.button("Generate", type="primary")

    if submit_button:
        generated_pixmaps = []

        if batch_size > 1 and seed > 0:
            st.warning(
                f"Seed {seed} is constant and batch size is {batch_size}: Images will be identical."
            )

        requests = [
            {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "model_checkpoint": selected_model,
                "width": int(width),
                "height": int(height),
                "steps": int(inference_steps),
                "cfg_scale": guidance,
                "seed": seed if seed > 0 else get_random_seed(RANDOM_BIT_LENGTH),
                "scheduler": scheduler_type,
                "loras": lora_configs or None,
                "custom_vae": custom_vae,
            }
            for _ in range(batch_size)
        ]
        scheduler_config = {"_class_name": scheduler_type}

        try:
            job_ids = api_client.batch_generate(MODEL_TYPE, requests)["job_ids"]
        except Exception as e:
            st.error(f"Generation failed: {e}")
            job_ids = []

        for i, job_id in enumerate(job_ids):
            try:
                image, result = wait_for_image(job_id, f"Image {i + 1}/{len(job_ids)}")
                generated_pixmaps.append(
                    (
                        image,
                        result.get("metadata", {}),
                        scheduler_config,
                        result.get("seed", requests[i]["seed"]),
                        selected_model,
                    )
                )
            except Exception as e:
                st.error(f"Image {i + 1} failed: {e}")

        st.session_state[RESULTS_KEY] = (generated_pixmaps, lora_metadata)

with model_comparison_tab:
    comp_info_col, comp_btn_col = st.columns([2, 1])

    with comp_info_col:
        st.markdown(
            "**Generate images with consistent parameters across selected models.**"
        )
        target_models = st.multiselect(
            label="Select target models",
            max_selections=6,
            help="Generate image using the same settings across different models",
            options=list(model_options.keys()),
            default=[],
        )

    with comp_btn_col:
        gen_button = st.button(
            "Generate over Models", type="primary", disabled=(len(target_models) == 0)
        )

    if gen_button:
        generated_pixmaps = []
        gen_seed = seed if seed > 0 else get_random_seed(RANDOM_BIT_LENGTH)
        scheduler_config = {"_class_name": scheduler_type}

        try:
            results = api_client.compare_models(
                model_type=MODEL_TYPE,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                models=target_models,
                width=int(width),
                height=int(height),
                steps=int(inference_steps),
                cfg_scale=guidance,
                seed=gen_seed,
                scheduler=scheduler_type,
            )
            comparisons = results.get("comparisons", [])
        except Exception as e:
            st.error(f"Model comparison failed: {e}")
            comparisons = []

        for i, comparison in enumerate(comparisons):
            model_name = comparison["model"]
            try:
                image, result = wait_for_image(
                    comparison["job_id"], f"{model_name} ({i + 1}/{len(comparisons)})"
                )
                generated_pixmaps.append(
                    (
                        image,
                        result.get("metadata", {}),
                        scheduler_config,
                        result.get("seed", gen_seed),
                        model_name,
                    )
                )
            except Exception as e:
                st.error(f"Failed to generate with {model_name}: {e}")

        st.session_state[RESULTS_KEY] = (generated_pixmaps, lora_metadata)

# === OUTPUT SECTION ===
if st.session_state.get(RESULTS_KEY, ([], {}))[0]:
    stored_pixmaps, stored_lora_metadata = st.session_state[RESULTS_KEY]
    display_generation_results(
        generated_pixmaps=stored_pixmaps,
        lora_metadata=stored_lora_metadata,
        model_type=MODEL_TYPE,
    )
else:
    st.info("Select generation method and perform inference.")
