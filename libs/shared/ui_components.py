#!/usr/bin/env python
"""
Shared Streamlit UI Components

Provides reusable UI components and patterns for the Stable Diffusion web app.
This reduces code duplication across pages while maintaining consistency.
"""

import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Type

import streamlit as st
from PIL import Image

from libs.shared.config import get_app_config
from libs.shared.utils import (
    enumerate_models,
    read_safetensors_header,
    random_string,
    get_gpu,
)
from libs.globals.vars import schedulers, RANDOM_BIT_LENGTH
from libs.stablediffusion.funcs import get_random_seed

logger = logging.getLogger(__name__)


@dataclass
class ModelSelection:
    """Container for model selection data."""
    name: str
    path: Optional[Path]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class LoraSelection:
    """Container for LoRA selection data."""
    name: str
    path: Optional[Path]
    merge_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationSettings:
    """Container for image generation settings."""
    positive_prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 7.0
    inference_steps: int = 50
    width: int = 512
    height: int = 768
    seed: int = -1
    scheduler_index: int = 0
    batch_size: int = 1


def load_model_metadata(
    model_name: str,
    model_path: Path,
    fallback_name: str = "no model available"
) -> ModelSelection:
    """
    Load model metadata safely with fallback.
    
    Args:
        model_name: Name of the model
        model_path: Path to the model file
        fallback_name: Name to use if loading fails
        
    Returns:
        ModelSelection with metadata loaded
    """
    try:
        metadata = read_safetensors_header(model_path)
        return ModelSelection(
            name=model_name,
            path=model_path.absolute() if model_path else None,
            metadata=metadata,
        )
    except Exception:
        return ModelSelection(
            name=fallback_name,
            path=None,
            metadata={},
        )


def create_model_selector(
    label: str,
    model_type: str,
    category: str,
    on_change: Optional[Callable] = None,
    key_suffix: str = "",
) -> Optional[ModelSelection]:
    """
    Create a model selector widget with metadata loading.
    
    Args:
        label: Label for the selectbox
        model_type: Either 'sd15' or 'sdxl'
        category: Either 'checkpoints', 'loras', or 'vae'
        on_change: Optional callback when selection changes
        key_suffix: Optional suffix for widget key uniqueness
        
    Returns:
        ModelSelection if a model is selected, None otherwise
    """
    config = get_app_config()
    paths = config.get_model_paths(model_type)
    model_path = paths[category]
    
    available_models = enumerate_models(model_path)
    
    if not available_models:
        st.warning(f"No {category} found in {model_path}")
        return None
    
    selected = st.selectbox(
        label=label,
        options=list(available_models.keys()),
        index=0,
        on_change=on_change,
        key=f"{model_type}_{category}_select{key_suffix}",
    )
    
    if selected:
        with st.spinner(f"Loading metadata..."):
            return load_model_metadata(
                model_name=selected,
                model_path=available_models.get(selected),
            )
    return None


def create_lora_selector(
    label: str,
    model_type: str,
    max_selections: int = 5,
    on_change: Optional[Callable] = None,
    key_suffix: str = "",
) -> Dict[str, LoraSelection]:
    """
    Create a multi-select LoRA selector with strength sliders.
    
    Args:
        label: Label for the multiselect
        model_type: Either 'sd15' or 'sdxl'
        max_selections: Maximum number of LoRAs that can be selected
        on_change: Optional callback when selection changes
        key_suffix: Optional suffix for widget key uniqueness
        
    Returns:
        Dictionary of LoRA name to LoraSelection
    """
    config = get_app_config()
    paths = config.get_model_paths(model_type)
    lora_path = paths["loras"]
    
    available_loras = enumerate_models(lora_path)
    
    selected_loras = st.multiselect(
        label=label,
        options=list(available_loras.keys()),
        max_selections=max_selections,
        default=[],
        on_change=on_change,
        key=f"{model_type}_lora_select{key_suffix}",
    )
    
    lora_selections = {}
    
    with st.spinner("Loading LoRA metadata..."):
        for i, lora_name in enumerate(selected_loras):
            lora_file_path = available_loras.get(lora_name)
            
            try:
                strength = st.slider(
                    label=f"{lora_name} merge strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.1,
                    key=f"{model_type}_lora_strength_{i}{key_suffix}",
                )
                
                metadata = read_safetensors_header(lora_file_path) if lora_file_path else {}
                
                lora_selections[f"lora_{i}"] = LoraSelection(
                    name=lora_name,
                    path=lora_file_path.absolute() if lora_file_path else None,
                    merge_strength=strength,
                    metadata=metadata,
                )
            except Exception:
                lora_selections[f"lora_{i}"] = LoraSelection(
                    name="not available",
                    path=None,
                    merge_strength=0,
                    metadata={},
                )
    
    return lora_selections


def create_vae_selector(
    model_type: str,
    on_change: Optional[Callable] = None,
    key_suffix: str = "",
) -> Tuple[bool, Optional[ModelSelection]]:
    """
    Create a VAE override selector.
    
    Args:
        model_type: Either 'sd15' or 'sdxl'
        on_change: Optional callback when selection changes
        key_suffix: Optional suffix for widget key uniqueness
        
    Returns:
        Tuple of (override_enabled, vae_selection or None)
    """
    override_vae = st.checkbox(
        "Override VAE",
        value=False,
        on_change=on_change,
        key=f"{model_type}_vae_override{key_suffix}",
    )
    
    if not override_vae:
        return False, None
    
    vae_selection = create_model_selector(
        label=f"Select {model_type.upper()} VAE",
        model_type=model_type,
        category="vae",
        on_change=on_change,
        key_suffix=key_suffix,
    )
    
    return True, vae_selection


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
    gen_json = {
        "model_name": model_metadata.get("model_checkpoint") or model_metadata.get("name"),
        "loras": [
            {
                "lora_name": lora.name if hasattr(lora, 'name') else lora_metadata.get(l, {}).get("name"),
                "merge_strength": lora.merge_strength if hasattr(lora, 'merge_strength') else lora_metadata.get(l, {}).get("merge_strength"),
            }
            for l, lora in (lora_metadata.items() if isinstance(lora_metadata, dict) else [])
        ],
        "output_parameters": output_parameters,
        "scheduler_config": scheduler_config,
    }
    
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
    
    st.success(f"Generation success! Inference produced {len(generated_pixmaps)} images:")
    
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
            gen_json = {
                "model_name": model_metadata.get("model_checkpoint") or model_metadata.get("name"),
                "loras": [
                    {
                        "lora_name": lora_metadata.get(l, {}).get("name"),
                        "merge_strength": lora_metadata.get(l, {}).get("merge_strength"),
                    }
                    for l in lora_metadata
                ] if isinstance(lora_metadata, dict) else [],
                "output_parameters": output_parameters,
                "scheduler_config": scheduler_config,
            }
            
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


def get_scheduler_from_name(scheduler_name: str) -> Optional[Type]:
    """
    Get scheduler class from name.

    Args:
        scheduler_name: Name of the scheduler

    Returns:
        Scheduler class or None if not found
    """
    return schedulers.get(scheduler_name)


def get_scheduler_names() -> List[str]:
    """Get list of available scheduler names."""
    return list(schedulers.keys())


def display_device_info() -> None:
    """Display current device information in sidebar."""
    st.markdown(f"**Device: {get_gpu()}**")


def reset_streamlit_cache() -> None:
    """Clear Streamlit's resource cache."""
    st.cache_resource.clear()


def create_sidebar_model_selection(
    model_type: str,
    on_change: Optional[Callable] = None,
) -> Tuple[Optional[ModelSelection], Dict[str, LoraSelection], bool, Optional[ModelSelection]]:
    """
    Create the complete sidebar with model, LoRA, and VAE selection.
    
    Args:
        model_type: Either 'sd15' or 'sdxl'
        on_change: Callback for when selections change
        
    Returns:
        Tuple of (model_selection, lora_selections, override_vae, vae_selection)
    """
    with st.sidebar:
        # Model selection
        model = create_model_selector(
            label=f"Select {model_type.upper()} Model",
            model_type=model_type,
            category="checkpoints",
            on_change=on_change,
        )
        
        # LoRA selection
        loras = create_lora_selector(
            label=f"Select {model_type.upper()} Lora",
            model_type=model_type,
            on_change=on_change,
        )
        
        # VAE selection
        override_vae, vae = create_vae_selector(
            model_type=model_type,
            on_change=on_change,
        )
        
        # Device info
        display_device_info()
    
    return model, loras, override_vae, vae

