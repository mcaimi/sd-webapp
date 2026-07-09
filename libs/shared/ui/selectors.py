#!/usr/bin/env python
"""
Model and Component Selectors

Provides Streamlit UI components for selecting models, LoRAs, and VAEs.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import streamlit as st

from libs.shared.models import ModelSelection, LoraSelection
from libs.shared.config import get_app_config
from libs.shared.utils import enumerate_models, read_safetensors_header

logger = logging.getLogger(__name__)


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

    if not selected_loras:
        return lora_selections

    # Load metadata in parallel for better performance
    def load_lora_metadata(i: int, lora_name: str, lora_path: Optional[Path]) -> Tuple:
        """Load LoRA metadata (runs in thread pool)."""
        try:
            metadata = read_safetensors_header(lora_path) if lora_path else {}
            return i, lora_name, lora_path, metadata, None
        except Exception as e:
            logger.warning("Failed to load LoRA metadata for %s: %s", lora_name, e)
            return i, lora_name, None, {}, e

    with st.spinner("Loading LoRA metadata..."):
        # Pre-load all metadata in parallel (I/O bound operation)
        metadata_results = {}
        with ThreadPoolExecutor(max_workers=min(4, len(selected_loras))) as executor:
            futures = {
                executor.submit(
                    load_lora_metadata, i, lora_name, available_loras.get(lora_name)
                ): i
                for i, lora_name in enumerate(selected_loras)
            }

            for future in as_completed(futures):
                i, lora_name, lora_path, metadata, error = future.result()
                metadata_results[i] = (lora_name, lora_path, metadata, error)

        # Create UI elements in main thread (must be sequential)
        for i, lora_name in enumerate(selected_loras):
            lora_name_loaded, lora_path, metadata, error = metadata_results[i]

            strength = st.slider(
                label=f"{lora_name} merge strength",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                key=f"{model_type}_lora_strength_{i}{key_suffix}",
            )

            if error is None:
                lora_selections[f"lora_{i}"] = LoraSelection(
                    name=lora_name_loaded,
                    path=lora_path.absolute() if lora_path else None,
                    merge_strength=strength,
                    metadata=metadata,
                )
            else:
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

        # Device info (imported from display module)
        from libs.shared.ui.display import display_device_info
        display_device_info()

    return model, loras, override_vae, vae
