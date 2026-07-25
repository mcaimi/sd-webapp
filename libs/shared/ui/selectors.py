#!/usr/bin/env python
"""
Model and Component Selectors

Provides Streamlit UI components for selecting models, LoRAs, and VAEs.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, Callable

import streamlit as st

from libs.shared.models import ModelSelection, LoraSelection
from libs.shared.api_client import SDAPIClient

logger = logging.getLogger(__name__)


def load_model_metadata(
    model_name: str,
    model_type: str,
    resource_type: str,
    api_client: SDAPIClient,
    fallback_name: str = "no model available",
) -> ModelSelection:
    """
    Load model metadata from API safely with fallback.

    Args:
        model_name: Name of the model
        model_type: Either 'sd15' or 'sdxl'
        resource_type: Either 'checkpoints', 'loras', or 'vaes'
        api_client: API client instance
        fallback_name: Name to use if loading fails

    Returns:
        ModelSelection with metadata loaded
    """
    try:
        metadata = api_client.get_model_metadata(model_type, resource_type, model_name)
        return ModelSelection(
            name=model_name,
            path=None,  # Path is server-side, not relevant for UI
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
    api_client: SDAPIClient,
    on_change: Optional[Callable] = None,
    key_suffix: str = "",
) -> Optional[ModelSelection]:
    """
    Create a model selector widget with metadata loading from API.

    Args:
        label: Label for the selectbox
        model_type: Either 'sd15' or 'sdxl'
        category: Either 'checkpoints', 'loras', or 'vaes'
        api_client: API client instance
        on_change: Optional callback when selection changes
        key_suffix: Optional suffix for widget key uniqueness

    Returns:
        ModelSelection if a model is selected, None otherwise
    """
    try:
        models_response = api_client.list_models(model_type, category)
        available_models = models_response.get("models", [])
    except Exception as e:
        st.error(f"Failed to fetch {category}: {e}")
        return None

    if not available_models:
        st.warning(f"No {category} found")
        return None

    # Extract model names from response
    model_names = [m["name"] for m in available_models]

    selected = st.selectbox(
        label=label,
        options=model_names,
        index=0,
        on_change=on_change,
        key=f"{model_type}_{category}_select{key_suffix}",
    )

    if selected:
        with st.spinner(f"Loading metadata..."):
            return load_model_metadata(
                model_name=selected,
                model_type=model_type,
                resource_type=category,
                api_client=api_client,
            )
    return None


def create_lora_selector(
    label: str,
    model_type: str,
    api_client: SDAPIClient,
    max_selections: int = 5,
    on_change: Optional[Callable] = None,
    key_suffix: str = "",
) -> Dict[str, LoraSelection]:
    """
    Create a multi-select LoRA selector with strength sliders using API.

    Args:
        label: Label for the multiselect
        model_type: Either 'sd15' or 'sdxl'
        api_client: API client instance
        max_selections: Maximum number of LoRAs that can be selected
        on_change: Optional callback when selection changes
        key_suffix: Optional suffix for widget key uniqueness

    Returns:
        Dictionary of LoRA name to LoraSelection
    """
    try:
        loras_response = api_client.list_models(model_type, "loras")
        available_loras = loras_response.get("models", [])
        lora_names = [m["name"] for m in available_loras]
    except Exception as e:
        st.error(f"Failed to fetch LoRAs: {e}")
        return {}

    selected_loras = st.multiselect(
        label=label,
        options=lora_names,
        max_selections=max_selections,
        default=[],
        on_change=on_change,
        key=f"{model_type}_lora_select{key_suffix}",
    )

    lora_selections = {}

    if not selected_loras:
        return lora_selections

    # Load metadata in parallel for better performance
    def load_lora_metadata(i: int, lora_name: str) -> Tuple:
        """Load LoRA metadata from API (runs in thread pool)."""
        try:
            metadata = api_client.get_model_metadata(model_type, "loras", lora_name)
            return i, lora_name, metadata, None
        except Exception as e:
            logger.warning("Failed to load LoRA metadata for %s: %s", lora_name, e)
            return i, lora_name, {}, e

    with st.spinner("Loading LoRA metadata..."):
        # Pre-load all metadata in parallel (I/O bound operation)
        metadata_results = {}
        with ThreadPoolExecutor(max_workers=min(4, len(selected_loras))) as executor:
            futures = {
                executor.submit(load_lora_metadata, i, lora_name): i
                for i, lora_name in enumerate(selected_loras)
            }

            for future in as_completed(futures):
                i, lora_name, metadata, error = future.result()
                metadata_results[i] = (lora_name, metadata, error)

        # Create UI elements in main thread (must be sequential)
        for i, lora_name in enumerate(selected_loras):
            lora_name_loaded, metadata, error = metadata_results[i]

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
                    path=None,  # Server-side path
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
    api_client: SDAPIClient,
    on_change: Optional[Callable] = None,
    key_suffix: str = "",
) -> Tuple[bool, Optional[ModelSelection]]:
    """
    Create a VAE override selector using API.

    Args:
        model_type: Either 'sd15' or 'sdxl'
        api_client: API client instance
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
        category="vaes",
        api_client=api_client,
        on_change=on_change,
        key_suffix=key_suffix,
    )

    return True, vae_selection


def create_sidebar_model_selection(
    model_type: str,
    api_client: SDAPIClient,
    on_change: Optional[Callable] = None,
) -> Tuple[
    Optional[ModelSelection], Dict[str, LoraSelection], bool, Optional[ModelSelection]
]:
    """
    Create the complete sidebar with model, LoRA, and VAE selection using API.

    Args:
        model_type: Either 'sd15' or 'sdxl'
        api_client: API client instance
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
            api_client=api_client,
            on_change=on_change,
        )

        # LoRA selection
        loras = create_lora_selector(
            label=f"Select {model_type.upper()} Lora",
            model_type=model_type,
            api_client=api_client,
            on_change=on_change,
        )

        # VAE selection
        override_vae, vae = create_vae_selector(
            model_type=model_type,
            api_client=api_client,
            on_change=on_change,
        )

        # Device info (imported from display module)
        from libs.shared.ui.display import display_device_info

        display_device_info(api_client)

    return model, loras, override_vae, vae
