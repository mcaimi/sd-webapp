#!/usr/bin/env python
"""
Cached API Access

Streamlit reruns the whole page script on every widget interaction.
These wrappers cache read-only API calls so a rerun doesn't refetch
model lists, metadata and system info from the backend each time.
Exceptions are never cached, so a failed call is retried on the next rerun.
"""

from typing import Any, Dict, List

import streamlit as st

from libs.shared.api_client import SDAPIClient
from libs.shared.config import get_app_config


@st.cache_resource
def get_api_client() -> SDAPIClient:
    """Get the shared API client (one HTTP session per server process)."""
    return SDAPIClient(get_app_config().api_url)


@st.cache_data(ttl=10, show_spinner=False)
def health_check() -> Dict[str, str]:
    """Check API health status."""
    return get_api_client().health_check()


@st.cache_data(ttl=10, show_spinner=False)
def get_system_info() -> Dict[str, Any]:
    """Get backend system information."""
    return get_api_client().get_system_info()


@st.cache_data(ttl=300, show_spinner=False)
def list_schedulers() -> List[str]:
    """List available noise schedulers."""
    return get_api_client().list_schedulers()


@st.cache_data(ttl=60, show_spinner=False)
def list_models(model_type: str, resource_type: str) -> Dict[str, Any]:
    """List available models of a given type."""
    return get_api_client().list_models(model_type, resource_type)


@st.cache_data(ttl=600, show_spinner=False)
def get_model_metadata(
    model_type: str, resource_type: str, model_name: str
) -> Dict[str, Any]:
    """Get safetensors header metadata for a model."""
    return get_api_client().get_model_metadata(model_type, resource_type, model_name)


def clear_model_cache() -> None:
    """Invalidate cached model listings (e.g. after a merge creates a checkpoint)."""
    list_models.clear()
