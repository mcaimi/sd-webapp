#!/usr/bin/env python
"""
Stable Diffusion Web Application

A Streamlit-based web interface for Stable Diffusion image generation.
Supports SD1.5 and SDXL models for text-to-image, inpainting, and model merging.
"""

import os
from pathlib import Path

import streamlit as st

from libs.shared.logging_config import setup_logging
from libs.shared.config import get_app_config

# Initialize logging before any other imports that might use it
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=Path("logs/sd_webapp.log") if os.getenv("LOG_TO_FILE") else None,
)


def main():
    """Initialize and run the Stable Diffusion web application."""
    # Load configuration (cached)
    config = get_app_config()
    
    # Load logo
    st.logo("assets/redhat.png")
    
    # Define application pages
    pages = {
        "Txt2Img": [
            st.Page(
                "pages/sd15_page.py",
                title="Stable Diffusion 1.5",
                icon=":material/chat:",
            ),
            st.Page(
                "pages/sdxl_page.py",
                title="Stable Diffusion XL",
                icon=":material/chat:",
            ),
        ],
        "Inpainting": [
            st.Page(
                "pages/sd15_inpaint_page.py",
                title="SD15 Inpainting",
                icon=":material/edit:",
            ),
            st.Page(
                "pages/sdxl_inpaint_page.py",
                title="SDXL Inpainting",
                icon=":material/edit:",
            ),
        ],
        "Model Merging": [
            st.Page(
                "pages/sd15_tools_page.py",
                title="SD15 Checkpoint Tools",
                icon=":material/settings:",
            ),
            st.Page(
                "pages/sdxl_tools_page.py",
                title="SDXL Checkpoint Tools",
                icon=":material/settings:",
            ),
        ],
    }
    
    # Setup navigation
    navigation = st.navigation(pages)
    
    # Configure page settings
    st.set_page_config(
        page_title="Stable Diffusion WebApp",
        layout="wide",
        page_icon=":material/edit:",
    )
    
    # Run the selected page
    navigation.run()


if __name__ == "__main__":
    main()
