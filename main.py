#!/usr/bin/env python
#
# Stable Diffusion Web App
# Streamlit version
#

import os
from datetime import datetime
import json

try:
    import streamlit as st
    from dotenv import dotenv_values
except Exception as e:
    print(f"Caught fatal exception: {e}")

# local imports
from libs.shared.settings import Properties

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# MAIN
if __name__ == "__main__":
    # load logo
    st.logo("assets/redhat.png")

    # define app pages
    sd15_page = st.Page(
        "pages/sd15_page.py", title="Stable Diffusion 1.5", icon=":material/chat:"
    )
    sdxl_page = st.Page(
        "pages/sdxl_page.py", title="Stable Diffusion XL", icon=":material/chat:"
    )
    explore_page = st.Page(
        "pages/sd15_tools_page.py", title="SD15 Checkpoint Tools", icon=":material/settings:"
    )
    enabled_sections = [sd15_page, sdxl_page, explore_page]

    # setup application main page
    pg = st.navigation(enabled_sections)
    st.set_page_config(
        page_title="Red Hat Opensource AI", layout="wide", page_icon=":material/edit:"
    )

    # run app
    pg.run()
