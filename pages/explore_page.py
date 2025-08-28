#!/usr/bin/env python

try:
    import streamlit as st
    from dotenv import dotenv_values

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.utils import enumerate_models, read_safetensors_header, check_or_create_path
        from libs.stablediffusion.sd15 import gen_noise, load_custom_vae
        from libs.stablediffusion.sd15 import SD15PipelineGenerator
        from libs.globals.vars import schedulers
except Exception as e:
    print(f"Caught fatal exception: {e}")

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# load header
st.html("assets/explore_header.html")

# paths sanity check
with st.spinner("Checking Paths"):
    for path in [ appSettings.config_parameters.checkpoints.sd15.path, appSettings.config_parameters.loras.sd15.path,
                appSettings.config_parameters.checkpoints.sdxl.path, appSettings.config_parameters.loras.sdxl.path,
                appSettings.config_parameters.vae.sdxl.path, appSettings.config_parameters.vae.sd15.path ]:
        check_or_create_path(path)

# model selection section
model_selection, model_info = st.columns([1,2], border=True)

# draw selection menu
selected_checkpoint = model_selection.selectbox("Select a model",
                                options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
                                index=0)

# show model metadata
with st.spinner("Loading Model Metadata...."):
    model_metadata = {
        "model_checkpoint": selected_checkpoint,
        "model_path": enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint).absolute(),
        "metadata": read_safetensors_header(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint))
    }
    model_info.json(model_metadata, expanded=False)

# lora selections section
lora_selection, lora_info = st.columns([1,2], border=True)

# draw selection menu
selected_lora = lora_selection.selectbox("Select LoRA Adapter",
                                options=enumerate_models(appSettings.config_parameters.loras.sd15.path),
                                index=0)

# show lora metadata
with st.spinner("Loading Lora Metadata...."):
    lora_metadata = {
        "name": selected_lora,
        "lora_path": enumerate_models(appSettings.config_parameters.loras.sd15.path).get(selected_lora).absolute(),
        "metadata": read_safetensors_header(enumerate_models(appSettings.config_parameters.loras.sd15.path).get(selected_lora)),
    }
    lora_info.json(lora_metadata, expanded=False)

# vae section
vae_selection, vae_info = st.columns([1,2], border=True)

# draw selection menu
selected_vae = vae_selection.selectbox(label="Select SD15 VAE",
                            options=enumerate_models(appSettings.config_parameters.vae.sd15.path).keys(),
                            index=0)

# show model metadata
with st.spinner("Loading VAE Metadata...."):
    vae_metadata = {
        "vae_checkpoint": selected_vae,
        "vae_path": enumerate_models(appSettings.config_parameters.vae.sd15.path).get(selected_vae).absolute(),
        "metadata": read_safetensors_header(enumerate_models(appSettings.config_parameters.vae.sd15.path).get(selected_vae))
    }
    vae_info.json(vae_metadata, expanded=False)
