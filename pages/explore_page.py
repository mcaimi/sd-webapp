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
        from libs.stablediffusion.sd15_merge import load_checkpoint_dict, save_checkpoint, compare_checkpoint_keys, merge_checkpoints_linear
        from libs.globals.vars import schedulers
        from pathlib import PosixPath
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

# declare function tabs
sd15_info, sd15_merger = st.tabs(["SD15 Explorer", "SD15 Model Merger"])

with sd15_info:
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

with sd15_merger:
    # model selection section
    model_a, model_b = st.columns([1,1], border=True)

    # draw selection menu
    selected_checkpoint_a = model_a.selectbox("Select Source model A",
                                    options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
                                    index=0)
    selected_checkpoint_b = model_b.selectbox("Select Source model B",
                                    options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
                                    index=0)

    # analyze checkpoints...
    with st.expander(label="Merge Preflight Check", expanded=True):
        info_a, info_b, compatibility = st.columns([2,2,1], border = True)

        # read safetensors headers
        json_a = read_safetensors_header(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint_a))
        json_b = read_safetensors_header(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint_b))

        # Display Info
        info_a.json(json_a, expanded=False)
        info_b.json(json_b, expanded=False)

        # preflight check
        can_merge = compare_checkpoint_keys(json_a, json_b)
        if can_merge:
            compatibility.success('Can merge, OK!', icon="✅")
        else:
            compatibility.error('Cannot Merge Checkpoints.', icon="🚨")

    # display merge controls
    toolbox, merge_controls = st.columns([2,1], border=True)

    # tool settings
    save_output = toolbox.checkbox("Save Resulting Merge", value=True)
    if save_output:
        output_name = toolbox.text_input("Merged Checkpoint Name", value="merge_output.safetensors")

    test_resulting_checkpoint = toolbox.checkbox("Compare Checkpoints after Merge", value=False)

    # merge button
    alpha = merge_controls.text_input("Alpha", value=0.5)
    do_merge = merge_controls.button("Merge Checkpoints", type="primary", disabled=(not can_merge))

    # merge!
    if do_merge:
        with st.spinner(f"Loading Checkpoint {selected_checkpoint_a}"):
            a = load_checkpoint_dict(PosixPath(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint_a)))
        
        with st.spinner(f"Loading Checkpoint {selected_checkpoint_b}"):
            b = load_checkpoint_dict(PosixPath(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint_b)))

        with st.spinner("Merging..."):
            merged_checkpoint = merge_checkpoints_linear(a,b,alpha)

        # save resulting checkpoint
        if save_output:
            with st.spinner("Saving file..."):
                out = save_checkpoint(merged_checkpoint, "/".join((appSettings.config_parameters.checkpoints.sd15.path, output_name)))
                if out:
                    st.success(f"Checkpoint {output_name} saved.")        
