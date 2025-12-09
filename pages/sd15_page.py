#!/usr/bin/env python
#
# Stable Diffusion 1.5 Generation Page
#

try:
    import streamlit as st
    from dotenv import dotenv_values
    import json

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from libs.shared.settings import Properties
        from libs.shared.utils import (
            enumerate_models,
            read_safetensors_header,
            check_or_create_path,
            random_string,
            get_gpu
        )
        from libs.stablediffusion.sd15 import (
            load_custom_vae,
            SD15PipelineGenerator,
        )
        from libs.globals.vars import schedulers, RANDOM_BIT_LENGTH
        from libs.stablediffusion.funcs import get_random_seed
except Exception as e:
    print(f"Caught fatal exception: {e}")

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# load header
st.html("assets/sd15_header.html")

# paths sanity check
appSettings.setup_paths()


# reset object cache
def reset_model_cache() -> None:
    st.cache_resource.clear()


# load model and associated resources
@st.cache_resource
def load_sd15_model(requested_model):
    return SD15PipelineGenerator(model_checkpoint=requested_model)


# populate sidebar
with st.sidebar:
    # select model
    selected_model = st.selectbox(
        label="Select SD15 Model",
        options=enumerate_models(
            appSettings.config_parameters.checkpoints.sd15.path
        ).keys(),
        index=0,
        on_change=reset_model_cache,
    )

    # show model metadata
    with st.spinner("Loading Model Metadata...."):
        try:
            model_metadata = {
                "model_checkpoint": selected_model,
                "model_path": enumerate_models(
                    appSettings.config_parameters.checkpoints.sd15.path
                )
                .get(selected_model)
                .absolute(),
                "metadata": read_safetensors_header(
                    enumerate_models(
                        appSettings.config_parameters.checkpoints.sd15.path
                    ).get(selected_model)
                ),
            }
        except:
            model_metadata = {
                "model_checkpoint": "no model available",
                "model_path": None,
                "metadata": {},
            }

    # select lora
    selected_lora = st.multiselect(
        label="Select SD15 Lora",
        options=enumerate_models(appSettings.config_parameters.loras.sd15.path),
        max_selections=5,
        default=[],
        on_change=reset_model_cache,
    )

    # show lora metadata
    with st.spinner("Loading Lora Metadata...."):
        lora_metadata = {}
        for i, l in enumerate(selected_lora):
            try:
                lora_metadata[f"lora_{i}"] = {
                    "name": l,
                    "lora_path": enumerate_models(
                        appSettings.config_parameters.loras.sd15.path
                    )
                    .get(l)
                    .absolute(),
                    "merge_strength": st.slider(
                        label=f"{l} merge strength",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        step=0.1,
                    ),
                    "metadata": read_safetensors_header(
                        enumerate_models(appSettings.config_parameters.loras.sd15.path).get(
                            l
                        )
                    ),
                }
            except:
                lora_metadata[f"lora_{i}"] = {
                    "name": "not available",
                    "lora_path": None,
                    "merge_strength": 0,
                    "metadata": {},
                }

    # select vae
    override_vae = st.checkbox("Override VAE", value=False, on_change=reset_model_cache)
    if override_vae:
        selected_vae = st.selectbox(
            label="Select SD15 VAE",
            options=enumerate_models(
                appSettings.config_parameters.vae.sd15.path
            ).keys(),
            index=0,
            on_change=reset_model_cache,
        )

        # show model metadata
        with st.spinner("Loading VAE Metadata...."):
            try:
                vae_metadata = {
                    "vae_checkpoint": selected_vae,
                    "vae_path": enumerate_models(
                        appSettings.config_parameters.vae.sd15.path
                    )
                    .get(selected_vae)
                    .absolute(),
                    "metadata": read_safetensors_header(
                        enumerate_models(appSettings.config_parameters.vae.sd15.path).get(
                            selected_vae
                        )
                    ),
                }
            except:
                vae_metadata = {
                    "vae_checkpoint": "no available vae",
                    "vae_path": None,
                    "metadata": {},
                }

    st.markdown(f"**Device: {get_gpu()}**")

# main page
st.markdown("### **Stable Diffusion Generation Page, v1.5**")
st.markdown("*Generate images using Stable Diffusion 1.5 models.*")

# common features
# image generate
# align text prompts on the left
positive_prompt = st.text_area(
    "Positive Prompt", placeholder="Write here what you want in the image"
)
negative_prompt = st.text_area(
    "Negative Prompt", placeholder="Write here what you don't want in the image"
)

# settings section
with st.expander("Generation Settings..."):
    guidance = st.slider(
        "Guidance Scale", value=7.0, min_value=0.0, max_value=50.0, step=0.1
    )
    with st.container(border=True):
        w, h = st.columns([1, 1])
        width = w.number_input("Image Width", value=512)
        height = h.number_input("Image Height", value=768)

    inference_steps = st.number_input("Inference Steps", value=20)

    with st.container(border=True):
        batch_size = st.number_input("Batch Size", min_value=1, value=1)

    with st.container(border=True):
        sched, seedbox = st.columns([1, 1])
        scheduler_type = sched.selectbox("Noise Scheduler", options=schedulers, index=0)
        seed = seedbox.number_input(
            "Random Seed",
            min_value=-1,
            max_value=None,
            step=1,
            help="Generation Seed. -1 Means Random Seed",
        )

# application tabs
image_gen_tab, model_comparison_tab = st.tabs(["Image Generation", "Model Comparison"])

with image_gen_tab:
    gen_info_col, gen_btn_col = st.columns([2, 1])

    with gen_info_col:
        st.markdown(
            f"**Generate Images using model {model_metadata.get('model_checkpoint')}**."
        )

    with gen_btn_col:
        submit_button = st.button("Generate", type="primary")

    # generate new image
    if submit_button:
        # load model..
        with st.spinner(
            f"Loading Stable Diffusion Model {model_metadata.get('model_checkpoint')}..."
        ):
            sd_generator = load_sd15_model(model_metadata.get("model_path"))
            sd_generator.loadSDPipeline()

        with st.spinner(f"Merging LoRA Adapters..."):
            sd_generator.addLorasToPipeline(loras=lora_metadata)

        if override_vae:
            with st.spinner(f"Loading VAE {vae_metadata.get('vae_path')}..."):
                sd_generator.vae = load_custom_vae(vae_metadata.get("vae_path"))

        with st.spinner(f"Moving pipeline to device: {sd_generator.accelerator}"):
            sd_generator.pipeToConfiguredDevice()

        # run inference
        generated_pixmaps = []
        if (batch_size > 1) and (seed > 0):
            st.warning(
                f"Seed {seed} is constant and batch size is: {batch_size}: Images will be all the same..."
            )

        for i in range(batch_size):
            # get seed
            gen_seed = seed if (seed > 0) else get_random_seed(RANDOM_BIT_LENGTH)
            # generate
            with st.spinner(f"Generating image ... {i + 1}/{batch_size}"):
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
            # append image to list
            generated_pixmaps.append(
                (output_image, output_parameters, scheduler_config, gen_seed)
            )

with model_comparison_tab:
    comp_info_col, comp_btn_col = st.columns([2, 1])

    with comp_info_col:
        st.markdown(
            f"**Generate images with consistent parameters across selected models.**"
        )
        # multiple models selection
        target_models = st.multiselect(
            label="Select target models",
            max_selections=6,
            help="Generate image using the same settings across different models to compare checkpoints",
            options=enumerate_models(
                appSettings.config_parameters.checkpoints.sd15.path
            ),
            default=[],
        )

    with comp_btn_col:
        gen_button = st.button(
            "Generate over Models", type="primary", disabled=(len(target_models) == 0)
        )

    if gen_button:
        # images holder
        generated_pixmaps = []
        # generate seed
        gen_seed = seed if (seed > 0) else get_random_seed(RANDOM_BIT_LENGTH)

        # iterate over models
        for i, model in enumerate(target_models):
            # set model name
            model_metadata["model_checkpoint"] = model
            ckpt = enumerate_models(
                appSettings.config_parameters.checkpoints.sd15.path
            ).get(model)
            pipeline = load_sd15_model(ckpt.absolute())
            pipeline.loadSDPipeline()

            with st.spinner(f"Merging LoRA Adapters..."):
                pipeline.addLorasToPipeline(loras=lora_metadata)

            if override_vae:
                with st.spinner(f"Loading VAE {vae_metadata.get('vae_path')}..."):
                    pipeline.vae = load_custom_vae(vae_metadata.get("vae_path"))

            with st.spinner(f"Moving pipeline to device: {pipeline.accelerator}"):
                pipeline.pipeToConfiguredDevice()

            with st.spinner(f"Generating image ... {i + 1} with {model}"):
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
            # append image to list
            generated_pixmaps.append(
                (output_image, output_parameters, scheduler_config, gen_seed)
            )
            reset_model_cache()

# output section
try:
    if len(generated_pixmaps) > 0:
        st.success(
            f"Generation success! Inference produced {len(generated_pixmaps)} images:"
        )
        with st.container():
            for element in generated_pixmaps:
                # get image data
                output_image, output_parameters, scheduler_config, gen_seed = element

                # display results
                img_out, parms_out = st.columns(
                    [2, 1], border=True, vertical_alignment="center"
                )

                # display image with streamlit
                with img_out:
                    image_bytes = st.image(output_image, output_format="PNG")

                # display and save inference parameters
                gen_json: dict = {
                            "model_name": model_metadata.get("model_checkpoint"),
                            "lora_names": [
                                lora_metadata.get(l)["name"] for l in lora_metadata
                            ],
                            "output_parameters": output_parameters,
                            "scheduler_config": scheduler_config,
                        }
        
                json_filename = f"sd15_{gen_seed}_{random_string()}.json"
                with parms_out:
                    st.json(gen_json,
                        expanded=False,
                    )
                
                print("/".join((appSettings.config_parameters.storage.output_json,json_filename)))
                with open("/".join((appSettings.config_parameters.storage.output_json,json_filename)), "w") as f:
                    json.dump(gen_json, f)

                # save image bytes
                png_file = f"sd15_{gen_seed}_{random_string()}.png"
                print("/".join((appSettings.config_parameters.storage.output_images, png_file)))
                with open("/".join((appSettings.config_parameters.storage.output_images, png_file)), "wb") as f:
                    # save image
                    from io import BytesIO
                    from torchvision import transforms as tvT

                    # ndarray -> image
                    pil_img = tvT.ToPILImage()
                    # image -> png byte stream
                    png_bytes = BytesIO()
                    pil_img(output_image).save(png_bytes, format="PNG")

                    # write
                    f.write(png_bytes.getvalue())
                    
except NameError as e:
    st.info("Select generation method and perform inference.")
