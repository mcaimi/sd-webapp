#!/usr/bin/env python
# 
# Stable Diffusion 1.5 Generation Page
#

try:
    import streamlit as st
    from dotenv import dotenv_values

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.utils import enumerate_models, read_safetensors_header, check_or_create_path
        from libs.stablediffusion.sd15 import gen_noise
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
st.html("assets/sd15_header.html")

# paths sanity check
with st.spinner("Checking Paths"):
    for path in [ appSettings.config_parameters.checkpoints.sd15.path, appSettings.config_parameters.loras.sd15.path,
                appSettings.config_parameters.checkpoints.sdxl.path, appSettings.config_parameters.loras.sdxl.path ]:
        check_or_create_path(path)

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
    selected_model = st.selectbox(label="Select SD15 Model",
                                options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
                                index=0,
                                on_change=reset_model_cache)

    # show model metadata
    with st.spinner("Loading Model Metadata...."):
        model_metadata = {
            "model_checkpoint": selected_model,
            "model_path": enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_model).absolute(),
            "metadata": read_safetensors_header(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_model))
        }
        st.json(model_metadata, expanded=False)

    # select lora
    selected_lora = st.multiselect(label="Select SD15 Lora",
                                options=enumerate_models(appSettings.config_parameters.loras.sd15.path),
                                max_selections=5, default=[], on_change=reset_model_cache)

    # show lora metadata
    lora_metadata = {}
    for i, l in enumerate(selected_lora):
        lora_metadata[f"lora_{i}"] = {
            "name": l,
            "lora_path": enumerate_models(appSettings.config_parameters.loras.sd15.path).get(l).absolute(),
            "metadata": read_safetensors_header(enumerate_models(appSettings.config_parameters.loras.sd15.path).get(l)),
        }
    st.json(lora_metadata, expanded=False)

# main page
st.markdown("**Stable Diffusion Generation Page, v1.5**")

# left -> settings, right -> generated image
# split page into two columns
settings, image_output = st.columns([3,1])

# input section
with settings.container():
    # align text prompts on the left
    positive_prompt = st.text_area("Positive Prompt", placeholder="Write here what you want in the image")
    negative_prompt = st.text_area("Negative Prompt", placeholder="Write here what you don't want in the image")
    submit_button = st.button("Generate", type="primary")

    # settings section
    with st.expander("Generation Settings..."):
        guidance = st.slider("Guidance Scale", value=7.0, min_value=0.0, max_value=50.0, step=0.1)
        with st.container(border=True):
            w, h = st.columns([1,1])
            width = w.number_input("Image Width", value=512)
            height = h.number_input("Image Height", value=768)
        inference_steps = st.number_input("Inference Steps", value=20)
        with st.container(border=True):
            sched, seedbox = st.columns([1,1])
            scheduler_type = sched.selectbox("Noise Scheduler", options=schedulers, index=0)
            seed = seedbox.number_input("Random Seed", min_value=-1, max_value=None, step=1)
        
    # generate new image
    if submit_button:
        # load model..
        with st.spinner(f"Loading Stable Diffusion Model {model_metadata.get('model_checkpoint')}..."):
            sd_generator = load_sd15_model(model_metadata.get("model_path"))
            sd_generator.loadSDPipeline()

        with st.spinner(f"Merging LoRA Adapters..."):
            sd_generator.addLorasToPipeline(loras=[lora_metadata[k].get("lora_path") for k in lora_metadata.keys()])
        
        with st.spinner(f"Moving pipeline to device: {sd_generator.accelerator}"):
            sd_generator.pipeToConfiguredDevice()
        # run inference
        with st.spinner(f"Generating image ..."):
            output_image, output_parameters = sd_generator.forward(positive_prompt=positive_prompt,
                                                                        negative_prompt=negative_prompt,
                                                                        steps=inference_steps,
                                                                        scheduler_type=scheduler_type,
                                                                        width=width, height=height,
                                                                        seed=seed,
                                                                        cfg=guidance)
            scheduler_config = sd_generator.getSchedulerConfig()

# output section
with image_output.container():
    noise_image, noise_parameters = gen_noise(width=width, height=height)
    # display image
    try:
        st.markdown("**Generated Image**")
        image_bytes = st.image(output_image, output_format="PNG")
        
        # save image
        from io import BytesIO
        from torchvision import transforms as tvT
        
        # ndarray -> image
        pil_img = tvT.ToPILImage()
        # image -> png byte stream
        png_bytes = BytesIO()
        pil_img(output_image).save(png_bytes, format="PNG")

        st.download_button(label="Download Image", 
                        data=png_bytes.getvalue(), 
                        type="primary",
                        file_name=f"sd15_{seed}.png",
                        icon=":material/download:")

        # display recap on the right
        st.markdown("**Generation Parameters**")
        image_metadata = st.json({
            "model_name": model_metadata.get("model_checkpoint"),
            "lora_names": [lora_metadata.get(l)["name"] for l in lora_metadata],
            "output_parameters": output_parameters,
            "scheduler_config": scheduler_config
        })
    except Exception as e:
        st.markdown(f"**Noise Placeholder** {e}")
        image_bytes = st.image(noise_image)
        # display recap on the right
        st.markdown("**Noise Parameters**")
        image_metadata = st.json({
            "model_name": model_metadata.get("model_checkpoint"),
            "lora_names": [lora_metadata.get(l)["name"] for l in lora_metadata],
            "output_parameters": noise_parameters
        }, expanded=False)