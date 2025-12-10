#!/usr/bin/env python
#
# Stable Diffusion XL Inpainting Page
#

try:
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
    from dotenv import dotenv_values
    import numpy as np
    from PIL import Image
    from io import BytesIO
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
        from libs.globals.vars import RANDOM_BIT_LENGTH
        from libs.stablediffusion.funcs import get_random_seed
        from libs.stablediffusion.sdxl import (
            gen_noise,
            load_custom_vae,
            SDXLPipelineGenerator,
        )
        from libs.globals.vars import schedulers
except Exception as e:
    print(f"Caught fatal exception: {e}")

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# load header
st.html("assets/sdxl_header.html")

# paths sanity check
appSettings.setup_paths()


# reset object cache
def reset_model_cache() -> None:
    st.cache_resource.clear()


# load model and associated resources
@st.cache_resource
def load_sdxl_model(requested_model):
    return SDXLPipelineGenerator(model_checkpoint=requested_model)


# populate sidebar
with st.sidebar:
    # select model
    selected_model = st.selectbox(
        label="Select SDXL Model",
        options=enumerate_models(
            appSettings.config_parameters.checkpoints.sdxl.path
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
                    appSettings.config_parameters.checkpoints.sdxl.path
                )
                .get(selected_model)
                .absolute(),
                "metadata": read_safetensors_header(
                    enumerate_models(
                        appSettings.config_parameters.checkpoints.sdxl.path
                    ).get(selected_model)
                ),
            }
        except:
            model_metadata = {
                "model_checkpoint": "undefined",
                "model_path": None,
                "metadata": {}
            }

    # select lora
    selected_lora = st.multiselect(
        label="Select SDXL Lora",
        options=enumerate_models(appSettings.config_parameters.loras.sdxl.path),
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
                        appSettings.config_parameters.loras.sdxl.path
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
                        enumerate_models(appSettings.config_parameters.loras.sdxl.path).get(
                            l
                        )
                    ),
                }
            except:
                lora_metadata[f"lora_{i}"] = {
                    "metadata": {},
                    "merge_strength": 0,
                    "lora_path": None,
                    "name": "undefined"
                }

    # select vae
    override_vae = st.checkbox("Override VAE", value=False, on_change=reset_model_cache)
    if override_vae:
        selected_vae = st.selectbox(
            label="Select SDXL VAE",
            options=enumerate_models(
                appSettings.config_parameters.vae.sdxl.path
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
                        appSettings.config_parameters.vae.sdxl.path
                    )
                    .get(selected_vae)
                    .absolute(),
                    "metadata": read_safetensors_header(
                        enumerate_models(appSettings.config_parameters.vae.sdxl.path).get(
                            selected_vae
                        )
                    ),
                }
            except:
                vae_metadata = {
                    "vae_checkpoint": "undefined",
                    "vae_path": None,
                    "metadata": {}
                }

# main page
st.markdown("### **Stable Diffusion XL Inpainting Page**")
st.markdown("*Inpaint images using Stable Diffusion XL models.*")

# image upload section
st.markdown("#### **Upload Image**")
uploaded_image = st.file_uploader(
    "Choose an image to inpaint",
    type=["png", "jpg", "jpeg"],
    help="Upload an image that you want to inpaint",
)

mask_image = None

# image and mask display
if uploaded_image is not None:
    # application tabs
    draw_mask_tab, upload_mask_tab = st.tabs(["Freeform Mask", "Upload Mask Rasterfile"])

    # tabs
    with draw_mask_tab:

        # image and settings columns
        image_mask_col, settings_col = st.columns([2,1])
        with settings_col:
            with st.container(border=True, horizontal_alignment="left", vertical_alignment="top"):
                # toolbox
                st.info("Draw a mask on the image: white areas will be inpainted and black areas will be preserved.")
                # Specify canvas parameters in application
                drawing_mode = st.selectbox(
                    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
                )

                stroke_width = st.slider("Stroke width: ", 1, 25, 3)
                if drawing_mode == 'point':
                    point_display_radius = st.slider("Point display radius: ", 1, 25, 3)

                c1, c2 = st.columns([1,1])
                with c1:
                    bg_color = st.color_picker("Background color hex: ", "#000")
                with c2:
                    stroke_color = st.color_picker("Stroke color hex: ", "#eee")

                # realtime update
                realtime_update = st.checkbox("Update in realtime", True)

        with image_mask_col:
            input_image = Image.open(uploaded_image).convert("RGB") if uploaded_image else None

            # Create a canvas component
            canvas_result = st_canvas(
                fill_color=bg_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=input_image,
                update_streamlit=realtime_update,
                height=input_image.height if input_image else 150,
                width=input_image.width if input_image else 320,
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                key="canvas",
            )

            if canvas_result.image_data is not None:
                # create black canvas
                black_background = Image.fromarray(np.zeros((input_image.height, input_image.width))).convert("RGBA")
                mask_image = black_background + canvas_result.image_data
                settings_col.image(mask_image)

                # download mask button
                from torchvision import transforms as tvT
                pil_img = tvT.ToPILImage()
                png_bytes = BytesIO()
                pil_img(mask_image).save(png_bytes, format="PNG")
                
                settings_col.download_button(
                    label="Download Mask Bitmap",
                    data=png_bytes.getvalue(),
                    type="primary",
                    file_name=f"sdxl_mask.png",
                    icon=":material/download:",
                )

    with upload_mask_tab:
        # Load and display the original image
        input_image = Image.open(uploaded_image).convert("RGB")
        
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.markdown("**Original Image**")
            st.image(input_image, width="stretch")
        
        with col2:
            st.markdown("**Mask Image**")
            st.info("Upload a mask image where white areas will be inpainted and black areas will be preserved.")
            
            # Upload mask
            uploaded_mask = st.file_uploader(
                "Upload mask image",
                type=["png", "jpg", "jpeg"],
                help="Upload a mask image. White areas will be inpainted, black areas will be preserved. The mask should match the image dimensions.",
            )
            
            if uploaded_mask is not None:
                mask_image = Image.open(uploaded_mask).convert("L")
                # Resize mask to match input image if needed
                if mask_image.size != input_image.size:
                    st.warning(f"Mask size ({mask_image.size}) doesn't match image size ({input_image.size}). Resizing mask...")
                    mask_image = mask_image.resize(input_image.size, Image.Resampling.LANCZOS)
                st.image(mask_image, width="stretch")
            elif mask_image is not None:
                st.info("Using mask from freeform tab...")
                st.image(mask_image, width="stretch")
            else:
                st.info("Please upload a mask image to mark areas for inpainting")
        
    if mask_image is not None:
        # prompts section
        st.markdown("#### **Prompts**")
        positive_prompt = st.text_area(
            "Positive Prompt",
            placeholder="Describe what you want to generate in the masked area",
            value="",
        )
        negative_prompt = st.text_area(
            "Negative Prompt",
            placeholder="Describe what you don't want in the image",
            value="",
        )
        
        # settings section
        with st.expander("Inpainting Settings..."):
            guidance = st.slider(
                "Guidance Scale", value=7.0, min_value=0.0, max_value=50.0, step=0.1
            )
            inference_steps = st.number_input("Inference Steps", value=20, min_value=1)
            
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
        
        # generate button
        st.markdown("---")
        gen_info_col, gen_btn_col = st.columns([2, 1])
        
        with gen_info_col:
            st.markdown(
                f"**Inpaint using model {model_metadata.get('model_checkpoint')}**."
            )
        
        with gen_btn_col:
            submit_button = st.button("Inpaint", type="primary", disabled=(mask_image is None))
        
        # generate inpainted image
        if submit_button and mask_image is not None:
            # load model
            with st.spinner(
                f"Loading Stable Diffusion XL Inpaint Model {model_metadata.get('model_checkpoint')}..."
            ):
                sd_generator = load_sdxl_model(model_metadata.get("model_path"))
                sd_generator.loadSDXLInpaintPipeline()
            
            with st.spinner(f"Merging LoRA Adapters..."):
                sd_generator.addLorasToPipeline(loras=lora_metadata)
            
            if override_vae:
                with st.spinner(f"Loading VAE {vae_metadata.get('vae_path')}..."):
                    custom_vae = load_custom_vae(vae_metadata.get("vae_path"))
                    sd_generator.inpaint_pipeline.vae = custom_vae
            
            with st.spinner(f"Moving pipeline to device: {sd_generator.accelerator}"):
                sd_generator.pipeToConfiguredDevice()
            
            # run inference
            gen_seed = seed if (seed > 0) else get_random_seed(RANDOM_BIT_LENGTH)
            
            with st.spinner("Inpainting image..."):
                output_image, output_parameters = sd_generator.forward_inpaint(
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    mask_image=mask_image,
                    scheduler_type=scheduler_type,
                    steps=inference_steps,
                    cfg=guidance,
                    seed=gen_seed,
                )
                scheduler_config = sd_generator.getSchedulerConfig()
            
            # display results
            st.success("Inpainting completed!")
            
            result_col1, result_col2 = st.columns([2,1])
            
            with result_col1:
                st.markdown("**Inpainted Result**")
                st.image(output_image, width="content")
                
                # save image
                png_file = f"sdxl_inpaint_{gen_seed}_{random_string()}.png"
                print("/".join((appSettings.config_parameters.storage.output_images, png_file)))
                with open("/".join((appSettings.config_parameters.storage.output_images, png_file)), "wb") as f:
                    from torchvision import transforms as tvT
                    pil_img = tvT.ToPILImage()
                    png_bytes = BytesIO()
                    pil_img(output_image).save(png_bytes, format="PNG")
                    
                    # write
                    f.write(png_bytes.getvalue())

            with result_col2:
                st.markdown("**Generation Parameters**")

                gen_json: str = {
                        "model_name": model_metadata.get("model_checkpoint"),
                        "loras": [
                            {
                                "lora_name": lora_metadata.get(l)["name"],
                                "merge_strength": lora_metadata.get(l)["merge_strength"]
                            } for l in lora_metadata
                        ],
                        "output_parameters": output_parameters,
                        "scheduler_config": scheduler_config,
                        "seed": gen_seed,
                    }

                json_filename: str = f"sdxl_inpaint_{gen_seed}_{random_string()}.json"
                st.json(
                    gen_json,
                    expanded=False,
                )

                # save parameters
                print("/".join((appSettings.config_parameters.storage.output_json,json_filename)))
                with open("/".join((appSettings.config_parameters.storage.output_json,json_filename)), "w") as f:
                    json.dump(gen_json, f)

        else:
            st.info("Please upload an image to begin inpainting.")

