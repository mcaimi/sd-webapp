#!/usr/bin/env python
"""
Stable Diffusion XL Inpainting Page

Provides inpainting functionality with:
- Freeform mask drawing
- Mask image upload
- Model and LoRA selection
- Custom VAE support
"""

import json
from io import BytesIO

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from libs.shared.config import get_app_config
from libs.shared.utils import enumerate_models, read_safetensors_header, random_string, get_gpu
from libs.shared.ui_components import (
    create_prompt_inputs,
    save_generation_output,
    reset_streamlit_cache,
)
from libs.stablediffusion.sdxl import SDXLPipelineGenerator, load_custom_vae
from libs.globals.vars import schedulers, RANDOM_BIT_LENGTH
from libs.stablediffusion.funcs import get_random_seed


# Configuration
MODEL_TYPE = "sdxl"
config = get_app_config()

# Load header
st.html("assets/sdxl_header.html")


def reset_model_cache() -> None:
    """Clear cached models when selection changes."""
    reset_streamlit_cache()


@st.cache_resource
def load_sdxl_model(model_path):
    """Load and cache an SDXL model."""
    return SDXLPipelineGenerator(model_checkpoint=model_path)


# === SIDEBAR ===
with st.sidebar:
    # Model selection
    model_options = enumerate_models(config.checkpoints_sdxl_path)
    selected_model = st.selectbox(
        label="Select SDXL Model",
        options=list(model_options.keys()),
        index=0,
        on_change=reset_model_cache,
    )

    # Load model metadata
    with st.spinner("Loading Model Metadata..."):
        try:
            model_metadata = {
                "model_checkpoint": selected_model,
                "model_path": model_options.get(selected_model).absolute() if selected_model else None,
                "metadata": read_safetensors_header(model_options.get(selected_model)) if selected_model else {},
            }
        except Exception:
            model_metadata = {
                "model_checkpoint": "undefined",
                "model_path": None,
                "metadata": {},
            }

    # LoRA selection
    lora_options = enumerate_models(config.loras_sdxl_path)
    selected_lora = st.multiselect(
        label="Select SDXL Lora",
        options=list(lora_options.keys()),
        max_selections=5,
        default=[],
        on_change=reset_model_cache,
    )

    # LoRA metadata and strength sliders
    with st.spinner("Loading Lora Metadata..."):
        lora_metadata = {}
        for i, lora_name in enumerate(selected_lora):
            try:
                lora_path = lora_options.get(lora_name)
                lora_metadata[f"lora_{i}"] = {
                    "name": lora_name,
                    "lora_path": lora_path.absolute() if lora_path else None,
                    "merge_strength": st.slider(
                        label=f"{lora_name} merge strength",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        step=0.1,
                        key=f"lora_strength_{i}",
                    ),
                    "metadata": read_safetensors_header(lora_path) if lora_path else {},
                }
            except Exception:
                lora_metadata[f"lora_{i}"] = {
                    "name": "undefined",
                    "lora_path": None,
                    "merge_strength": 0,
                    "metadata": {},
                }

    # VAE selection
    override_vae = st.checkbox("Override VAE", value=False, on_change=reset_model_cache)
    vae_metadata = None
    if override_vae:
        vae_options = enumerate_models(config.vae_sdxl_path)
        selected_vae = st.selectbox(
            label="Select SDXL VAE",
            options=list(vae_options.keys()),
            index=0,
            on_change=reset_model_cache,
        )

        with st.spinner("Loading VAE Metadata..."):
            try:
                vae_path = vae_options.get(selected_vae)
                vae_metadata = {
                    "vae_checkpoint": selected_vae,
                    "vae_path": vae_path.absolute() if vae_path else None,
                    "metadata": read_safetensors_header(vae_path) if vae_path else {},
                }
            except Exception:
                vae_metadata = {
                    "vae_checkpoint": "undefined",
                    "vae_path": None,
                    "metadata": {},
                }


# === MAIN PAGE ===
st.markdown("### **Stable Diffusion XL Inpainting Page**")
st.markdown("*Inpaint images using Stable Diffusion XL models.*")

# Image upload section
st.markdown("#### **Upload Image**")
uploaded_image = st.file_uploader(
    "Choose an image to inpaint",
    type=["png", "jpg", "jpeg"],
    help="Upload an image that you want to inpaint",
)

mask_image = None

if uploaded_image is not None:
    # Tabs for mask creation methods
    draw_mask_tab, upload_mask_tab = st.tabs(["Freeform Mask", "Upload Mask Rasterfile"])

    with draw_mask_tab:
        image_mask_col, settings_col = st.columns([2, 1])

        with settings_col:
            with st.container(border=True):
                st.info("Draw a mask on the image: white areas will be inpainted and black areas will be preserved.")

                drawing_mode = st.selectbox(
                    "Drawing tool:",
                    ("freedraw", "point", "line", "rect", "circle", "transform"),
                )
                stroke_width = st.slider("Stroke width:", 1, 25, 3)
                point_display_radius = 0
                if drawing_mode == 'point':
                    point_display_radius = st.slider("Point display radius:", 1, 25, 3)

                c1, c2 = st.columns([1, 1])
                with c1:
                    bg_color = st.color_picker("Background color:", "#000")
                with c2:
                    stroke_color = st.color_picker("Stroke color:", "#eee")

                realtime_update = st.checkbox("Update in realtime", True)

        with image_mask_col:
            input_image = Image.open(uploaded_image).convert("RGB")

            canvas_result = st_canvas(
                fill_color=bg_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=input_image,
                update_streamlit=realtime_update,
                height=input_image.height,
                width=input_image.width,
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius,
                key="canvas",
            )

            if canvas_result.image_data is not None:
                black_background = Image.fromarray(
                    np.zeros((input_image.height, input_image.width))
                ).convert("RGBA")
                mask_image = black_background + canvas_result.image_data
                settings_col.image(mask_image)

                # Download mask button
                from torchvision import transforms as tvT
                pil_img = tvT.ToPILImage()
                png_bytes = BytesIO()
                pil_img(mask_image).save(png_bytes, format="PNG")

                settings_col.download_button(
                    label="Download Mask Bitmap",
                    data=png_bytes.getvalue(),
                    type="primary",
                    file_name="sdxl_mask.png",
                    icon=":material/download:",
                )

    with upload_mask_tab:
        input_image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Original Image**")
            st.image(input_image, width='content')

        with col2:
            st.markdown("**Mask Image**")
            st.info("Upload a mask image where white areas will be inpainted and black areas will be preserved.")

            uploaded_mask = st.file_uploader(
                "Upload mask image",
                type=["png", "jpg", "jpeg"],
                help="Upload a mask image. White areas will be inpainted, black areas preserved.",
            )

            if uploaded_mask is not None:
                mask_image = Image.open(uploaded_mask).convert("L")
                if mask_image.size != input_image.size:
                    st.warning(f"Mask size ({mask_image.size}) doesn't match image size ({input_image.size}). Resizing...")
                    mask_image = mask_image.resize(input_image.size, Image.Resampling.LANCZOS)
                st.image(mask_image, width='content')
            elif mask_image is not None:
                st.info("Using mask from freeform tab...")
                st.image(mask_image, width='content')
            else:
                st.info("Please upload a mask image to mark areas for inpainting")

    if mask_image is not None:
        # Prompts section
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

        # Settings section
        with st.expander("Inpainting Settings..."):
            guidance = st.slider("Guidance Scale", value=7.0, min_value=0.0, max_value=50.0, step=0.1)
            inference_steps = st.number_input("Inference Steps", value=20, min_value=1)

            with st.container(border=True):
                sched, seedbox = st.columns([1, 1])
                scheduler_type = sched.selectbox("Noise Scheduler", options=schedulers, index=0)
                seed = seedbox.number_input(
                    "Random Seed", min_value=-1, max_value=None, step=1,
                    help="Generation Seed. -1 Means Random Seed",
                )

        # Generate button
        st.markdown("---")
        gen_info_col, gen_btn_col = st.columns([2, 1])

        with gen_info_col:
            st.markdown(f"**Inpaint using model {model_metadata.get('model_checkpoint')}**.")

        with gen_btn_col:
            submit_button = st.button("Inpaint", type="primary", disabled=(mask_image is None))

        if submit_button and mask_image is not None:
            with st.spinner(f"Loading Stable Diffusion XL Inpaint Model {model_metadata.get('model_checkpoint')}..."):
                sd_generator = load_sdxl_model(model_metadata.get("model_path"))
                sd_generator.loadSDXLInpaintPipeline()

            with st.spinner("Merging LoRA Adapters..."):
                sd_generator.addLorasToPipeline(loras=lora_metadata)

            if override_vae and vae_metadata:
                with st.spinner(f"Loading VAE {vae_metadata.get('vae_path')}..."):
                    sd_generator.inpaint_pipeline.vae = load_custom_vae(vae_metadata.get("vae_path"))

            with st.spinner(f"Moving pipeline to device: {sd_generator.accelerator}"):
                sd_generator.pipeToConfiguredDevice()

            gen_seed = seed if seed > 0 else get_random_seed(RANDOM_BIT_LENGTH)
            inference_uuid = random_string()

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

            st.success("Inpainting completed!")

            result_col1, result_col2 = st.columns([2, 1])

            with result_col1:
                st.markdown("**Inpainted Result**")
                st.image(output_image, width='content')

            with result_col2:
                st.markdown("**Generation Parameters**")
                gen_json = {
                    "model_name": model_metadata.get("model_checkpoint"),
                    "loras": [
                        {
                            "lora_name": lora_metadata.get(l, {}).get("name"),
                            "merge_strength": lora_metadata.get(l, {}).get("merge_strength"),
                        }
                        for l in lora_metadata
                    ],
                    "output_parameters": output_parameters,
                    "scheduler_config": scheduler_config,
                    "seed": gen_seed,
                }
                st.json(gen_json, expanded=False)

            # Save outputs
            save_generation_output(
                output_image=output_image,
                output_parameters=output_parameters,
                model_metadata=model_metadata,
                lora_metadata=lora_metadata,
                scheduler_config=scheduler_config,
                seed=gen_seed,
                model_type=MODEL_TYPE,
                prefix="inpaint",
            )
else:
    st.info("Please upload an image to begin inpainting.")
