#!/usr/bin/env python
"""
Stable Diffusion 1.5 Inpainting Page

Provides inpainting functionality via API:
- Freeform mask drawing
- Mask image upload
- Model and LoRA selection
- Custom VAE support
"""

import time
from io import BytesIO

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from libs.shared.config import get_app_config
from libs.shared.persistence import save_generation_output
from libs.shared.api_client import SDAPIClient
from libs.globals.vars import RANDOM_BIT_LENGTH
from libs.shared.utils import get_random_seed


# Configuration
MODEL_TYPE = "sd15"
config = get_app_config()
api_client = SDAPIClient()

# Load header
st.html("assets/sd15_header.html")


# === SIDEBAR ===
with st.sidebar:
    # API Health Check
    try:
        health = api_client.health_check()
        st.success(f"API: {health['status']}", icon="✅")
    except Exception as e:
        st.error(f"API unavailable: {e}", icon="🚨")

    # Model selection
    try:
        models_response = api_client.list_models(MODEL_TYPE, "checkpoints")
        model_options = {m["name"]: m for m in models_response["models"]}
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        model_options = {}

    selected_model = st.selectbox(
        label="Select SD15 Model",
        options=list(model_options.keys()),
        index=0,
    )

    # Load model metadata
    model_metadata = {"model_checkpoint": selected_model, "metadata": {}}
    if selected_model:
        with st.spinner("Loading Model Metadata..."):
            try:
                metadata_response = api_client.get_model_metadata(
                    MODEL_TYPE, "checkpoints", selected_model
                )
                model_metadata["metadata"] = metadata_response.get("metadata", {})
            except Exception as e:
                st.warning(f"Could not load metadata: {e}")

    # LoRA selection
    try:
        loras_response = api_client.list_models(MODEL_TYPE, "loras")
        lora_options = {m["name"]: m for m in loras_response["models"]}
    except Exception:
        lora_options = {}

    selected_lora = st.multiselect(
        label="Select SD15 Lora",
        options=list(lora_options.keys()),
        max_selections=5,
        default=[],
    )

    # LoRA metadata and strength sliders
    lora_metadata = {}
    lora_configs = []
    for i, lora_name in enumerate(selected_lora):
        strength = st.slider(
            label=f"{lora_name} merge strength",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            key=f"lora_strength_{i}",
        )
        lora_configs.append({"name": lora_name, "strength": strength})

        try:
            lora_meta = api_client.get_model_metadata(MODEL_TYPE, "loras", lora_name)
            lora_metadata[f"lora_{i}"] = {
                "name": lora_name,
                "merge_strength": strength,
                "metadata": lora_meta.get("metadata", {}),
            }
        except Exception:
            lora_metadata[f"lora_{i}"] = {
                "name": lora_name,
                "merge_strength": strength,
                "metadata": {},
            }

    # VAE selection
    override_vae = st.checkbox("Override VAE", value=False)
    vae_metadata = None
    selected_vae_name = None
    if override_vae:
        try:
            vae_response = api_client.list_models(MODEL_TYPE, "vaes")
            vae_options = {m["name"]: m for m in vae_response["models"]}
        except Exception:
            vae_options = {}

        selected_vae = st.selectbox(
            label="Select SD15 VAE",
            options=list(vae_options.keys()),
            index=0,
        )
        selected_vae_name = selected_vae

        with st.spinner("Loading VAE Metadata..."):
            try:
                vae_meta = api_client.get_model_metadata(MODEL_TYPE, "vaes", selected_vae)
                vae_metadata = {
                    "vae_checkpoint": selected_vae,
                    "metadata": vae_meta.get("metadata", {}),
                }
            except Exception:
                vae_metadata = {
                    "vae_checkpoint": selected_vae,
                    "metadata": {},
                }


# === MAIN PAGE ===
st.markdown("### **Stable Diffusion 1.5 Inpainting Page**")
st.markdown("*Inpaint images using Stable Diffusion 1.5 models.*")

# Fetch available schedulers from API
try:
    available_schedulers = api_client.list_schedulers()
except Exception as e:
    st.error(f"Failed to fetch schedulers: {e}")
    available_schedulers = ["DPM++ 2M"]  # Fallback default

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
    draw_mask_tab, upload_mask_tab = st.tabs(
        ["Freeform Mask", "Upload Mask Rasterfile"]
    )

    with draw_mask_tab:
        image_mask_col, settings_col = st.columns([2, 1])

        with settings_col:
            with st.container(border=True):
                st.info(
                    "Draw a mask on the image: white areas will be inpainted and black areas will be preserved."
                )

                drawing_mode = st.selectbox(
                    "Drawing tool:",
                    ("freedraw", "point", "line", "rect", "circle", "transform"),
                )
                stroke_width = st.slider("Stroke width:", 1, 25, 3)
                point_display_radius = 0
                if drawing_mode == "point":
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
                png_bytes = BytesIO()
                mask_image.save(png_bytes, format="PNG")

                settings_col.download_button(
                    label="Download Mask Bitmap",
                    data=png_bytes.getvalue(),
                    type="primary",
                    file_name="sd15_mask.png",
                    icon=":material/download:",
                )

    with upload_mask_tab:
        input_image = Image.open(uploaded_image).convert("RGB")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Original Image**")
            st.image(input_image, width="content")

        with col2:
            st.markdown("**Mask Image**")
            st.info(
                "Upload a mask image where white areas will be inpainted and black areas will be preserved."
            )

            uploaded_mask = st.file_uploader(
                "Upload mask image",
                type=["png", "jpg", "jpeg"],
                help="Upload a mask image. White areas will be inpainted, black areas preserved.",
            )

            if uploaded_mask is not None:
                mask_image = Image.open(uploaded_mask).convert("L")
                if mask_image.size != input_image.size:
                    st.warning(
                        f"Mask size ({mask_image.size}) doesn't match image size ({input_image.size}). Resizing..."
                    )
                    mask_image = mask_image.resize(
                        input_image.size, Image.Resampling.LANCZOS
                    )
                st.image(mask_image, width="content")
            elif mask_image is not None:
                st.info("Using mask from freeform tab...")
                st.image(mask_image, width="content")
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
            guidance = st.slider(
                "Guidance Scale", value=7.0, min_value=0.0, max_value=50.0, step=0.1
            )
            inference_steps = st.number_input("Inference Steps", value=20, min_value=1)

            with st.container(border=True):
                sched, seedbox = st.columns([1, 1])
                scheduler_type = sched.selectbox(
                    "Noise Scheduler", options=available_schedulers, index=0
                )
                seed = seedbox.number_input(
                    "Random Seed",
                    min_value=-1,
                    max_value=None,
                    step=1,
                    help="Generation Seed. -1 Means Random Seed",
                )

        # Generate button
        st.markdown("---")
        gen_info_col, gen_btn_col = st.columns([2, 1])

        with gen_info_col:
            st.markdown(
                f"**Inpaint using model {model_metadata.get('model_checkpoint')}**."
            )

        with gen_btn_col:
            submit_button = st.button(
                "Inpaint", type="primary", disabled=(mask_image is None)
            )

        if submit_button and mask_image is not None:
            gen_seed = seed if seed > 0 else get_random_seed(RANDOM_BIT_LENGTH)

            # Prepare LoRA configs for API
            lora_configs = []
            if lora_metadata:
                for lora_key, lora_info in lora_metadata.items():
                    lora_configs.append({
                        "lora_name": lora_info.get("name"),
                        "merge_strength": lora_info.get("merge_strength"),
                    })

            try:
                with st.spinner("Submitting inpaint job to API..."):
                    inpaint_progress = st.progress(0, text="Initializing...")

                    # Convert mask_image to grayscale if it's RGBA
                    if mask_image.mode == "RGBA":
                        mask_image = mask_image.convert("L")

                    # Submit inpaint job
                    job_id, _ = api_client.inpaint_image(
                        model_type=MODEL_TYPE,
                        positive_prompt=positive_prompt,
                        image=input_image,
                        mask_image=mask_image,
                        negative_prompt=negative_prompt,
                        model_checkpoint=selected_model,
                        steps=inference_steps,
                        cfg_scale=guidance,
                        seed=gen_seed,
                        scheduler=scheduler_type,
                        loras=lora_configs if lora_configs else None,
                        custom_vae=selected_vae_name if override_vae else None,
                    )

                    # Poll for completion
                    while True:
                        status = api_client.get_job_status(job_id)

                        if status["status"] == "running":
                            progress = status.get("progress", 0.0)
                            inpaint_progress.progress(progress, text=f"Inpainting... {int(progress*100)}%")
                            time.sleep(0.5)
                        elif status["status"] == "completed":
                            inpaint_progress.progress(1.0, text="Complete!")
                            inpaint_progress.empty()

                            result = status.get("result", {})
                            output_image = api_client.decode_image_from_result(result)
                            output_parameters = result.get("parameters", {})

                            st.success("Inpainting completed!")

                            result_col1, result_col2 = st.columns(2)

                            with result_col1:
                                st.markdown("**Inpainted Result**")
                                st.image(output_image, width="content")

                            with result_col2:
                                st.markdown("**Generation Parameters**")
                                gen_json = {
                                    "model_name": model_metadata.get("model_checkpoint"),
                                    "loras": lora_configs,
                                    "output_parameters": output_parameters,
                                    "seed": gen_seed,
                                }
                                st.json(gen_json, expanded=False)

                            # Save outputs
                            save_generation_output(
                                output_image=output_image,
                                output_parameters=output_parameters,
                                model_metadata=model_metadata,
                                lora_metadata=lora_metadata,
                                scheduler_config=None,
                                seed=gen_seed,
                                model_type=MODEL_TYPE,
                                prefix="inpaint",
                            )
                            break
                        elif status["status"] == "failed":
                            error = status.get("error", "Unknown error")
                            st.error(f"❌ Inpainting failed: {error}")
                            inpaint_progress.empty()
                            break
                        else:
                            time.sleep(0.5)

            except Exception as e:
                st.error(f"❌ API request failed: {e}")
    else:
        st.info("Please upload an image to begin inpainting.")
