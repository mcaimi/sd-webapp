#!/usr/bin/env python
"""
Stable Diffusion XL Merge Features Page

Provides model merging tools via API backend:
- Checkpoint structure analysis (uses API for listing)
- Model merging via API
- Batch merging via API
- Recipe merging via API
"""

import time
from pathlib import Path

import streamlit as st

from libs.shared.config import get_app_config
from libs.shared.api_client import SDAPIClient


# Configuration
MODEL_TYPE = "sdxl"
config = get_app_config()
api_client = SDAPIClient()

# Load header
st.html("assets/explore_header.html")

# Ensure paths exist
config.setup_paths()

# === TABS ===
sdxl_info, sdxl_merger, xl_batch_merger, xl_recipe_builder = st.tabs(
    ["Checkpoint Explorer", "Advanced Merger", "Batch Processing", "Recipe Builder"]
)


# === CHECKPOINT EXPLORER ===
with sdxl_info:
    st.markdown("### 📊 Checkpoint Explorer")
    st.markdown("*Browse and inspect model metadata via API*")

    # Check API health
    try:
        health = api_client.health_check()
        st.success(f"API: {health['status']}", icon="✅")
    except Exception as e:
        st.error(f"API unavailable: {e}", icon="🚨")

    model_selection, model_info = st.columns([1, 2], border=True)

    # Fetch models from API
    try:
        models_response = api_client.list_models(MODEL_TYPE, "checkpoints")
        model_options = {m["name"]: m for m in models_response["models"]}
    except Exception as e:
        st.error(f"Failed to load models from API: {e}")
        model_options = {}

    selected_checkpoint = model_selection.selectbox(
        "Select a model",
        options=list(model_options.keys()),
        index=0,
    )

    # Fetch metadata from API
    if selected_checkpoint:
        with st.spinner("Loading Model Metadata..."):
            try:
                metadata_response = api_client.get_model_metadata(
                    MODEL_TYPE, "checkpoints", selected_checkpoint
                )
                model_metadata = {
                    "model_checkpoint": selected_checkpoint,
                    "model_path": metadata_response.get("path"),
                    "metadata": metadata_response.get("metadata", {}),
                }
                model_info.json(model_metadata, expanded=False)
            except Exception as e:
                model_info.json({"exception": str(e)}, expanded=False)

    # LoRA section
    lora_selection, lora_info = st.columns([1, 2], border=True)

    try:
        loras_response = api_client.list_models(MODEL_TYPE, "loras")
        lora_options = {m["name"]: m for m in loras_response["models"]}
    except Exception:
        lora_options = {}

    selected_lora = lora_selection.selectbox(
        "Select LoRA Adapter",
        options=list(lora_options.keys()),
        index=0,
    )

    if selected_lora:
        with st.spinner("Loading Lora Metadata..."):
            try:
                lora_meta = api_client.get_model_metadata(MODEL_TYPE, "loras", selected_lora)
                lora_metadata = {
                    "name": selected_lora,
                    "lora_path": lora_meta.get("path"),
                    "metadata": lora_meta.get("metadata", {}),
                }
                lora_info.json(lora_metadata, expanded=False)
            except Exception as e:
                lora_info.json({"exception": str(e)}, expanded=False)

    # VAE section
    vae_selection, vae_info = st.columns([1, 2], border=True)

    try:
        vae_response = api_client.list_models(MODEL_TYPE, "vaes")
        vae_options = {m["name"]: m for m in vae_response["models"]}
    except Exception:
        vae_options = {}

    selected_vae = vae_selection.selectbox(
        label="Select SDXL VAE",
        options=list(vae_options.keys()),
        index=0,
    )

    if selected_vae:
        with st.spinner("Loading VAE Metadata..."):
            try:
                vae_meta = api_client.get_model_metadata(MODEL_TYPE, "vaes", selected_vae)
                vae_metadata = {
                    "vae_checkpoint": selected_vae,
                    "vae_path": vae_meta.get("path"),
                    "metadata": vae_meta.get("metadata", {}),
                }
                vae_info.json(vae_metadata, expanded=False)
            except Exception as e:
                vae_info.json({"exception": str(e)}, expanded=False)


# === ADVANCED MERGER ===
with sdxl_merger:
    st.markdown("### 🔀 XL Model Merger")
    st.markdown("*Merge two SDXL models using different methods (via API)*")

    try:
        models_response = api_client.list_models(MODEL_TYPE, "checkpoints")
        model_options = {m["name"]: m for m in models_response["models"]}
    except Exception as e:
        st.error(f"Failed to load models from API: {e}")
        model_options = {}

    model_a, model_b = st.columns([1, 1], border=True)

    selected_checkpoint_a = model_a.selectbox(
        "🎯 Base Model (A)",
        options=list(model_options.keys()),
        index=0,
        help="The primary model that will be used as the foundation",
    )
    selected_checkpoint_b = model_b.selectbox(
        "🎨 Target Model (B)",
        options=list(model_options.keys()),
        index=0,
        help="The model whose features will be merged into the base",
    )

    # Merge configuration
    st.markdown("### ⚙️ Choose Merge Parameters")

    merge_controls, ops_controls = st.columns([1, 1])

    with merge_controls:
        merge_method = st.selectbox(
            "🧮 Merge Algorithm",
            options=["linear", "slerp", "additive", "subtract"],
            format_func=lambda x: {
                "linear": "📊 Linear (Classic blend)",
                "slerp": "🌊 SLERP (Spherical interpolation)",
                "additive": "➕ Additive (Add features)",
                "subtract": "➖ Subtract (Remove features)",
            }[x],
        )

        alpha = st.slider("💪 Mix Strength", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        preserve_metadata = st.checkbox("📋 Preserve Metadata", value=True)

    with ops_controls:
        do_advanced_merge = st.button(
            f"🔬 Advanced {merge_method.upper()} Merge",
            type="primary",       
        )

        st.markdown("**💾 Output Settings**")
        save_output = st.checkbox("Save Result", value=True)
        if save_output:
            output_name = st.text_input(
                   "Output Filename",
                value=f"merged_{merge_method}_{selected_checkpoint_a.split('.')[0]}.safetensors",
           )

    merge_column, output_column = st.columns([2, 2])

    if do_advanced_merge:
        try:
            with st.spinner(f"🔄 Executing {merge_method.upper()} merge..."):
                merge_progress = st.progress(0, text="Merging...")

                try:
                    # Submit merge job to API
                    job_id, _ = api_client.merge_models(
                        model_type=MODEL_TYPE,
                        base_model=selected_checkpoint_a,
                        target_model=selected_checkpoint_b,
                        method=merge_method,
                        alpha=alpha,
                        output_name=output_name,
                        preserve_metadata=preserve_metadata,
                    )

                    # Poll for completion with progress updates
                    while True:
                        status = api_client.get_job_status(job_id)

                        if status["status"] == "running":
                            progress = status.get("progress", 0.0)
                            merge_progress.progress(progress, text=f"Merging... {int(progress*100)}%")
                            time.sleep(0.5)
                        elif status["status"] == "completed":
                            merge_progress.progress(1.0, text="Complete!")
                            merge_column.success("✅ Merge completed successfully!")
                            merge_column.info(f"📁 Saved to: {output_name}")

                            with output_column.expander("📊 Merge Details", expanded=True):
                                st.json({
                                    "method": merge_method,
                                    "alpha": alpha,
                                    "base_model": selected_checkpoint_a,
                                    "target_model": selected_checkpoint_b,
                                    "output_file": output_name,
                                })

                            merge_progress.empty()
                            break
                        elif status["status"] == "failed":
                            error = status.get("error", "Unknown error")
                            st.error(f"❌ Merge failed: {error}")
                            merge_progress.empty()
                            break
                        else:
                            time.sleep(0.5)
                except Exception as e:
                    st.error(f"❌ Job Polling failed: {e}")
        except Exception as e:
            st.error(f"❌ Merge failed: {e}")

    with st.expander("🗄️ Cache Information"):
        st.json(st.session_state.xl_merge_pipeline.get_cache_info())


# === BATCH PROCESSING ===
with xl_batch_merger:
    st.markdown("### 📦 Batch Model Processing")
    st.markdown("*Create many merges between a Base Model and several Target Models (via API)*")

    st.markdown("#### 🎯 Select Base Model")
    base_model_col, base_info_col = st.columns([1, 2])

    try:
        models_response = api_client.list_models(MODEL_TYPE, "checkpoints")
        model_options = {m["name"]: m for m in models_response["models"]}
    except Exception as e:
        st.error(f"Failed to load models from API: {e}")
        model_options = {}

    with base_model_col:
        selected_base = st.selectbox(
            "Base Model",
            options=list(model_options.keys()),
            index=0,
        )

    with base_info_col:
        try:
            base_meta = api_client.get_model_metadata(MODEL_TYPE, "checkpoints", selected_base)
            st.json({"base_model": selected_base, "metadata": base_meta.get("metadata", {})}, expanded=False)
        except Exception as e:
            st.json({"base_model": selected_base, "error": str(e)}, expanded=False)

    st.markdown("#### 🎨 Select Target Models")
    target_models = st.multiselect(
        "Target Models",
        options=[m for m in model_options.keys() if m != selected_base],
    )

    if target_models:
        st.info(f"📊 Will create {len(target_models)} merged models")

        st.markdown("#### ⚙️ Batch Configuration")
        merge_settings, output_settings = st.columns([1, 1])

        with merge_settings:
            batch_method = st.selectbox(
                "Merge Method",
                options=["linear", "slerp", "additive", "subtract"],
                format_func=lambda x: {"linear": "📊 Linear", "slerp": "🌊 SLERP", "additive": "➕ Additive", "subtract": "➖ Subtractive"}[x],
            )
            batch_alpha = st.slider("Merge Strength", 0.0, 1.0, 0.5, 0.05)
            preserve_batch_metadata = st.checkbox("Preserve Metadata", value=True)

        with output_settings:
            output_subdir = st.text_input("Output Subdirectory", value="batch_merged")
            st.markdown("**📁 Output Preview:**")
            for target in target_models[:3]:
                st.text(f"📄 merged_{target}")
            if len(target_models) > 3:
                st.text(f"... and {len(target_models) - 3} more")

        st.markdown("#### 🚀 Execute Batch Processing")

        if st.button("🔄 Start Batch Merge", type="primary", disabled=len(target_models) == 0):
            try:
                with st.spinner(f"🔄 Processing {len(target_models)} models..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Submit batch merge job to API
                        job_id, _ = api_client.batch_merge_models(
                            model_type=MODEL_TYPE,
                            base_model=selected_base,
                            target_models=target_models,
                            method=batch_method,
                            alpha=batch_alpha,
                            output_subdir=output_subdir,
                            preserve_metadata=preserve_batch_metadata,
                        )

                        # Poll for completion with progress
                        while True:
                            status = api_client.get_job_status(job_id)

                            if status["status"] == "running":
                                progress = status.get("progress", 0.0)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing batch... {int(progress*100)}%")
                                time.sleep(0.5)
                            elif status["status"] == "completed":
                                progress_bar.empty()
                                status_text.empty()

                                result = status.get("result", {})
                                results = result.get("results", [])

                                st.markdown("#### 📊 Batch Results")
                                successful = sum(1 for r in results if r.get("success", False))
                                st.metric("Successful Merges", f"{successful}/{len(results)}")
                                st.dataframe([
                                    {
                                        "Model": r.get("model", ""),
                                        "Status": "✅ Success" if r.get("success", False) else "❌ Failed",
                                        "Path": r.get("path", "N/A")
                                    }
                                    for r in results
                                ], width='content')

                                if successful > 0:
                                    st.success(f"✅ Batch processing completed! {successful} models merged.")
                                else:
                                    st.error("❌ Batch processing failed for all models.")
                                break
                            elif status["status"] == "failed":
                                error = status.get("error", "Unknown error")
                                st.error(f"❌ Batch processing failed: {error}")
                                progress_bar.empty()
                                status_text.empty()
                                break
                            else:
                                time.sleep(0.5)

                    except Exception as e:
                        st.error(f"❌ Batch processing failed: {e}")
            except Exception as e:
                st.error(f"❌ Batch processing failed: {e}")
    else:
        st.info("👆 Select target models to begin batch processing")


# === RECIPE BUILDER ===
with xl_recipe_builder:
    st.markdown("### 🧪 Merge Recipe Builder")
    st.markdown("*Create complex multi-step merge operations (via API)*")

    if "xl_current_recipe" not in st.session_state:
        st.session_state.xl_current_recipe = {"base_model": "", "steps": []}

    st.markdown("#### 📋 Recipe Configuration")

    try:
        models_response = api_client.list_models(MODEL_TYPE, "checkpoints")
        model_options = {m["name"]: m for m in models_response["models"]}
    except Exception as e:
        st.error(f"Failed to load models from API: {e}")
        model_options = {}

    recipe_base = st.selectbox(
        "🎯 Recipe Base Model",
        options=list(model_options.keys()),
        index=0,
    )

    if recipe_base != st.session_state.xl_current_recipe["base_model"]:
        st.session_state.xl_current_recipe["base_model"] = recipe_base

    st.markdown("#### 🔧 Recipe Steps")

    with st.expander("➕ Add Merge Step", expanded=len(st.session_state.xl_current_recipe["steps"]) == 0):
        merge_step_selection, merge_step_settings = st.columns([1, 1])

        with merge_step_selection:
            step_target = st.selectbox(
                "Target Model",
                options=[m for m in model_options.keys() if m != recipe_base],
                key="new_step_target",
            )
            step_method = st.selectbox(
                "Merge Method",
                options=["linear", "slerp", "additive", "subtract"],
                key="new_step_method",
            )

        with merge_step_settings:
            step_alpha = st.slider("Step Strength", 0.0, 1.0, 0.5, 0.05, key="new_step_alpha")
            step_progress = st.checkbox("Show Progress", value=True, key="new_step_progress")

        if st.button("➕ Add Step"):
            st.session_state.xl_current_recipe["steps"].append({
                "target_model": step_target,  # Just store the name, not the path
                "method": step_method,
                "alpha": step_alpha,
                "show_progress": step_progress,
            })
            st.rerun()

    if st.session_state.xl_current_recipe["steps"]:
        st.markdown("#### 📜 Current Recipe")

        for i, step in enumerate(st.session_state.xl_current_recipe["steps"]):
            with st.container(border=True):
                step_info_col, step_action_col = st.columns([4, 1])

                with step_info_col:
                    st.markdown(f"**Step {i + 1}:** {step['method'].upper()} merge")
                    st.text(f"🎨 Target: {step['target_model']}")
                    st.text(f"💪 Alpha: {step['alpha']}")

                with step_action_col:
                    if st.button("🗑️ Delete", key=f"delete_step_{i}"):
                        st.session_state.xl_current_recipe["steps"].pop(i)
                        st.rerun()

        st.markdown("#### 🚀 Execute Recipe")

        result_pane, exec_pane = st.columns([2, 1])

        with result_pane:
            recipe_output = st.text_input(
                "Recipe Output Name",
                value=f"recipe_output_{len(st.session_state.xl_current_recipe['steps'])}steps.safetensors",
            )

        with exec_pane:
            if st.button("🧪 Execute Recipe", type="primary", disabled=len(st.session_state.xl_current_recipe["steps"]) == 0):
                try:
                    with st.spinner(f"🧪 Executing {len(st.session_state.xl_current_recipe['steps'])}-step recipe..."):
                        try:
                            # Convert recipe steps to API format
                            api_steps = []
                            for step in st.session_state.xl_current_recipe["steps"]:
                                # Extract just the filename from the full path
                                target_model_name = Path(step["target_model"]).name
                                api_steps.append({
                                    "target_model": target_model_name,
                                    "method": step["method"],
                                    "alpha": step["alpha"],
                                })

                            # Submit recipe job to API
                            job_id, _ = api_client.recipe_merge_models(
                                model_type=MODEL_TYPE,
                                base_model=recipe_base,
                                steps=api_steps,
                                output_name=recipe_output,
                            )

                            # Poll for completion
                            while True:
                                status = api_client.get_job_status(job_id)

                                if status["status"] == "completed":
                                    st.success("✅ Recipe executed successfully!")
                                    st.info(f"📁 Saved to: {recipe_output}")
                                    with result_pane.expander("📊 Recipe Summary"):
                                        st.json({
                                            "base_model": recipe_base,
                                            "steps_count": len(st.session_state.xl_current_recipe["steps"]),
                                            "output_file": recipe_output,
                                        })
                                    break
                                elif status["status"] == "failed":
                                    error = status.get("error", "Unknown error")
                                    st.error(f"❌ Recipe execution failed: {error}")
                                    break
                                else:
                                    time.sleep(0.5)

                        except Exception as e:
                            st.error(f"❌ Recipe execution failed: {e}")
                except Exception as e:
                    st.error(f"❌ Recipe execution failed: {e}")

            if st.button("🗑️ Clear Recipe"):
                st.session_state.xl_current_recipe = {"base_model": recipe_base, "steps": []}
                st.rerun()
    else:
        st.info("👆 Add merge steps to build your recipe")
