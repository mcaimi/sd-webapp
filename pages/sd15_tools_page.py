#!/usr/bin/env python
"""
Stable Diffusion 1.5 Merge Features Page

Provides model merging tools:
- Checkpoint structure analysis
- Simple linear merge
- Advanced 2-model merging
- Batch Merging
- Multi-step merging recipes
"""

import os
from pathlib import Path

import streamlit as st

from libs.shared.config import get_app_config
from libs.shared.utils import enumerate_models, read_safetensors_header, get_gpu
from libs.stablediffusion.mergeops import load_checkpoint_dict, validate_checkpoints
from libs.stablediffusion.merge import SD15MergePipeline, MergeConfig, MergeMethod


# Configuration
MODEL_TYPE = "sd15"
config = get_app_config()

# Load header
st.html("assets/explore_header.html")

# Initialize merge pipeline in session state
if "merge_pipeline" not in st.session_state:
    st.session_state.merge_pipeline = SD15MergePipeline(device=get_gpu()[0])

# Ensure paths exist
config.setup_paths()

# Get model paths
checkpoint_path = config.checkpoints_sd15_path
lora_path = config.loras_sd15_path
vae_path = config.vae_sd15_path

# === TABS ===
sd15_info, sd15_merger, batch_merger, recipe_builder = st.tabs(
    ["Checkpoint Explorer", "Advanced Merger", "Batch Processing", "Recipe Builder"]
)


# === CHECKPOINT EXPLORER ===
with sd15_info:
    model_selection, model_info = st.columns([1, 2], border=True)

    selected_checkpoint = model_selection.selectbox(
        "Select a model",
        options=list(enumerate_models(checkpoint_path).keys()),
        index=0,
    )

    with st.spinner("Loading Model Metadata..."):
        try:
            model_options = enumerate_models(checkpoint_path)
            model_metadata = {
                "model_checkpoint": selected_checkpoint,
                "model_path": model_options.get(selected_checkpoint).absolute() if selected_checkpoint else None,
                "metadata": read_safetensors_header(model_options.get(selected_checkpoint)) if selected_checkpoint else {},
            }
            model_info.json(model_metadata, expanded=False)
        except Exception as e:
            model_info.json({"exception": str(e)}, expanded=False)

    # LoRA section
    lora_selection, lora_info = st.columns([1, 2], border=True)

    lora_options = enumerate_models(lora_path)
    selected_lora = lora_selection.selectbox(
        "Select LoRA Adapter",
        options=list(lora_options.keys()),
        index=0,
    )

    with st.spinner("Loading Lora Metadata..."):
        try:
            lora_metadata = {
                "name": selected_lora,
                "lora_path": lora_options.get(selected_lora).absolute() if selected_lora else None,
                "metadata": read_safetensors_header(lora_options.get(selected_lora)) if selected_lora else {},
            }
            lora_info.json(lora_metadata, expanded=False)
        except Exception as e:
            lora_info.json({"exception": str(e)}, expanded=False)

    # VAE section
    vae_selection, vae_info = st.columns([1, 2], border=True)

    vae_options = enumerate_models(vae_path)
    selected_vae = vae_selection.selectbox(
        label="Select SD15 VAE",
        options=list(vae_options.keys()),
        index=0,
    )

    with st.spinner("Loading VAE Metadata..."):
        try:
            vae_metadata = {
                "vae_checkpoint": selected_vae,
                "vae_path": vae_options.get(selected_vae).absolute() if selected_vae else None,
                "metadata": read_safetensors_header(vae_options.get(selected_vae)) if selected_vae else {},
            }
            vae_info.json(vae_metadata, expanded=False)
        except Exception as e:
            vae_info.json({"exception": str(e)}, expanded=False)


# === ADVANCED MERGER ===
with sd15_merger:
    st.markdown("### 🔀 Model Merger")
    st.markdown("*Merge two SD1.5 models using different methods*")

    model_options = enumerate_models(checkpoint_path)
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

    # Compatibility check
    with st.expander("🔍 Merge Compatibility Check", expanded=True):
        info_a, info_b, compatibility = st.columns([2, 2, 1], border=True)

        ckpt_path_a = model_options.get(selected_checkpoint_a)
        ckpt_path_b = model_options.get(selected_checkpoint_b)

        json_a = read_safetensors_header(ckpt_path_a)
        json_b = read_safetensors_header(ckpt_path_b)

        info_a.markdown("**Base Model Info**")
        info_a.json(json_a, expanded=False)
        info_b.markdown("**Target Model Info**")
        info_b.json(json_b, expanded=False)

        can_merge = False
        try:
            with st.spinner("Validating compatibility..."):
                base_weights, _ = load_checkpoint_dict(ckpt_path_a, device=get_gpu()[0])
                target_weights, _ = load_checkpoint_dict(ckpt_path_b, device=get_gpu()[0])
                can_merge, errors = validate_checkpoints(base_weights, target_weights)

                if can_merge:
                    compatibility.success("Compatible!", icon="✅")
                    compatibility.metric("Layers", len(base_weights))
                else:
                    compatibility.error("Incompatible", icon="🚨")
                    with compatibility.expander("Error Details"):
                        for error in errors[:3]:
                            st.error(error)
        except Exception as e:
            print(e)
            compatibility.error(f"Validation failed: {str(e)[:50]}...", icon="⚠️")

    # Merge configuration
    with st.expander("🔍 Merge Setup", expanded=False):
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
            show_progress = st.checkbox("📊 Show Progress", value=True)
            preserve_metadata = st.checkbox("📋 Preserve Metadata", value=True)

        with ops_controls:
            do_advanced_merge = st.button(
                f"🔬 Advanced {merge_method.upper()} Merge",
                type="primary",
                disabled=not can_merge,
            )

            st.markdown("**💾 Output Settings**")
            save_output = st.checkbox("Save Result", value=True)
            if save_output:
                output_name = st.text_input(
                    "Output Filename",
                    value=f"merged_{merge_method}_{selected_checkpoint_a.split('.')[0]}.safetensors",
                )

    merge_column, output_column = st.columns([2, 2])

    if do_advanced_merge and can_merge:
        try:
            output_path = f"{checkpoint_path}/{output_name}" if save_output else None

            with st.spinner(f"🔄 Executing {merge_method.upper()} merge..."):
                merge_progress = st.progress(0, text="Merging...")

                merge_config = MergeConfig(
                    method=MergeMethod(merge_method),
                    alpha=alpha,
                    device=get_gpu()[0],
                    preserve_metadata=preserve_metadata,
                    progress_callback=lambda p: merge_progress.progress(p) if show_progress else None,
                )

                if output_path:
                    result = st.session_state.merge_pipeline.merge_for_pipeline_generator(
                        base_model=ckpt_path_a,
                        target_model=ckpt_path_b,
                        config=merge_config,
                        output_path=output_path,
                    )

                    if result:
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
                    else:
                        st.error("❌ Merge failed. Check console for details.")
                else:
                    st.warning("⚠️ Merge not saved (save output was disabled)")

        except Exception as e:
            st.error(f"❌ Merge failed: {e}")

    with st.expander("🗄️ Cache Information"):
        st.json(st.session_state.merge_pipeline.get_cache_info())


# === BATCH PROCESSING ===
with batch_merger:
    st.markdown("### 📦 Batch Model Processing")
    st.markdown("*Create many merges between a Base Model and several Target Models*")

    st.markdown("#### 🎯 Select Base Model")
    base_model_col, base_info_col = st.columns([1, 2])

    model_options = enumerate_models(checkpoint_path)
    with base_model_col:
        selected_base = st.selectbox(
            "Base Model",
            options=list(model_options.keys()),
            index=0,
        )

    with base_info_col:
        base_path = model_options.get(selected_base)
        base_metadata = read_safetensors_header(base_path)
        st.json({"base_model": selected_base, "metadata": base_metadata}, expanded=False)

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
                batch_config = MergeConfig(
                    method=MergeMethod(batch_method),
                    alpha=batch_alpha,
                    device=get_gpu()[0],
                    preserve_metadata=preserve_batch_metadata,
                )

                target_paths = [model_options.get(t) for t in target_models]
                output_dir = Path(checkpoint_path) / output_subdir
                os.makedirs(output_dir, exist_ok=True)

                with st.spinner(f"🔄 Processing {len(target_models)} models..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    for i, (target_model, target_path) in enumerate(zip(target_models, target_paths)):
                        progress_bar.progress((i + 1) / len(target_models))
                        status_text.text(f"Processing {target_model} ({i + 1}/{len(target_models)})")

                        output_name = f"merged_{target_model}"
                        output_path = output_dir / output_name

                        try:
                            result = st.session_state.merge_pipeline.merge_for_pipeline_generator(
                                base_model=base_path,
                                target_model=target_path,
                                config=batch_config,
                                output_path=output_path,
                            )
                            results.append({
                                "model": target_model,
                                "status": "✅ Success" if result else "❌ Failed",
                                "path": str(output_path) if result else "N/A",
                            })
                        except Exception as e:
                            results.append({
                                "model": target_model,
                                "status": f"❌ Error: {str(e)[:30]}...",
                                "path": "N/A",
                            })

                    progress_bar.empty()
                    status_text.empty()

                st.markdown("#### 📊 Batch Results")
                successful = sum(1 for r in results if "Success" in r["status"])
                st.metric("Successful Merges", f"{successful}/{len(results)}")
                st.dataframe([{"Model": r["model"], "Status": r["status"], "Path": r["path"]} for r in results], width='content')

                if successful > 0:
                    st.success(f"✅ Batch processing completed! {successful} models merged.")
                else:
                    st.error("❌ Batch processing failed for all models.")

            except Exception as e:
                st.error(f"❌ Batch processing failed: {e}")
    else:
        st.info("👆 Select target models to begin batch processing")


# === RECIPE BUILDER ===
with recipe_builder:
    st.markdown("### 🧪 Merge Recipe Builder")
    st.markdown("*Create complex multi-step merge operations*")

    if "current_recipe" not in st.session_state:
        st.session_state.current_recipe = {"base_model": "", "steps": []}

    st.markdown("#### 📋 Recipe Configuration")

    model_options = enumerate_models(checkpoint_path)
    recipe_base = st.selectbox(
        "🎯 Recipe Base Model",
        options=list(model_options.keys()),
        index=0,
    )

    if recipe_base != st.session_state.current_recipe["base_model"]:
        st.session_state.current_recipe["base_model"] = recipe_base

    st.markdown("#### 🔧 Recipe Steps")

    with st.expander("➕ Add Merge Step", expanded=len(st.session_state.current_recipe["steps"]) == 0):
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
            st.session_state.current_recipe["steps"].append({
                "target_model": str(model_options.get(step_target)),
                "method": step_method,
                "alpha": step_alpha,
                "show_progress": step_progress,
            })
            st.rerun()

    if st.session_state.current_recipe["steps"]:
        st.markdown("#### 📜 Current Recipe")

        for i, step in enumerate(st.session_state.current_recipe["steps"]):
            with st.container(border=True):
                step_info_col, step_action_col = st.columns([4, 1])

                with step_info_col:
                    st.markdown(f"**Step {i + 1}:** {step['method'].upper()} merge")
                    st.text(f"🎨 Target: {Path(step['target_model']).name}")
                    st.text(f"💪 Alpha: {step['alpha']}")

                with step_action_col:
                    if st.button("🗑️ Delete", key=f"delete_step_{i}"):
                        st.session_state.current_recipe["steps"].pop(i)
                        st.rerun()

        st.markdown("#### 🚀 Execute Recipe")

        result_pane, exec_pane = st.columns([2, 1])

        with result_pane:
            recipe_output = st.text_input(
                "Recipe Output Name",
                value=f"recipe_output_{len(st.session_state.current_recipe['steps'])}steps.safetensors",
            )

        with exec_pane:
            if st.button("🧪 Execute Recipe", type="primary", disabled=len(st.session_state.current_recipe["steps"]) == 0):
                try:
                    recipe = st.session_state.merge_pipeline.create_merge_recipe(
                        base_model=str(model_options.get(recipe_base)),
                        merge_steps=st.session_state.current_recipe["steps"],
                    )

                    output_path = f"{checkpoint_path}/{recipe_output}"

                    with st.spinner(f"🧪 Executing {len(st.session_state.current_recipe['steps'])}-step recipe..."):
                        success = st.session_state.merge_pipeline.execute_merge_recipe(recipe, output_path)

                        if success:
                            st.success("✅ Recipe executed successfully!")
                            st.info(f"📁 Saved to: {recipe_output}")
                            with result_pane.expander("📊 Recipe Summary"):
                                st.json({
                                    "base_model": recipe_base,
                                    "steps_count": len(st.session_state.current_recipe["steps"]),
                                    "output_file": recipe_output,
                                    "recipe_meta": recipe,
                                })
                        else:
                            st.error("❌ Recipe execution failed. Check console for details.")

                except Exception as e:
                    st.error(f"❌ Recipe execution failed: {e}")

            if st.button("🗑️ Clear Recipe"):
                st.session_state.current_recipe = {"base_model": recipe_base, "steps": []}
                st.rerun()
    else:
        st.info("👆 Add merge steps to build your recipe")
