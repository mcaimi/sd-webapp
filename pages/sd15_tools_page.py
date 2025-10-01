#!/usr/bin/env python
"""
Stable Diffusion 1.5 Merge Features Page

- Checkpoint structure analysis
- Simple linear merge
- Advanced 2-model merging
- Batch Merging
- Multi-step merging

"""

try:
    import streamlit as st
    from dotenv import dotenv_values

    with st.spinner("** LOADING INTERFACE... **"):
        # local imports
        from libs.shared.settings import Properties
        from libs.shared.session import Session
        from libs.shared.utils import enumerate_models, read_safetensors_header, check_or_create_path, get_gpu
        from libs.stablediffusion.sd15 import gen_noise, load_custom_vae
        from libs.stablediffusion.sd15 import SD15PipelineGenerator
        from libs.stablediffusion.sd15_merge import (
            load_checkpoint_dict, save_checkpoint, validate_checkpoints,
            SD15MergePipeline, MergeConfig, MergeMethod, validate_checkpoints,
            batch_merge_checkpoints
        )
        import os
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

# Initialize merge pipeline in session state
if 'merge_pipeline' not in st.session_state:
    st.session_state.merge_pipeline = SD15MergePipeline(device=get_gpu()[0])

# paths sanity check
with st.spinner("Checking Paths"):
    for path in [ appSettings.config_parameters.checkpoints.sd15.path, appSettings.config_parameters.loras.sd15.path,
                appSettings.config_parameters.checkpoints.sdxl.path, appSettings.config_parameters.loras.sdxl.path,
                appSettings.config_parameters.vae.sdxl.path, appSettings.config_parameters.vae.sd15.path ]:
        check_or_create_path(path)

# declare function tabs
sd15_info, sd15_merger, batch_merger, recipe_builder = st.tabs(["Checkpoint Explorer", "Advanced Merger", "Batch Processing", "Recipe Builder"])

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
    st.markdown("### 🔀 Model Merger")
    st.markdown("*Merge two SD1.5 models using different methods*")
    
    # Model selection section
    model_selection = st.container()
    with model_selection:
        model_a, model_b = st.columns([1,1], border=True)

        # Draw selection menus
        selected_checkpoint_a = model_a.selectbox("🎯 Base Model (A)",
                                        options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
                                        index=0,
                                        help="The primary model that will be used as the foundation")
        selected_checkpoint_b = model_b.selectbox("🎨 Target Model (B)", 
                                        options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
                                        index=0,
                                        help="The model whose features will be merged into the base")

    # Enhanced preflight check
    with st.expander(label="🔍 Merge Compatibility Check", expanded=True):
        info_a, info_b, compatibility = st.columns([2,2,1], border=True)

        # Get checkpoint paths
        ckpt_path_a = enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint_a)
        ckpt_path_b = enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_checkpoint_b)

        # Read headers
        json_a = read_safetensors_header(ckpt_path_a)
        json_b = read_safetensors_header(ckpt_path_b)

        # Display info
        info_a.markdown("**Base Model Info**")
        info_a.json(json_a, expanded=False)
        info_b.markdown("**Target Model Info**") 
        info_b.json(json_b, expanded=False)

        # Enhanced validation
        try:
            # Load just the keys for validation (lightweight)
            with st.spinner("Validating compatibility..."):
                base_weights, _ = load_checkpoint_dict(ckpt_path_a, device=get_gpu()[0])
                target_weights, _ = load_checkpoint_dict(ckpt_path_b, device=get_gpu()[0])
                
                can_merge, errors = validate_checkpoints(base_weights, target_weights)
                
                if can_merge:
                    compatibility.success('Compatible!', icon="✅")
                    compatibility.metric("Layers", len(base_weights))
                else:
                    compatibility.error('Incompatible', icon="🚨")
                    with compatibility.expander("Error Details"):
                        for error in errors[:3]:  # Show first 3 errors
                            st.error(error)
        except Exception as e:
            print(e)
            compatibility.error(f"Validation failed: {str(e)[:50]}...", icon="⚠️")
            can_merge = False

    # Merge configuration section
    with st.expander(label="🔍 Merge Setup", expanded=False):
        st.markdown("### ⚙️ Choose Merge Parameters")
        
        merge_controls, ops_controls = st.columns([1, 1])
        
        with merge_controls:
            # Merge method selection
            merge_method = st.selectbox(
                "🧮 Merge Algorithm",
                options=["linear", "slerp", "additive", "subtract"],
                format_func=lambda x: {
                    "linear": "📊 Linear (Classic blend)",
                    "slerp": "🌊 SLERP (Spherical interpolation)", 
                    "additive": "➕ Additive (Add features)",
                    "subtract": "➖ Subtract (Remove features)",
                }[x],
                help="Choose the merge algorithm. SLERP preserves model characteristics better, Additive adds features without losing the base."
            )
            
            # Alpha/strength control
            alpha = st.slider(
                "💪 Mix Strength", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05,
                help="0.0 = Pure base model, 1.0 = Pure target model"
            )
            
            # Progress tracking
            show_progress = st.checkbox("📊 Show Progress", value=True)
            preserve_metadata = st.checkbox("📋 Preserve Metadata", value=True)

        with ops_controls:
            # Advanced merge button
            do_advanced_merge = st.button(
                f"🔬 Advanced {merge_method.upper()} Merge", 
                type="primary", 
                disabled=not can_merge,
                help="Execute merge with current configuration"
            )
            
            # Output settings
            st.markdown("**💾 Output Settings**")
            save_output = st.checkbox("Save Result", value=True)
            if save_output:
                output_name = st.text_input(
                    "Output Filename", 
                    value=f"merged_{merge_method}_{selected_checkpoint_a.split('.')[0]}.safetensors",
                    help="Name for the merged checkpoint file"
                )
  
    merge_column, output_column = st.columns([2,2])

    # Execute advanced merge
    if do_advanced_merge and can_merge:
        try:            
            # Execute merge
            output_path = f"{appSettings.config_parameters.checkpoints.sd15.path}/{output_name}" if save_output else None
            
            with st.spinner(f"🔄 Executing {merge_method.upper()} merge..."):
                # progress bar
                merge_progress = st.progress(0, text="Merging....")

                # Create merge configuration
                config = MergeConfig(
                    method=MergeMethod(merge_method),
                    alpha=alpha,
                    device=get_gpu()[0],
                    preserve_metadata=preserve_metadata,
                    progress_callback=lambda p: merge_progress.progress(p) if show_progress else None
                )

                if output_path:
                    result = st.session_state.merge_pipeline.merge_for_pipeline_generator(
                        base_model=ckpt_path_a,
                        target_model=ckpt_path_b,
                        config=config,
                        output_path=output_path
                    )
                    
                    if result:
                        merge_column.success(f"✅ Merge completed successfully!")
                        merge_column.info(f"📁 Saved to: {output_name}")
                        
                        # Show merge info
                        with output_column.expander("📊 Merge Details", expanded=True):
                            merge_info = {
                                "method": merge_method,
                                "alpha": alpha,
                                "base_model": selected_checkpoint_a,
                                "target_model": selected_checkpoint_b,
                                "output_file": output_name
                            }
                            st.json(merge_info)

                        # empty progress bar
                        merge_progress.empty()
                    else:
                        st.error("❌ Merge failed. Check console for details.")
                else:
                    st.warning("⚠️ Merge not saved (save output was disabled)")
                    
        except Exception as e:
            st.error(f"❌ Merge failed: {str(e)}")
            
    # Cache information
    with st.expander("🗄️ Cache Information"):
        cache_info = st.session_state.merge_pipeline.get_cache_info()
        st.json(cache_info)

with batch_merger:
    st.markdown("### 📦 Batch Model Processing")
    st.markdown("*Create Many Merges between a Base Model and Several Target Models in an automated Batch*")
    
    # Base model selection
    st.markdown("#### 🎯 Select Base Model")
    base_model_col, base_info_col = st.columns([1, 2])
    
    with base_model_col:
        selected_base = st.selectbox(
            "Base Model",
            options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
            index=0,
            help="This model will be merged with all selected target models"
        )
    
    with base_info_col:
        base_path = enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(selected_base)
        base_metadata = read_safetensors_header(base_path)
        st.json({"base_model": selected_base, "metadata": base_metadata}, expanded=False)
    
    # Target models selection
    st.markdown("#### 🎨 Select Target Models")
    target_models = st.multiselect(
        "Target Models",
        options=[model for model in enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys() 
                if model != selected_base],
        help="Select multiple models to merge with the base model"
    )
    
    if target_models:
        st.info(f"📊 Will create {len(target_models)} merged models")
        
        # Batch configuration
        st.markdown("#### ⚙️ Batch Configuration")
        merge_settings, output_settings = st.columns([1, 1])
        
        with merge_settings:
            batch_method = st.selectbox(
                "Merge Method",
                options=["linear", "slerp", "additive", "subtract"],
                format_func=lambda x: {
                    "linear": "📊 Linear",
                    "slerp": "🌊 SLERP", 
                    "additive": "➕ Additive",
                    "subtract": "➖ Subtractive"
                }[x]
            )
            
            batch_alpha = st.slider("Merge Strength", 0.0, 1.0, 0.5, 0.05)
            preserve_batch_metadata = st.checkbox("Preserve Metadata", value=True)
        
        with output_settings:
            # Output directory
            output_subdir = st.text_input(
                "Output Subdirectory", 
                value="batch_merged",
                help="Subdirectory within checkpoints folder for batch outputs"
            )
            
            # Preview output names
            st.markdown("**📁 Output Preview:**")
            for target in target_models[:3]:  # Show first 3
                output_name = f"merged_{target}"
                st.text(f"📄 {output_name}")
            if len(target_models) > 3:
                st.text(f"... and {len(target_models) - 3} more")
        
        # Execute batch processing
        st.markdown("#### 🚀 Execute Batch Processing")
        
        if st.button("🔄 Start Batch Merge", type="primary", disabled=len(target_models) == 0):
            try:
                # Create batch configuration
                batch_config = MergeConfig(
                    method=MergeMethod(batch_method),
                    alpha=batch_alpha,
                    device=get_gpu()[0],
                    preserve_metadata=preserve_batch_metadata
                )
                
                # Prepare target checkpoint paths
                target_paths = [
                    enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(target)
                    for target in target_models
                ]
                
                # Create output directory
                output_dir = PosixPath(appSettings.config_parameters.checkpoints.sd15.path) / output_subdir
                os.makedirs(output_dir, exist_ok=True)
                
                # Execute batch merge
                with st.spinner(f"🔄 Processing {len(target_models)} models..."):
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for i, (target_model, target_path) in enumerate(zip(target_models, target_paths)):
                        # Update progress
                        progress = (i + 1) / len(target_models)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {target_model} ({i+1}/{len(target_models)})")
                        
                        # Execute single merge
                        output_name = f"merged_{target_model}"
                        output_path = output_dir / output_name
                        
                        try:
                            result = st.session_state.merge_pipeline.merge_for_pipeline_generator(
                                base_model=base_path,
                                target_model=target_path,
                                config=batch_config,
                                output_path=output_path
                            )
                            
                            if result:
                                results.append({"model": target_model, "status": "✅ Success", "path": str(output_path)})
                            else:
                                results.append({"model": target_model, "status": "❌ Failed", "path": "N/A"})
                                
                        except Exception as e:
                            results.append({"model": target_model, "status": f"❌ Error: {str(e)[:30]}...", "path": "N/A"})
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                # Show results
                st.markdown("#### 📊 Batch Results")
                successful = sum(1 for r in results if "Success" in r["status"])
                st.metric("Successful Merges", f"{successful}/{len(results)}")
                
                # Results table
                results_df = []
                for result in results:
                    results_df.append({
                        "Model": result["model"],
                        "Status": result["status"],
                        "Output Path": result["path"]
                    })
                
                st.dataframe(results_df, use_container_width=True)
                
                if successful > 0:
                    st.success(f"✅ Batch processing completed! {successful} models merged successfully.")
                else:
                    st.error("❌ Batch processing failed for all models.")
                    
            except Exception as e:
                st.error(f"❌ Batch processing failed: {str(e)}")
    else:
        st.info("👆 Select target models to begin batch processing")

with recipe_builder:
    st.markdown("### 🧪 Merge Recipe Builder")
    st.markdown("*Create complex multi-step merge operations*")
    
    # Initialize recipe in session state
    if 'current_recipe' not in st.session_state:
        st.session_state.current_recipe = {
            'base_model': '',
            'steps': []
        }
    
    # Recipe configuration
    st.markdown("#### 📋 Recipe Configuration")
    
    # Base model for recipe
    recipe_base = st.selectbox(
        "🎯 Recipe Base Model",
        options=enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys(),
        index=0,
        help="Starting model for the recipe"
    )
    
    if recipe_base != st.session_state.current_recipe['base_model']:
        st.session_state.current_recipe['base_model'] = recipe_base
    
    # Recipe steps management
    st.markdown("#### 🔧 Recipe Steps")
    
    # Add new step
    with st.expander("➕ Add Merge Step", expanded=len(st.session_state.current_recipe['steps']) == 0):
        merge_step_selection, merge_step_settings = st.columns([1, 1])
        
        with merge_step_selection:
            step_target = st.selectbox(
                "Target Model",
                options=[model for model in enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).keys() 
                        if model != recipe_base],
                key="new_step_target"
            )
            
            step_method = st.selectbox(
                "Merge Method",
                options=["linear", "slerp", "additive", "subtract"],
                key="new_step_method"
            )
        
        with merge_step_settings:
            step_alpha = st.slider("Step Strength", 0.0, 1.0, 0.5, 0.05, key="new_step_alpha")
            step_progress = st.checkbox("Show Progress", value=True, key="new_step_progress")
           
        if st.button("➕ Add Step"):
            new_step = {
                "target_model": str(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(step_target)),
                "method": step_method,
                "alpha": step_alpha,
                "show_progress": step_progress
            }
            
            st.session_state.current_recipe['steps'].append(new_step)
            st.rerun()
    
    # Display current recipe
    if st.session_state.current_recipe['steps']:
        st.markdown("#### 📜 Current Recipe")
        
        for i, step in enumerate(st.session_state.current_recipe['steps']):
            with st.container(border=True):
                step_info_col, step_action_col = st.columns([4, 1])
                
                with step_info_col:
                    st.markdown(f"**Step {i+1}:** {step['method'].upper()} merge")
                    st.text(f"🎨 Target: {PosixPath(step['target_model']).name}")
                    st.text(f"💪 Alpha: {step['alpha']}")
                
                with step_action_col:
                    if st.button("🗑️ Delete Step", key=f"delete_step_{i}", help="Delete step"):
                        st.session_state.current_recipe['steps'].pop(i)
                        st.rerun()
        
        # Recipe actions
        st.markdown("#### 🚀 Execute Recipe")
        
        result_pane, exec_pane = st.columns([2, 1])
        
        with result_pane:
            recipe_output = st.text_input(
                "Recipe Output Name",
                value=f"recipe_output_{len(st.session_state.current_recipe['steps'])}steps.safetensors",
                help="Name for the final merged model"
            )
        
        with exec_pane:
            if st.button("🧪 Execute Recipe", type="primary", 
                        disabled=len(st.session_state.current_recipe['steps']) == 0):
                try:
                    # Create recipe for execution
                    recipe = st.session_state.merge_pipeline.create_merge_recipe(
                        base_model=str(enumerate_models(appSettings.config_parameters.checkpoints.sd15.path).get(recipe_base)),
                        merge_steps=st.session_state.current_recipe['steps']
                    )
                    
                    # Output path
                    output_path = f"{appSettings.config_parameters.checkpoints.sd15.path}/{recipe_output}"
                    
                    # Execute recipe
                    with st.spinner(f"🧪 Executing {len(st.session_state.current_recipe['steps'])}-step recipe..."):
                        success = st.session_state.merge_pipeline.execute_merge_recipe(recipe, output_path)
                        
                        if success:
                            st.success("✅ Recipe executed successfully!")
                            st.info(f"📁 Saved to: {recipe_output}")
                            
                            # Show recipe summary
                            with result_pane.expander("📊 Recipe Summary"):
                                st.json({
                                    "base_model": recipe_base,
                                    "steps_count": len(st.session_state.current_recipe['steps']),
                                    "output_file": recipe_output,
                                    "recipe_meta": recipe,
                                })
                        else:
                            st.error("❌ Recipe execution failed. Check console for details.")
                            
                except Exception as e:
                    st.error(f"❌ Recipe execution failed: {str(e)}")
            
            if st.button("🗑️ Clear Recipe"):
                st.session_state.current_recipe = {'base_model': recipe_base, 'steps': []}
                st.rerun()
    
    else:
        st.info("👆 Add merge steps to build your recipe")