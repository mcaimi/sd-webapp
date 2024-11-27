#!/usr/bin/env python

# import libs
try:
    import os
    import gradio as gr
    from pathlib import Path
    from libs.callbacks import local_prediction
    from libs.utilities import read_safetensors_header, get_gpu
    from libs.vars import GRADIO_CUSTOM_PATH, GRADIO_MODELS_PATH, LORA_MODELS_PATH, schedulers
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

class StableDiffusionUI(object):
    def __init__(self):
        self.sd_ui = None
        self.sd_pipeline = None
        self.accelerator = None

    # generation callback
    def gen_callback(self, prompt, negative_prompt, scheduler_type, steps, width, height, cfg, seed):
        # check if model is ready
        if self.sd_pipeline is None:
            gr.Warning("Model is not ready, load one first!", duration=5)
            import numpy as np
            return np.random.rand(width, height, 3), {"generation": {"status": "no model loaded", "output": "noise matrix"}}
        else:
            # call local callback function
            print(f"Using Scheduler {scheduler_type}")
            self.sd_pipeline.scheduler = schedulers.get(scheduler_type).from_config(self.sd_pipeline.scheduler.config)
            return local_prediction(self.sd_pipeline,
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    steps=steps,
                                    width=width, height=height,
                                    seed=seed,
                                    guidance_scale=cfg,
                                    scheduler=scheduler_type,
                                    accelerator=self.accelerator)

    # model scraper
    def availableModelFiles(self, models_path=GRADIO_MODELS_PATH):
        try:
            mp = Path(models_path)
            files = mp.glob("**/*.safetensors")
            return [os.path.basename(fullname) for fullname in files]
        except Exception:
            raise gr.Error(f"Failed to list model safetensors from {models_path}", duration=5)

    # read html components
    def html_component(self, path):
        try:
            with open(path) as x:
                return "".join([i.strip() for i in x.readlines()])
        except Exception:
            raise gr.Error(f"Html Component {path} not found", duration=5)

    # load stable diffusion model from disk
    def loadModel(self, model, lora=None):
        if model is None:
            gr.Error("Please select a model from the dropdown list")
            return
        else:
            gr.Info(f"Loading model {model}", duration=5)
            model = "/".join((GRADIO_MODELS_PATH, model))

        try:
            from diffusers import StableDiffusionPipeline
        except Exception as e:
            raise gr.Error(f"Cannot import: {e}", duration=5)

        # clean resources while changing model checkpoint
        if self.sd_pipeline is not None:
            print(f"Cleaning up pipeline before loading {model}")
            del self.sd_pipeline
            self.sd_pipeline = None

        # check for GPU
        try:
            self.accelerator, self.dtype = get_gpu()
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(model, torch_dtype=self.dtype, use_safetensors=True)
        except Exception as e:
            raise gr.Error(f"Caught Exception {e}", duration=5)

        # add low rank adaptation weights
        if lora is not None:
            for weightsfile in lora:
                print(f"Loading Lora: {lora}")
                self.sd_pipeline.load_lora_weights("/".join((LORA_MODELS_PATH, weightsfile)))

        # send model pipeline to the appropriate compute device
        self.sd_pipeline.to(self.accelerator)

    # build interface for a locally hosted model
    def buildUi(self):
        # render interface
        with gr.Blocks(theme=gr.themes.Soft()) as sdInterface:
            gr.HTML(value=self.html_component("assets/header.html"))
            with gr.Row():
                model_dropdown = gr.Dropdown(scale=2, min_width=300, multiselect=False, label="SD Model", choices=self.availableModelFiles())
                load_btn = gr.Button(value="Load Model", variant="primary")
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt")
                    negative_prompt = gr.Textbox(label="Negative Prompt")
                    with gr.Row():
                        lora_dropdown = gr.Dropdown(scale=2, min_width=300, multiselect=True, label="LoRA", choices=self.availableModelFiles(models_path=LORA_MODELS_PATH))
                        scheduler_dropdown = gr.Dropdown(scale=2, min_width=300, multiselect=False, label="Scheduler", choices=[i for i in schedulers.keys()])
                    with gr.Row():
                        submit_btn = gr.Button(value="Generate", variant="primary")
                        clear_btn = gr.ClearButton(value="Clear")
                    with gr.Accordion("Additional Parameters", open=False):
                        steps = gr.Slider(label="Denoising Steps", value=5, minimum=1, maximum=100, step=1)
                        cfg = gr.Slider(label="Guidance Scale", value=7, minimum=1, maximum=100, step=0.5)
                        seed = gr.Number(label="Seed", value=-1)
                        with gr.Row():
                            width = gr.Number(label="Width", value=512)
                            height = gr.Number(label="Height", value=512)
                with gr.Column():
                    output_image = gr.Image(label="Generated Image", format="png", show_download_button=True)
                    with gr.Accordion("Image Parameters", open=False):
                        json_out = gr.JSON(label="Generation Parameters")

            @gr.render(inputs=lora_dropdown)
            def show_lora_panel(choices):
                if choices is not None and len(choices) >= 2:
                    with gr.Row():
                        for i in choices:
                            gr.Number(label=i, value=1.0)

            # attach function callbacks
            submit_btn.click(fn=self.gen_callback, inputs=[prompt, negative_prompt, scheduler_dropdown, steps, width, height, cfg, seed], outputs=[output_image, json_out], api_name=False)
            load_btn.click(fn=self.loadModel, inputs=[model_dropdown, lora_dropdown])
            clear_btn.add(components=[prompt, negative_prompt, steps, width, height, cfg, seed])
            model_dropdown.input(fn=read_safetensors_header, inputs=[model_dropdown])

        self.sd_ui = sdInterface

    # register application in FastAPI
    def registerFastApiEndpoint(self, fastApiApp, path=GRADIO_CUSTOM_PATH):
        fastApiApp = gr.mount_gradio_app(fastApiApp, self.sd_ui, path=path)
