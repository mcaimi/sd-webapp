#!/usr/bin/env python

# load libs
try:
    import numpy as np
    from torch import Generator
    from diffusers import StableDiffusionPipeline
    from libs.globals.vars import RANDOM_BIT_LENGTH, schedulers
    from libs.shared.utils import get_gpu
    import random
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

# An example JSON payload:
#
# // example payload:
#  {
#    "instances": [
#      {
#        "prompt": "photo of the beach",
#        "negative_prompt": "ugly, deformed, bad anatomy",
#        "num_inference_steps": 20,
#        "width": 512,
#        "height": 512,
#        "guidance_scale": 7,
#        "seed": 772847624537827,
#      }
#    ]
#  }

# prepare the payload
def format_metadata(prompt,
                    negative_prompt="",
                    steps=10,
                    width=512, height=512,
                    cfg=7,
                    seed=-1, scheduler=None):
    # prepare seed
    if seed == -1:
        custom_seed = random.getrandbits(RANDOM_BIT_LENGTH)
        print(f"Generating with random seed: {custom_seed}")
    else:
        custom_seed = seed
        print(f"Generating with constant seed: {custom_seed}")

    # prepare payload
    json_data = {
        "instances": [
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "width": width,
                "height": height,
                "guidance_scale": cfg,
                "seed": custom_seed,
                "scheduler": scheduler,
            }
        ]
    }

    # return built payload
    return json_data


# callback for sd generation on a local machine
def local_prediction(model_pipeline,
                     prompt,
                     negative_prompt="",
                     steps=10,
                     width=512, height=512,
                     guidance_scale=7,
                     seed=-1,
                     scheduler=None,
                     accelerator="cpu"):
    # prepare generator object
    if seed == -1:
        gen = Generator(accelerator).manual_seed(random.getrandbits(RANDOM_BIT_LENGTH))
    else:
        gen = Generator(accelerator).manual_seed(seed)

    # generate image from prompt
    prediction = model_pipeline(prompt=prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=steps,
                                width=width,
                                height=height,
                                guidance_scale=guidance_scale,
                                generator=gen)

    # generation metagada payload
    metadata = format_metadata(prompt=prompt,
                               negative_prompt=negative_prompt,
                               steps=steps, width=width, height=height,
                               cfg=guidance_scale, seed=seed,
                               scheduler=scheduler)

    return np.array(prediction.images[0]), metadata

# generate sample noise
def noise_latent(width: int = 512, height: int = 512, channels: int = 3):
    noise_latent = np.random.rand(width, height, channels)
    noise_properties = {
        "width": width,
        "height": height,
        "channels": channels,
        "generation": {
            "status": "idle",
            "output": "noise matrix",
        }
    }
    return noise_latent, noise_properties


# SD1.5 Generator Class
class SD15Generator():
    def __init__(self, model_checkpoint: str):
        self.model_checkpoint = model_checkpoint
        self.sd_pipeline = None

    # generation callback
    def gen_callback(self, positive_prompt, negative_prompt, scheduler_type, steps, width, height, cfg, seed):
        # check if model is ready
        if self.sd_pipeline is None:
            import numpy as np
            return np.random.rand(width, height, 3), {"generation": {"status": "no model loaded", "output": "noise matrix"}}
        else:
            # call local callback function
            print(f"Using Scheduler {scheduler_type}")
            self.sd_pipeline.scheduler = schedulers.get(scheduler_type).from_config(self.sd_pipeline.scheduler.config)
            return local_prediction(self.sd_pipeline,
                                    prompt=positive_prompt,
                                    negative_prompt=negative_prompt,
                                    steps=steps,
                                    width=width, height=height,
                                    seed=seed,
                                    guidance_scale=cfg,
                                    scheduler=scheduler_type,
                                    accelerator=self.accelerator)

    # load stable diffusion model from disk
    def loadModel(self):
        # check for GPU
        try:
            self.accelerator, self.dtype = get_gpu()
            self.sd_pipeline = StableDiffusionPipeline.from_single_file(self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True)
        except Exception as e:
            raise gr.Error(f"Caught Exception {e}", duration=5)

        # send model pipeline to the appropriate compute device
        self.sd_pipeline.to(self.accelerator)

    # load lora adapters
    def loadLoras(self, loras: list = None):
        # add low rank adaptation weights
        if lora is not None:
            for weightsfile in lora:
                print(f"Loading Lora: {lora}")
                if self.sd_pipeline is not None:
                    self.sd_pipeline.load_lora_weights("/".join((LORA_MODELS_PATH, weightsfile)))
                else:
                    raise Exception("SD Model is not loaded")
