#!/usr/bin/env python

# load libs
try:
    import numpy as np
    from torch import Generator
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, AutoencoderKL, UNet2DConditionModel
    from libs.globals.vars import RANDOM_BIT_LENGTH, schedulers
    from libs.shared.utils import get_gpu
    from pathlib import PosixPath
    import random
    from PIL import Image
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
#        "width": 1024,
#        "height": 1024,
#        "guidance_scale": 7,
#        "seed": 772847624537827,
#      }
#    ]
#  }


# prepare the payload
def format_metadata(
    prompt,
    negative_prompt="",
    steps=10,
    width=1024,
    height=1024,
    cfg=7,
    seed=-1,
    scheduler=None,
):
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


# callback for sdxl generation on a local machine
def local_prediction(
    model_pipeline,
    prompt,
    negative_prompt="",
    steps=10,
    width=1024,
    height=1024,
    guidance_scale=7,
    seed=-1,
    scheduler=None,
    accelerator="cpu",
):
    # prepare generator object
    if seed == -1:
        gen = Generator(accelerator).manual_seed(random.getrandbits(RANDOM_BIT_LENGTH))
    else:
        gen = Generator(accelerator).manual_seed(seed)

    # generate image from prompt
    prediction = model_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        generator=gen,
    )

    # generation metadata payload
    metadata = format_metadata(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=width,
        height=height,
        cfg=guidance_scale,
        seed=seed,
        scheduler=scheduler,
    )

    return np.array(prediction.images[0]), metadata


# callback for SDXL inpainting on a local machine
def local_inpaint_prediction(
    model_pipeline,
    prompt,
    image,
    mask_image,
    negative_prompt="",
    steps=10,
    guidance_scale=7,
    seed=-1,
    scheduler=None,
    accelerator="cpu",
):
    # prepare generator object
    if seed == -1:
        gen = Generator(accelerator).manual_seed(random.getrandbits(RANDOM_BIT_LENGTH))
    else:
        gen = Generator(accelerator).manual_seed(seed)

    # ensure image and mask are PIL Images
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
    if isinstance(mask_image, np.ndarray):
        mask_image = Image.fromarray((mask_image * 255).astype(np.uint8) if mask_image.max() <= 1.0 else mask_image.astype(np.uint8))
    
    # convert mask to grayscale if needed
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")

    # generate inpainted image
    prediction = model_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )

    # generation metadata payload
    metadata = format_metadata(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        width=image.width,
        height=image.height,
        cfg=guidance_scale,
        seed=seed,
        scheduler=scheduler,
    )
    metadata["inpainting"] = True

    return np.array(prediction.images[0]), metadata


# generate sample noise
def gen_noise(width: int = 1024, height: int = 1024, channels: int = 3):
    noise_image = np.random.rand(width, height, channels)
    noise_properties = {
        "width": width,
        "height": height,
        "channels": channels,
        "generation": {
            "status": "idle",
            "output": "noise matrix",
        },
    }
    return noise_image, noise_properties


# load custom UNET weights for SDXL
def load_custom_unet(ckpt: str | PosixPath) -> UNet2DConditionModel:
    try:
        if type(ckpt) is str:
            fname = ckpt
        elif type(ckpt) is PosixPath:
            fname = ckpt.absolute()
        else:
            raise Exception(
                "load_custom_unet(): filename must be a string or Path object"
            )

        # load the UNet Model from checkpoint
        _unet = UNet2DConditionModel.from_single_file(
            fname, subfolder="unet", use_safetensors=True
        )
        return _unet
    except Exception as e:
        raise (e)


# load custom Variational Autoencoder
def load_custom_vae(ckpt: str | PosixPath) -> AutoencoderKL:
    try:
        if type(ckpt) is str:
            fname = ckpt
        elif type(ckpt) is PosixPath:
            fname = ckpt.absolute()
        else:
            raise Exception(
                "load_custom_vae(): filename must be a string or Path object"
            )

        # load the vae from checkpoint
        print(f"Loading custom VAE {ckpt}")
        _vae = AutoencoderKL.from_single_file(
            fname, subfolder="vae", use_safetensors=True
        )
        return _vae
    except Exception as e:
        raise (e)


# SDXL Generator Class
class SDXLPipelineGenerator:
    def __init__(self, model_checkpoint: str):
        self.model_checkpoint = model_checkpoint
        self.sdxl_pipeline = None
        self.inpaint_pipeline = None

    # generation callback
    def forward(
        self,
        positive_prompt,
        negative_prompt,
        scheduler_type,
        steps,
        width,
        height,
        cfg,
        seed,
    ):
        # check if model is ready
        if self.sdxl_pipeline is None:
            import numpy as np

            return np.random.rand(width, height, 3), {
                "generation": {"status": "no model loaded", "output": "noise matrix"}
            }
        else:
            # call local callback function
            print(f"Using Scheduler {scheduler_type}")
            self.sdxl_pipeline.scheduler = schedulers.get(scheduler_type).from_config(
                self.sdxl_pipeline.scheduler.config
            )
            return local_prediction(
                self.sdxl_pipeline,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                width=width,
                height=height,
                seed=seed,
                guidance_scale=cfg,
                scheduler=scheduler_type,
                accelerator=self.accelerator,
            )

    # return scheduler config
    def getSchedulerConfig(self):
        if self.inpaint_pipeline is not None:
            return self.inpaint_pipeline.scheduler.config
        elif self.sdxl_pipeline is not None:
            return self.sdxl_pipeline.scheduler.config
        else:
            return None

    # load stable diffusion xl model from disk
    def loadSDXLPipeline(self):
        # check for GPU
        try:
            self.accelerator, self.dtype = get_gpu()
            print(f"Loading SDXL Checkpoint {self.model_checkpoint}")
            self.sdxl_pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
            )
        except Exception as e:
            raise Exception(f"Caught Exception {e}", duration=5)

    # load lora adapters
    def addLorasToPipeline(self, loras: dict = None):
        assert loras is not None

        # extract checkpoint files
        lora_checkpoints = [loras[k] for k in loras.keys()]

        # add low rank adapter weights
        if lora_checkpoints is not None:
            for entry in lora_checkpoints:
                weightsfile = PosixPath(entry.get("lora_path"))
                strength = entry.get("merge_strength")
                print(f"Loading Lora: {weightsfile}, fusion strength: {strength}")
                if self.sdxl_pipeline is not None:
                    self.sdxl_pipeline.load_lora_weights(
                        weightsfile,
                        adapter_name=f"name_{weightsfile.name.split('.')[0]}",
                    )
                    # merge lora
                    self.sdxl_pipeline.fuse_lora(
                        lora_scale=strength,
                        adapter_name=f"name_{weightsfile.name.split('.')[0]}",
                    )
                if self.inpaint_pipeline is not None:
                    self.inpaint_pipeline.load_lora_weights(
                        weightsfile,
                        adapter_name=f"name_{weightsfile.name.split('.')[0]}",
                    )
                    # merge lora
                    self.inpaint_pipeline.fuse_lora(
                        lora_scale=strength,
                        adapter_name=f"name_{weightsfile.name.split('.')[0]}",
                    )
                if self.sdxl_pipeline is None and self.inpaint_pipeline is None:
                    raise Exception("SDXL Model is not loaded")

    # send to accelerator
    def pipeToConfiguredDevice(self):
        # send model pipeline to the appropriate compute device
        if self.sdxl_pipeline is not None:
            self.sdxl_pipeline.to(self.accelerator)
        if self.inpaint_pipeline is not None:
            self.inpaint_pipeline.to(self.accelerator)

    # load custom VAE into pipeline
    def loadCustomVAE(self, vae_checkpoint: str | PosixPath):
        if self.sdxl_pipeline is not None:
            custom_vae = load_custom_vae(vae_checkpoint)
            self.sdxl_pipeline.vae = custom_vae
        else:
            raise Exception("SDXL Pipeline must be loaded before loading custom VAE")

    # inpainting callback
    def forward_inpaint(
        self,
        positive_prompt,
        negative_prompt,
        image,
        mask_image,
        scheduler_type,
        steps,
        cfg,
        seed,
    ):
        # check if model is ready
        if self.inpaint_pipeline is None:
            import numpy as np
            return np.random.rand(1024, 1024, 3), {
                "generation": {"status": "no inpainting model loaded", "output": "noise matrix"}
            }
        else:
            # call local inpainting callback function
            print(f"Using Scheduler {scheduler_type} for inpainting")
            self.inpaint_pipeline.scheduler = schedulers.get(scheduler_type).from_config(
                self.inpaint_pipeline.scheduler.config
            )
            return local_inpaint_prediction(
                self.inpaint_pipeline,
                prompt=positive_prompt,
                image=image,
                mask_image=mask_image,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed,
                guidance_scale=cfg,
                scheduler=scheduler_type,
                accelerator=self.accelerator,
            )

    # load stable diffusion xl inpainting pipeline from disk
    def loadSDXLInpaintPipeline(self):
        # check for GPU
        try:
            self.accelerator, self.dtype = get_gpu()
            print(f"Loading SDXL Inpaint Checkpoint {self.model_checkpoint}")
            self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
                self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
            )
        except Exception as e:
            raise Exception(f"Caught Exception {e}", duration=5)

    # get common SDXL resolutions
    @staticmethod
    def get_sdxl_resolutions():
        return {
            "1024x1024": (1024, 1024),
            "1152x896": (1152, 896),
            "896x1152": (896, 1152),
            "1216x832": (1216, 832),
            "832x1216": (832, 1216),
            "1344x768": (1344, 768),
            "768x1344": (768, 1344),
            "1536x640": (1536, 640),
            "640x1536": (640, 1536),
        }
