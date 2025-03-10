#!/usr/bin/env python

from diffusers.schedulers import (DPMSolverMultistepScheduler,
                                  DPMSolverSinglestepScheduler,
                                  EulerDiscreteScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  KDPM2DiscreteScheduler,
                                  HeunDiscreteScheduler,
                                  LMSDiscreteScheduler)

# define globals
GRADIO_CUSTOM_PATH = "/sdui"
GRADIO_MODELS_PATH = "models/stable-diffusion"
LORA_MODELS_PATH = "models/lora"

# scheduler types
schedulers = {"DPM++ 2M": DPMSolverMultistepScheduler,
              "DPM++ SDE": DPMSolverSinglestepScheduler,
              "DPM2": KDPM2DiscreteScheduler,
              "Euler a": EulerAncestralDiscreteScheduler,
              "Euler": EulerDiscreteScheduler,
              "Heun": HeunDiscreteScheduler,
              "LMS": LMSDiscreteScheduler}
