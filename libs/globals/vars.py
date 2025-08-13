#!/usr/bin/env python

# load schedulers
from diffusers.schedulers import (DPMSolverMultistepScheduler,
                                  DPMSolverSinglestepScheduler,
                                  EulerDiscreteScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  KDPM2DiscreteScheduler,
                                  HeunDiscreteScheduler,
                                  LMSDiscreteScheduler)

# define globals
DEFAULT_MODELS_PATH = "models/stable-diffusion"
DEFAULT_LORA_PATH = "models/lora"
SFT_HEADER_LEN = 8
RANDOM_BIT_LENGTH = 64

# scheduler types
schedulers = {"DPM++ 2M": DPMSolverMultistepScheduler,
              "DPM++ SDE": DPMSolverSinglestepScheduler,
              "DPM2": KDPM2DiscreteScheduler,
              "Euler a": EulerAncestralDiscreteScheduler,
              "Euler": EulerDiscreteScheduler,
              "Heun": HeunDiscreteScheduler,
              "LMS": LMSDiscreteScheduler}

