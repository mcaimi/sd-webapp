#!/usr/bin/env python

# load libs
try:
    from torch import Generator
    import random
except Exception as e:
    print(f"Caught exception: {e}")
    raise e

RANDOM_BIT_LENGTH = 64

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

    return prediction.images[0], metadata
