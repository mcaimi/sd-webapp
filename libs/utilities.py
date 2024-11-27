#!/usr/bin/env python

import struct
import json
from .vars import GRADIO_MODELS_PATH, LORA_MODELS_PATH

SFT_HEADER_LEN = 8


# open a safetensors file and read its header
def read_safetensors_header(filename: str) -> str:
    try:
        with open("/".join((GRADIO_MODELS_PATH, filename)), "rb") as s_fd:
            # read header
            header_bytes = s_fd.read(SFT_HEADER_LEN)
            metadata_len = struct.unpack("<Q", header_bytes)[0]

            # read metadata from file
            metadata = s_fd.read(metadata_len)
            return json.loads(metadata)
    except Exception as e:
        raise e


# detect available gpu
def get_gpu() -> (str, int):
    try:
        import torch
        import torch.cuda as cuda
        import torch.backends.mps as apple_mps
    except Exception as e:
        raise e

    accelerator = "cpu"
    dtype = torch.float16
    if apple_mps.is_available():
        print("Apple Metal Performance Shaders Available!")
        accelerator = "mps"
    elif cuda.is_available():
        device_name = cuda.get_device_name()
        device_capabilities = cuda.get_device_capability()
        device_available_mem, device_total_mem = [x / 1024**3 for x in cuda.mem_get_info()]
        print(f"A GPU is available! [{device_name} - {device_capabilities} - {device_available_mem}/{device_total_mem} GB VRAM]")
        accelerator = "cuda"
    else:
        print("NO GPU FOUND.")
        dtype = torch.float32

    return accelerator, dtype
