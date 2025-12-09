#!/usr/bin/env python

import os
try:
    import struct
    import json
    import torch
    from pathlib import Path, PosixPath
    from libs.globals.vars import DEFAULT_MODELS_PATH, DEFAULT_LORA_PATH, SFT_HEADER_LEN
except ImportError as e:
    print(f"Caught fatal exception: {e}")

# buil requests header object
def build_header(api_key: str) -> dict:
    if api_key is None or api_key == "":
        api_key = "apikey_openai"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"  # Bearer token format
    }
    return headers

# ensure directory structure exists
def check_or_create_path(targetpath: str) -> None:
    if not os.path.isdir(targetpath):
        os.makedirs(targetpath, exist_ok=True)

# scrape model filenames from the filesystem
def enumerate_models(path: str = DEFAULT_MODELS_PATH) -> dict:
    mp = Path(path)
    model_files = mp.glob("**/*.safetensors")

    # populate dictionary
    available_models = {}
    for filename in model_files:
        available_models[os.path.basename(filename)] = filename

    return available_models

# open a safetensors file and read its header
def read_safetensors_header(filename: str|PosixPath) -> str:
    try:
        if type(filename) == str:
            fname = filename
        elif type(filename) == PosixPath:
            fname = filename.absolute()
        else:
            raise Exception("read_safetensors_header(): filename must be a string or Path object")
        # open Path
        with open(fname, "rb") as s_fd:
            # read header
            header_bytes = s_fd.read(SFT_HEADER_LEN)
            metadata_len = struct.unpack("<Q", header_bytes)[0]

            # read metadata from file
            metadata = s_fd.read(metadata_len)
            return json.loads(metadata)
    except Exception as e:
        raise e

# detect available gpu
def get_gpu() -> (str, torch.dtype):
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

# generate a random string
def random_string(length=6):
    import string,random
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))