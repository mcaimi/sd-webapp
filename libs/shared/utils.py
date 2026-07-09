#!/usr/bin/env python
"""
Shared Utility Functions

Common utilities used across the application for file operations,
GPU detection, safetensors handling, etc.
"""

import os
import json
import logging
import random
import string
import struct
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.cuda as cuda
import torch.backends.mps as apple_mps

from libs.globals.vars import DEFAULT_MODELS_PATH, DEFAULT_LORA_PATH, SFT_HEADER_LEN

logger = logging.getLogger(__name__)


def check_or_create_path(target_path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        target_path: Path to the directory
    """
    path = Path(target_path) if isinstance(target_path, str) else target_path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


def enumerate_models(path: Union[str, Path] = DEFAULT_MODELS_PATH) -> Dict[str, Path]:
    """
    Enumerate all safetensors model files in a directory.
    
    Args:
        path: Directory to search for models
        
    Returns:
        Dictionary mapping filename to full Path
    """
    model_path = Path(path)
    model_files = model_path.glob("**/*.safetensors")
    
    return {
        filepath.name: filepath
        for filepath in model_files
    }


def read_safetensors_header(filename: Union[str, Path]) -> Dict:
    """
    Read and parse the metadata header from a safetensors file.

    Args:
        filename: Path to the safetensors file

    Returns:
        Parsed metadata dictionary

    Raises:
        TypeError: If filename is not str or Path
        FileNotFoundError: If the file doesn't exist
        Exception: If file cannot be read or parsed
    """
    if isinstance(filename, str):
        filepath = Path(filename)
    elif isinstance(filename, Path):
        filepath = filename.absolute()
    else:
        raise TypeError(
            f"read_safetensors_header(): filename must be str or Path, got {type(filename).__name__}"
        )

    if not filepath.exists():
        raise FileNotFoundError(f"Safetensors file not found: {filepath}")

    with open(filepath, "rb") as f:
        # Read header length (8 bytes, little-endian unsigned long long)
        header_bytes = f.read(SFT_HEADER_LEN)
        metadata_len = struct.unpack("<Q", header_bytes)[0]

        # Read and parse metadata
        metadata = f.read(metadata_len)
        return json.loads(metadata)


def get_gpu() -> Tuple[str, torch.dtype]:
    """
    Detect and return the best available compute device.
    
    Returns:
        Tuple of (device_name, recommended_dtype)
        Device will be one of: "mps", "cuda", or "cpu"
    """
    accelerator = "cpu"
    dtype = torch.float16

    if apple_mps.is_available():
        logger.info("Apple Metal Performance Shaders Available!")
        accelerator = "mps"
    elif cuda.is_available():
        device_name = cuda.get_device_name()
        device_capabilities = cuda.get_device_capability()
        device_available_mem, device_total_mem = [
            x / 1024**3 for x in cuda.mem_get_info()
        ]
        logger.info(
            "GPU available: %s - %s - %.1f/%.1f GB VRAM",
            device_name, device_capabilities, device_available_mem, device_total_mem
        )
        accelerator = "cuda"
    else:
        logger.warning("No GPU found. Using CPU (this will be slow).")
        dtype = torch.float32

    return accelerator, dtype


def random_string(length: int = 6) -> str:
    """
    Generate a random lowercase string.
    
    Args:
        length: Length of the string to generate
        
    Returns:
        Random string of specified length
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def build_header(api_key: str) -> Dict[str, str]:
    """
    Build HTTP request headers with authorization.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        Headers dictionary
    """
    if not api_key:
        api_key = "apikey_openai"
    
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
