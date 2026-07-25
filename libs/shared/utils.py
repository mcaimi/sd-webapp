#!/usr/bin/env python
"""
Shared Utility Functions

Common utilities used across the application.
Does NOT include local model analytics - use API client instead.
"""

import random
import string
from typing import NamedTuple

from libs.globals.vars import RANDOM_BIT_LENGTH


def random_string(length: int = 6) -> str:
    """
    Generate a random lowercase string.

    Args:
        length: Length of the string to generate

    Returns:
        Random string of specified length
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def get_random_seed(seed_len: int = RANDOM_BIT_LENGTH) -> int:
    """
    Generate a random seed value.

    Args:
        seed_len: Number of random bits to generate

    Returns:
        Random integer seed
    """
    return random.getrandbits(seed_len)


class GenerationMetadata(NamedTuple):
    """
    Metadata for a generation request.

    Attributes:
        positive_prompt: The positive prompt
        negative_prompt: The negative prompt
        width: Image width
        height: Image height
        steps: Number of inference steps
        cfg_scale: CFG scale
        seed: Random seed
        scheduler: Scheduler name
    """

    positive_prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    scheduler: str
