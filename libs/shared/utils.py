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

"""
Generation Metadata Container

Provides a container class for image generation metadata.
"""

import json
from typing import Dict, Any


class GenerationMetadata:
    """Container for image generation metadata with dynamic attribute access."""

    def __init__(self, metadict: Dict[str, Any]) -> None:
        """
        Initialize metadata from a dictionary.

        Args:
            metadict: Dictionary containing generation metadata

        Raises:
            ValueError: If metadata dictionary is invalid or missing required fields
        """
        self.metadata: Dict[str, Any] = metadict

        # Load generation parameters
        try:
            for k in self.metadata.keys():
                setattr(self, f"{k}", self.metadata.get(k))

            # Access generation data specifically
            instance_parms: Dict[str, Any] = self.output_parameters.get("instances")[0]

            # Set attributes
            for k in instance_parms.keys():
                setattr(self, f"{k}", instance_parms.get(k))
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError(f"Cannot load metadata from dictionary: {e}") from e