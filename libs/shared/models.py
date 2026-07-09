#!/usr/bin/env python
"""
Data Models for UI Components

Provides dataclass containers for model selection, LoRA configuration,
and generation settings used across the UI.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class ModelSelection:
    """Container for model selection data."""
    name: str
    path: Optional[Path]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoraSelection:
    """Container for LoRA selection data."""
    name: str
    path: Optional[Path]
    merge_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationSettings:
    """Container for image generation settings."""
    positive_prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 7.0
    inference_steps: int = 50
    width: int = 512
    height: int = 768
    seed: int = -1
    scheduler_index: int = 0
    batch_size: int = 1
