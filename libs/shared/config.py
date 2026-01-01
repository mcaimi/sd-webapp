#!/usr/bin/env python
"""
Centralized Application Configuration

Provides a single point of configuration loading and caching for the entire app.
All pages should import from here instead of loading config independently.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import dotenv_values
from yaml import safe_load, YAMLError

from libs.shared.parameters import Parameters
from libs.shared.utils import check_or_create_path


@dataclass
class AppConfig:
    """
    Centralized application configuration container.
    
    Provides access to all configuration parameters and utility methods
    for path management.
    """
    env: Dict[str, str]
    config_file: str
    parameters: Parameters
    
    @property
    def checkpoints_sd15_path(self) -> str:
        return self.parameters.checkpoints.sd15.path
    
    @property
    def checkpoints_sdxl_path(self) -> str:
        return self.parameters.checkpoints.sdxl.path
    
    @property
    def loras_sd15_path(self) -> str:
        return self.parameters.loras.sd15.path
    
    @property
    def loras_sdxl_path(self) -> str:
        return self.parameters.loras.sdxl.path
    
    @property
    def vae_sd15_path(self) -> str:
        return self.parameters.vae.sd15.path
    
    @property
    def vae_sdxl_path(self) -> str:
        return self.parameters.vae.sdxl.path
    
    @property
    def output_images_path(self) -> str:
        return self.parameters.storage.output_images
    
    @property
    def output_json_path(self) -> str:
        return self.parameters.storage.output_json
    
    def get_model_paths(self, model_type: str) -> Dict[str, str]:
        """
        Get all paths for a specific model type.
        
        Args:
            model_type: Either 'sd15' or 'sdxl'
            
        Returns:
            Dictionary with 'checkpoints', 'loras', and 'vae' paths
        """
        if model_type == "sd15":
            return {
                "checkpoints": self.checkpoints_sd15_path,
                "loras": self.loras_sd15_path,
                "vae": self.vae_sd15_path,
            }
        elif model_type == "sdxl":
            return {
                "checkpoints": self.checkpoints_sdxl_path,
                "loras": self.loras_sdxl_path,
                "vae": self.vae_sdxl_path,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def setup_paths(self) -> None:
        """Create all required directories if they don't exist."""
        paths = [
            self.checkpoints_sd15_path,
            self.checkpoints_sdxl_path,
            self.loras_sd15_path,
            self.loras_sdxl_path,
            self.vae_sd15_path,
            self.vae_sdxl_path,
            self.output_images_path,
            self.output_json_path,
        ]
        for path in paths:
            check_or_create_path(path)
    
    def get_output_path(self, filename: str, output_type: str = "images") -> Path:
        """
        Get full output path for a file.
        
        Args:
            filename: The filename to save
            output_type: Either 'images' or 'json'
            
        Returns:
            Full Path object to the output file
        """
        if output_type == "images":
            base_path = self.output_images_path
        elif output_type == "json":
            base_path = self.output_json_path
        else:
            raise ValueError(f"Unknown output type: {output_type}")
        
        return Path(base_path) / filename


def _load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, "r") as f:
            return safe_load(f)
    except YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML config: {e}")
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {config_file}")


@lru_cache(maxsize=1)
def get_app_config(env_file: str = ".env") -> AppConfig:
    """
    Get the application configuration (cached).
    
    This function is cached so configuration is only loaded once per session.
    
    Args:
        env_file: Path to the .env file
        
    Returns:
        AppConfig instance with all configuration loaded
    """
    # Load environment variables
    env = dotenv_values(env_file)
    
    # Get config file path from environment or use default
    config_file = env.get("CONFIG_FILE", "parameters.yaml")
    
    # Load YAML configuration
    config_dict = _load_yaml_config(config_file)
    parameters = Parameters(config_dict)
    
    # Create and return config object
    config = AppConfig(
        env=dict(env),
        config_file=config_file,
        parameters=parameters,
    )
    
    # Ensure all paths exist
    config.setup_paths()
    
    return config


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing or reloading)."""
    get_app_config.cache_clear()


# Convenience function for common pattern
def get_model_paths(model_type: str) -> Dict[str, str]:
    """Get model paths for the given model type."""
    return get_app_config().get_model_paths(model_type)

