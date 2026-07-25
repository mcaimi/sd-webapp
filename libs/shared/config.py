#!/usr/bin/env python
"""
Centralized Application Configuration

Provides a single point of configuration loading and caching for the entire app.
All pages should import from here instead of loading config independently.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Union

from dotenv import dotenv_values
from yaml import safe_load, YAMLError

from libs.shared.parameters import Parameters
from libs.shared.exceptions import ConfigurationError


# path sanity check: create if not existing
def check_or_create_path(target_path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        target_path: Path to the directory
    """
    path = Path(target_path) if isinstance(target_path, str) else target_path
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


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
    def output_images_path(self) -> str:
        return self.parameters.storage.output_images

    @property
    def output_json_path(self) -> str:
        return self.parameters.storage.output_json

    def setup_paths(self) -> None:
        """Create all required directories if they don't exist."""
        paths = [
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

        Raises:
            ConfigurationError: If output_type is not 'images' or 'json'
        """
        if output_type not in ("images", "json"):
            raise ConfigurationError(
                f"Invalid output_type: {output_type}. Must be 'images' or 'json'"
            )

        if output_type == "images":
            base_path = self.output_images_path
        else:  # json
            base_path = self.output_json_path

        return Path(base_path) / filename


def _load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, "r") as f:
            return safe_load(f)
    except YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML config: {e}")
    except FileNotFoundError:
        raise ConfigurationError(f"Config file not found: {config_file}")


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
