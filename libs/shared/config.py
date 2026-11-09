#!/usr/bin/env python
"""
Centralized Application Configuration

Provides a single point of configuration loading and caching for the entire app.
All pages should import from here instead of loading config independently.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any

from dotenv import dotenv_values
from yaml import safe_load, YAMLError

from libs.shared.parameters import Parameters
from libs.shared.exceptions import ConfigurationError


DEFAULT_API_URL = "http://localhost:8000"


@dataclass
class AppConfig:
    """
    Centralized application configuration container.

    Provides access to all configuration parameters.
    """

    env: Dict[str, str]
    config_file: str
    parameters: Parameters

    @property
    def api_url(self) -> str:
        """
        Base URL of the Stable Diffusion API backend.

        Resolution order: SD_API_URL process env var, SD_API_URL in .env,
        api.url in the YAML config, then DEFAULT_API_URL.
        """
        yaml_api = getattr(self.parameters, "api", None)
        return (
            os.environ.get("SD_API_URL")
            or self.env.get("SD_API_URL")
            or getattr(yaml_api, "url", None)
            or DEFAULT_API_URL
        )


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
    config_dict = _load_yaml_config(config_file) or {}
    parameters = Parameters(config_dict)

    # Create and return config object
    return AppConfig(
        env=dict(env),
        config_file=config_file,
        parameters=parameters,
    )


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing or reloading)."""
    get_app_config.cache_clear()
