#!/usr/bin/env python
"""
Application Settings - Legacy Compatibility Module

This module provides backward compatibility with the original Properties class.
New code should use libs.shared.config.get_app_config() instead.
"""

from yaml import safe_load, YAMLError

from libs.shared.parameters import Parameters
from libs.shared.console_utils import ANSIColors
from libs.shared.utils import check_or_create_path


class Properties:
    """
    Settings class that wraps configuration aspects of the application.
    
    Note: This is maintained for backward compatibility.
    New code should use get_app_config() from libs.shared.config.
    """
    
    def __init__(self, config_file: str) -> None:
        """
        Initialize Properties from a YAML configuration file.
        
        Args:
            config_file: Path to the YAML configuration file
            
        Raises:
            Exception: If configuration cannot be loaded
        """
        self.config_file_name = config_file
        self.tty_writer = ANSIColors()
        self.config_parameters: Parameters = None
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file_name, "r") as f:
                config_data = safe_load(f)
            
            self.config_parameters = Parameters(config_data)
        except YAMLError as e:
            self.tty_writer.print_error(str(e))
            raise
        except Exception as e:
            self.tty_writer.print_error(str(e))
            raise
    
    # Legacy method name alias
    def load_config_parms(self) -> None:
        """Load configuration (legacy alias for _load_config)."""
        self._load_config()
    
    def get_properties_object(self) -> Parameters:
        """Get the parameters object."""
        return self.config_parameters
    
    def setup_paths(self) -> None:
        """Create all required directories if they don't exist."""
        try:
            paths = [
                self.config_parameters.checkpoints.sd15.path,
                self.config_parameters.loras.sd15.path,
                self.config_parameters.checkpoints.sdxl.path,
                self.config_parameters.loras.sdxl.path,
                self.config_parameters.vae.sdxl.path,
                self.config_parameters.vae.sd15.path,
                self.config_parameters.storage.output_images,
                self.config_parameters.storage.output_json,
            ]
            for path in paths:
                check_or_create_path(path)
        except Exception as e:
            print(f"Properties.setup_paths: {e}")
    
    # Legacy attribute name
    @property
    def ttyWriter(self) -> ANSIColors:
        """Legacy alias for tty_writer."""
        return self.tty_writer
