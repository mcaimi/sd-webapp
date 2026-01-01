#!/usr/bin/env python
"""
Console Output Utilities

Provides ANSI color formatting for console output, including
colored text, error messages, warnings, and success messages.
"""

from typing import Dict, Optional


class ANSIColors:
    """
    ANSI color escape code manager for colored console output.
    
    Supports various colors with bright/normal intensity modifiers.
    """
    
    # Color codes (30-37 for foreground)
    COLORS: Dict[str, int] = {
        'BLACK': 0,
        'RED': 1,
        'GREEN': 2,
        'YELLOW': 3,
        'BLUE': 4,
        'PURPLE': 5,
        'CYAN': 6,
        'WHITE': 7,
    }
    
    # Intensity modifiers
    MODIFIERS: Dict[str, int] = {
        'NORMAL': 0,
        'BRIGHT': 1,
    }
    
    def __init__(self):
        """Initialize the ANSI colors lookup table."""
        self._escape_code = '\033[%s;%sm'
        self._reset_code = '\033[0m'
        self._build_color_table()
    
    def _build_color_table(self) -> None:
        """Build the lookup table of ANSI escape sequences."""
        self.ansi_escapes: Dict[str, Dict[str, str]] = {}
        
        for intensity, intensity_value in self.MODIFIERS.items():
            self.ansi_escapes[intensity] = {}
            for color, color_value in self.COLORS.items():
                self.ansi_escapes[intensity][color] = (
                    self._escape_code % (intensity_value, 30 + color_value)
                )
        
        self.ansi_escapes['RESET'] = self._reset_code
    
    # Legacy method names for backward compatibility
    def compile_ansicolors_hash(self) -> None:
        """Legacy alias for _build_color_table."""
        self._build_color_table()
    
    def get_ansicolors_hash(self) -> Dict:
        """Get the ANSI escape codes lookup table."""
        return self.ansi_escapes
    
    def get_code_for_color(
        self,
        modifier: str = "NORMAL",
        color: str = "WHITE"
    ) -> Optional[str]:
        """
        Get the escape sequence for a specific color and intensity.
        
        Args:
            modifier: Intensity modifier ('NORMAL' or 'BRIGHT')
            color: Color name (e.g., 'RED', 'GREEN', 'BLUE')
            
        Returns:
            ANSI escape sequence string, or None if invalid
        """
        if modifier in self.ansi_escapes:
            if isinstance(self.ansi_escapes[modifier], dict):
                return self.ansi_escapes[modifier].get(color)
        return None
    
    def color_write(
        self,
        text: str,
        modifier: str = "NORMAL",
        color: str = "WHITE"
    ) -> str:
        """
        Generate a colored string.
        
        Args:
            text: Text to colorize
            modifier: Intensity modifier
            color: Color name
            
        Returns:
            String with ANSI escape codes for coloring
        """
        color_code = self.get_code_for_color(modifier, color)
        if color_code:
            return f"{color_code}{text}{self._reset_code}"
        return text
    
    def color_print(
        self,
        text: str,
        modifier: str = "NORMAL",
        color: str = "WHITE"
    ) -> None:
        """Print colored text to console."""
        print(self.color_write(text, modifier, color))
    
    def error(self, text: str) -> str:
        """Generate an error string (bright red)."""
        return self.color_write(text, 'BRIGHT', 'RED')
    
    def print_error(self, text: str) -> None:
        """Print an error message (bright red)."""
        print(self.error(text))
    
    def warning(self, text: str) -> str:
        """Generate a warning string (bright yellow)."""
        return self.color_write(text, 'BRIGHT', 'YELLOW')
    
    def print_warning(self, text: str) -> None:
        """Print a warning message (bright yellow)."""
        print(self.warning(text))
    
    def success(self, text: str) -> str:
        """Generate a success string (bright green)."""
        return self.color_write(text, 'BRIGHT', 'GREEN')
    
    def print_success(self, text: str) -> None:
        """Print a success message (bright green)."""
        print(self.success(text))
