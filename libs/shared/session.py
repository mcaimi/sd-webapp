#!/usr/bin/env python
"""
Streamlit Session State Wrapper

Provides a wrapper class for Streamlit's session state management.
"""

from typing import Any


class Session:
    """Wrapper for Streamlit session state management."""

    def __init__(self, session_state: Any) -> None:
        """
        Initialize the session wrapper.

        Args:
            session_state: Streamlit SessionState object
        """
        self.streamlit_session: Any = session_state
        self.session_state: Any = self.streamlit_session

    def add_to_session_state(self, key: str, value: Any) -> None:
        """
        Add a key-value pair to the session state if key doesn't exist.

        Args:
            key: Session state key
            value: Value to store
        """
        if key not in self.streamlit_session:
            setattr(self.streamlit_session, key, value)

    def remove_from_session_state(self, key: str) -> None:
        """
        Remove a key from the session state if it exists.

        Args:
            key: Session state key to remove
        """
        if key in self.streamlit_session:
            del self.streamlit_session[key]

