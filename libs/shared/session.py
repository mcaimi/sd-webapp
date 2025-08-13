#!/usr/bin/env python

try:
    from streamlit import warning
except Exception as e:
    raise e

class Session(object):
    def __init__(self, session_state):
        self.streamlit_session = session_state
        self.session_state = self.streamlit_session

    def add_to_session_state(self, key, value) -> None:
        if key not in self.streamlit_session:
            setattr(self.streamlit_session, key, value)

    def remove_from_session_state(self, key) -> None:
        if key in self.streamlit_session:
            del self.streamlit_session[key]