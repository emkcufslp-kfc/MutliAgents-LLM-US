import os

from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str, default=None):
    value = os.environ.get(name)
    if value not in (None, ""):
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            secret_value = st.secrets[name]
            if secret_value not in (None, ""):
                return secret_value
    except Exception:
        pass

    return default
