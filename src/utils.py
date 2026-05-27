import os
import streamlit as st

def get_api_key(key_name):
    """
    Retrieves an API key from Streamlit secrets or environment variables.

    This function first attempts to fetch the requested key from Streamlit's
    secrets management system (`st.secrets`). If the key is not found there,
    it falls back to reading the value from environment variables via
    `os.getenv`.

    This allows the function to work seamlessly in both:
    - Streamlit Cloud / local Streamlit apps (using `st.secrets`)
    - Traditional Python environments (using `.env` or system environment variables)

    Args:
        key_name (str): The name of the secret key to retrieve (e.g., "TAVILY_API_KEY").

    Returns:
        str | None: The value of the API key if found, otherwise None.
    """
    
    return st.secrets.get(
        key_name,
        os.getenv(key_name)
)





