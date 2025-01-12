from groq import Groq
from typing import Generator
import requests
import streamlit as st
import os


def get_groq_api_key_from_secrets() -> str:
    """Get Groq API key from Streamlit secrets"""
    try:
        return st.secrets.groq_cloud["api_key"]
    except Exception as e:
        print(f"Error getting Groq API key: {e}")
        raise


def get_groq_api_key_from_env() -> str:
    """Get Groq API key"""
    try:
        return os.environ["GROQ_API_KEY"]
    except Exception as e:
        print(f"Error getting Groq API key: {e}")
        raise


def get_available_groq_models() -> dict:
    """Get list of available Groq models"""
    api_key = get_groq_api_key_from_secrets()
    url = "https://api.groq.com/openai/v1/models"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # from the response object construct a dict
        # all models are stored in response.json()['data']
        # dict should only contain list of dicts each dict has id, owned_by and context_window
        groq_response = response.json()
        models = dict()
        for model in groq_response["data"]:
            model_dict = {}
            model_dict["name"] = model["id"]
            model_dict["developer"] = model["owned_by"]
            model_dict["context_window"] = model["context_window"]
            models.update({model["id"]: model_dict})
        return models
    else:
        print(f"Error getting Groq models: {response.status_code}")
        raise


def get_groq_client() -> Groq:
    """Get Groq client"""
    try:
        client = Groq(api_key=get_groq_api_key_from_secrets())
    except Exception as e:
        print(f"Error creating Groq client: {e}")
        raise
    return client


def get_groq_responses_stream_for_streamlit(
    chat_completion,
) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def invoke_groq(
    model_name: str, prompt: str, max_tokens: int = 2048, stream: bool = False
):
    client = Groq(
        api_key=get_groq_api_key_from_env(),
    )
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=stream,
    )
    # chat_completion.choices[0].message.content
    return chat_completion.choices[0].message.content


def invoke_groq_stream(
    model_name: str, prompt: str, max_tokens: int = 2048, stream: bool = True
):
    """
    Streaming invocation of Groq API
    Prints the response in real-time and returns the complete response
    """
    client = Groq(
        api_key=get_groq_api_key_from_env(),
    )

    # Initialize an empty string to collect the full response
    full_response = ""

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=stream,
    )

    # Process the streaming response
    for chunk in chat_completion:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # Print immediately without newline
            full_response += content

    print()  # Add a newline at the end
    return full_response
