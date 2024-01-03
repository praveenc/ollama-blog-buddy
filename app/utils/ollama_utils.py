import ollama
from typing import List, Optional
from ollama import ResponseError, ChatResponse
from loguru import logger


# function to get embeddings from Ollama
def get_ollama_embeddings(
    texts: List[str],
    embed_model_id: str = "nomic-embed-text",
) -> str:
    try:
        embeddings = ollama.embed(
            model=embed_model_id,
            input=texts,
        )
        return embeddings["embeddings"]
    except ResponseError as e:
        logger.error(f"Ollama embeddings error: {e.error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error (embeddings): {str(e)}")
        raise


# function to get response from Ollama
def get_ollama_response(
    prompt: str,
    model_id: str = "llama3.2:3b-instruct-q8_0",
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """
    Function for synchronous invocation of Ollama models.

    Args:
        model_id (str): The Ollama model identifier
        max_tokens (int): Maximum number of tokens in the response
        temperature (float): Model temperature for response randomness

    Returns:
        str: Model response text
    """
    try:
        # prepare user message with messages api
        user_message = {"role": "user", "content": prompt}

        logger.info(f"Invoking ollama model: {model_id}")
        response: ChatResponse = ollama.chat(
            model=model_id,
            messages=[user_message],
            options={"num_predict": max_tokens, "temperature": temperature},
        )
        return response.message.content
    except ResponseError as e:
        logger.error(f"Ollama error: {e.error}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


def get_available_ollama_models(filter: Optional[str] = None) -> List[str]:
    """
    Get list of available Ollama models.
    Currently there no way to filter embedding models from text models
    """
    try:
        response = ollama.list()

        # Extract model names from the Model objects
        if filter is None:
            return [model.model for model in response.models]
        return [model.model for model in response.models if filter in model.model]
    except Exception as e:
        logger.error(f"Error getting Ollama models: {str(e)}")
        raise


# function to create a new Ollama model from existing model
def create_longcontext_model(
    base_model_id: str = "llama3.2", context_length: str = "8k"
):

    if context_length == "8k":
        num_ctx = 8192
    elif context_length == "4k":
        num_ctx = 4096
    else:
        num_ctx = 2048
    modelfile = f"""
FROM {base_model_id}
PARAMETER num_ctx {num_ctx}
PARAMETER top_p 0.95
PARAMETER temperature 0.1
"""
    model_name = f"{base_model_id}-{context_length}"
    try:
        response = ollama.create(model=model_name, modelfile=modelfile)
        return response["status"]
    except Exception as e:
        logger.error(f"Error creating Ollama model: {str(e)}")
        raise
