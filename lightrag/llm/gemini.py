import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("google-generativeai"):
    pm.install("google-generativeai")

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    logger,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Union
import os

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""
    pass


class GeminiError(Exception):
    """Generic error for issues related to Google Gemini"""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (GeminiError, InvalidResponseError)
    ),
)
async def gemini_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=None,
    api_key=None,
    stream=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using Google Gemini API with caching support.
    
    Args:
        model: Model name (e.g., "gemini-2.5-flash", "gemini-1.5-pro")
        prompt: The user prompt
        system_prompt: Optional system prompt
        history_messages: Optional conversation history
        api_key: Google Gemini API key
        stream: Whether to stream the response
        **kwargs: Additional parameters
        
    Returns:
        Generated text response or async iterator for streaming
    """
    if history_messages is None:
        history_messages = []
    
    # Get API key and configure client
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        raise ValueError("Google Gemini API key is required")

    genai.configure(api_key=api_key)
    
    # Remove LightRAG-specific kwargs
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    
    try:
        # Create model instance
        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Build conversation history
        chat_history = []
        for msg in history_messages:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        
        # Generation config
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
        }
        
        logger.debug(f"Gemini API request: model={model}, history_length={len(chat_history)}")
        
        if chat_history:
            # Continue existing chat
            chat = gemini_model.start_chat(history=chat_history)
            
            if stream:
                # Streaming response
                async def stream_generator():
                    response = chat.send_message(
                        prompt,
                        generation_config=generation_config,
                        stream=True
                    )
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                return stream_generator()
            else:
                # Non-streaming response
                response = chat.send_message(
                    prompt,
                    generation_config=generation_config,
                    stream=False
                )
        else:
            # Single message
            if stream:
                # Streaming response
                async def stream_generator():
                    response = gemini_model.generate_content(
                        prompt,
                        generation_config=generation_config,
                        stream=True
                    )
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                return stream_generator()
            else:
                # Non-streaming response
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=False
                )
        
        if not stream:
            if not response.text:
                raise InvalidResponseError("Empty response from Gemini")
            
            content = response.text
            logger.debug(f"Gemini response: {len(content)} characters")
            
            return content
            
    except Exception as e:
        logger.error(f"Gemini API request failed: {str(e)}")
        if "quota" in str(e).lower() or "rate limit" in str(e).lower():
            raise GeminiError(f"Gemini rate limit exceeded: {e}")
        elif "api key" in str(e).lower() or "authentication" in str(e).lower():
            raise GeminiError(f"Gemini authentication error: {e}")
        elif "safety" in str(e).lower():
            raise GeminiError(f"Gemini safety filter triggered: {e}")
        else:
            raise InvalidResponseError(f"Gemini API error: {e}")


# Generic Gemini completion function
async def gemini_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    """Generic Gemini completion function compatible with LightRAG interface"""
    if history_messages is None:
        history_messages = []
    
    # Handle keyword extraction
    if keyword_extraction:
        extraction_prompt = """Please extract keywords from the following text and return them in JSON format:
        {
            "high_level_keywords": ["keyword1", "keyword2"],
            "low_level_keywords": ["keyword1", "keyword2", "keyword3"]
        }
        Only return the JSON, no other text."""
        
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{extraction_prompt}"
        else:
            system_prompt = extraction_prompt
    
    # Get model name from global config
    hashing_kv = kwargs.get("hashing_kv")
    if hashing_kv and hasattr(hashing_kv, 'global_config'):
        model_name = hashing_kv.global_config.get("llm_model_name", "gemini-2.5-flash")
    else:
        model_name = "gemini-2.5-flash"
    
    result = await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    
    if keyword_extraction and isinstance(result, str):
        # Extract JSON from response for keyword extraction
        return locate_json_string_body_from_string(result)
    
    return result


# Embedding functions
@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (GeminiError, InvalidResponseError)
    ),
)
async def gemini_embed(
    texts: list[str],
    model: str = "text-embedding-004",
    api_key: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate embeddings using Google Gemini API.

    Args:
        texts: List of text strings to embed
        model: Embedding model name (e.g., "text-embedding-004")
        api_key: Google Gemini API key
        **kwargs: Additional parameters

    Returns:
        numpy array of embeddings
    """
    if not texts:
        return np.array([])

    # Configure client
    configure_gemini_client(api_key)

    logger.debug(f"Gemini embedding request: model={model}, texts_count={len(texts)}")
    verbose_debug(f"Texts to embed: {[text[:100] + '...' if len(text) > 100 else text for text in texts]}")

    try:
        embeddings = []

        # Process texts individually as Gemini embedding API doesn't support batch processing
        for text in texts:
            try:
                result = genai.embed_content(
                    model=f"models/{model}",
                    content=text,
                    task_type="retrieval_document",
                    **kwargs
                )

                if not result.get("embedding"):
                    raise InvalidResponseError(f"No embedding data for text: {text[:50]}...")

                embeddings.append(result["embedding"])

            except Exception as e:
                logger.error(f"Gemini embedding failed for text: {text[:50]}... Error: {str(e)}")
                raise

        result_array = np.array(embeddings, dtype=np.float32)

        logger.debug(f"Gemini embeddings generated: shape={result_array.shape}")
        verbose_debug(f"Embedding sample: {result_array[0][:5] if len(result_array) > 0 else 'empty'}")

        return result_array

    except Exception as e:
        logger.error(f"Gemini embedding request failed: {str(e)}")
        if "quota" in str(e).lower() or "rate limit" in str(e).lower():
            raise GeminiError(f"Gemini embedding rate limit exceeded: {e}")
        elif "api key" in str(e).lower() or "authentication" in str(e).lower():
            raise GeminiError(f"Gemini embedding authentication error: {e}")
        else:
            raise InvalidResponseError(f"Gemini embedding API error: {e}")


# Specific model functions
async def gemini_2_5_flash_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Gemini 2.5 Flash model completion"""
    if history_messages is None:
        history_messages = []

    # Force Gemini 2.5 Flash model
    kwargs["hashing_kv"] = kwargs.get("hashing_kv", {})
    if "global_config" not in kwargs["hashing_kv"]:
        kwargs["hashing_kv"]["global_config"] = {}
    kwargs["hashing_kv"]["global_config"]["llm_model_name"] = "gemini-2.5-flash"

    return await gemini_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


async def gemini_1_5_pro_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Gemini 1.5 Pro model completion"""
    if history_messages is None:
        history_messages = []

    # Force Gemini 1.5 Pro model
    kwargs["hashing_kv"] = kwargs.get("hashing_kv", {})
    if "global_config" not in kwargs["hashing_kv"]:
        kwargs["hashing_kv"]["global_config"] = {}
    kwargs["hashing_kv"]["global_config"]["llm_model_name"] = "gemini-1.5-pro"

    return await gemini_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )
