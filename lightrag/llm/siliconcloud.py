import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
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

import numpy as np
from typing import Union
import aiohttp
import base64
import struct
import os

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def siliconcloud_embedding(
    texts: list[str],
    model: str = "netease-youdao/bce-embedding-base_v1",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 512,
    api_key: str = None,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key

    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}

    base64_strings = []
    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            content = await response.json()
            if "code" in content:
                raise ValueError(content)
            base64_strings = [item["embedding"] for item in content["data"]]

    embeddings = []
    for string in base64_strings:
        decode_bytes = base64.b64decode(string)
        n = len(decode_bytes) // 4
        float_array = struct.unpack("<" + "f" * n, decode_bytes)
        embeddings.append(float_array)
    return np.array(embeddings)


# LLM text generation functions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
)
async def siliconflow_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=None,
    base_url="https://api.siliconflow.cn/v1",
    api_key=None,
    stream=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []

    # Remove LightRAG-specific kwargs
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    # Get API key
    api_key = api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        raise ValueError("SiliconFlow API key is required")

    # Create client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Set default parameters
    completion_kwargs = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 2048),
        **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
    }

    logger.debug(f"SiliconFlow API request: model={model}, messages_count={len(messages)}")

    try:
        if stream:
            # Return async iterator for streaming
            async def stream_generator():
                async with client:
                    stream_response = await client.chat.completions.create(**completion_kwargs)
                    async for chunk in stream_response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
            return stream_generator()
        else:
            # Non-streaming response
            async with client:
                response = await client.chat.completions.create(**completion_kwargs)

                if not response.choices:
                    raise InvalidResponseError("No choices in response")

                content = response.choices[0].message.content
                if content is None:
                    raise InvalidResponseError("Empty response content")

                logger.debug(f"SiliconFlow response: {len(content)} characters")

                return content

    except Exception as e:
        logger.error(f"SiliconFlow API request failed: {str(e)}")
        if "rate limit" in str(e).lower():
            raise RateLimitError(f"SiliconFlow rate limit exceeded: {e}")
        elif "connection" in str(e).lower():
            raise APIConnectionError(f"SiliconFlow connection error: {e}")
        elif "timeout" in str(e).lower():
            raise APITimeoutError(f"SiliconFlow timeout error: {e}")
        else:
            raise InvalidResponseError(f"SiliconFlow API error: {e}")


async def siliconflow_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    if keyword_extraction:
        kwargs["response_format"] = {"type": "json_object"}

    # Get model name from global config
    hashing_kv = kwargs.get("hashing_kv")
    if hashing_kv and hasattr(hashing_kv, 'global_config'):
        model_name = hashing_kv.global_config.get("llm_model_name", "Qwen/Qwen2.5-7B-Instruct")
    else:
        model_name = "Qwen/Qwen2.5-7B-Instruct"

    result = await siliconflow_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

    if keyword_extraction and isinstance(result, str):
        return locate_json_string_body_from_string(result)

    return result


# Enhanced embedding function with OpenAI-compatible API
@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
)
async def siliconflow_embed(
    texts, model="BAAI/bge-m3", base_url="https://api.siliconflow.cn/v1", api_key=None, **kwargs
) -> np.ndarray:
    if not texts:
        return np.array([])

    # Get API key
    api_key = api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        raise ValueError("SiliconFlow API key is required")

    # Create client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    logger.debug(f"SiliconFlow embedding request: model={model}, texts_count={len(texts)}")

    try:
        async with client:
            response = await client.embeddings.create(
                model=model,
                input=texts,
                encoding_format="float",
                **kwargs
            )

            if not response.data:
                raise InvalidResponseError("No embedding data in response")

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            result = np.array(embeddings, dtype=np.float32)

            logger.debug(f"SiliconFlow embeddings generated: shape={result.shape}")

            return result

    except Exception as e:
        logger.error(f"SiliconFlow embedding request failed: {str(e)}")
        if "rate limit" in str(e).lower():
            raise RateLimitError(f"SiliconFlow embedding rate limit exceeded: {e}")
        elif "connection" in str(e).lower():
            raise APIConnectionError(f"SiliconFlow embedding connection error: {e}")
        elif "timeout" in str(e).lower():
            raise APITimeoutError(f"SiliconFlow embedding timeout error: {e}")
        else:
            raise InvalidResponseError(f"SiliconFlow embedding API error: {e}")


# Keep the original embedding function for backward compatibility
# This is the original siliconcloud_embedding function that was already in the project
