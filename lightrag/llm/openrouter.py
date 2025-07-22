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
    APIStatusError,
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
from lightrag.api import __api_version__

import numpy as np
from typing import Union, Any
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
    stop=stop_after_attempt(5),  # 增加重试次数
    wait=wait_exponential(multiplier=2, min=15, max=120),  # 增加等待时间，最长2分钟
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def openrouter_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=None,
    base_url="https://openrouter.ai/api/v1",
    api_key=None,
    stream=False,
    site_url=None,
    site_name=None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """Complete a prompt using OpenRouter's unified API with caching support.
    
    Args:
        model: Model name (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet")
        prompt: The user prompt
        system_prompt: Optional system prompt
        history_messages: Optional conversation history
        base_url: OpenRouter API base URL
        api_key: OpenRouter API key
        stream: Whether to stream the response
        site_url: Optional site URL for rankings
        site_name: Optional site name for rankings
        **kwargs: Additional parameters
        
    Returns:
        Generated text response or async iterator for streaming
    """
    if history_messages is None:
        history_messages = []
    
    # Remove LightRAG-specific kwargs
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    
    # Get API key
    api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key is required")

    # OpenRouter-specific headers
    headers = {
        "User-Agent": f"LightRAG/{__api_version__}",
    }

    # Optional headers for rankings on openrouter.ai
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    # Create client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=headers,
    )
    
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
        **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "site_url", "site_name"]}
    }
    
    logger.debug(f"OpenRouter API request: model={model}, messages_count={len(messages)}")
    
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
                
                logger.debug(f"OpenRouter response: {len(content)} characters")
                
                return content
                
    except RateLimitError as e:
        logger.error(f"OpenRouter rate limit exceeded: {str(e)}")
        logger.warning("免费模型 qwen/qwen3-235b-a22b-07-25:free 每分钟限制5次请求")
        logger.info("建议：1) 等待1分钟后重试 2) 考虑使用付费模型 3) 降低并发数")
        # 重新抛出原始的 RateLimitError
        raise e
    except APIConnectionError as e:
        logger.error(f"OpenRouter connection error: {str(e)}")
        raise e
    except APITimeoutError as e:
        logger.error(f"OpenRouter timeout error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"OpenRouter API request failed: {str(e)}")
        # 对于其他错误，创建一个通用的异常
        raise Exception(f"OpenRouter API error: {e}")


# Generic OpenRouter completion function
async def openrouter_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, model=None, api_key=None, base_url=None, **kwargs
) -> Union[str, AsyncIterator[str]]:
    """Generic OpenRouter completion function compatible with LightRAG interface"""
    if history_messages is None:
        history_messages = []

    # Handle keyword extraction
    if keyword_extraction:
        kwargs["response_format"] = {"type": "json_object"}

    # Get model name from parameter, global config, or default
    if model:
        model_name = model
    else:
        hashing_kv = kwargs.get("hashing_kv")
        if hashing_kv and hasattr(hashing_kv, 'global_config'):
            model_name = hashing_kv.global_config.get("llm_model_name", "openai/gpt-4o-mini")
        else:
            model_name = "openai/gpt-4o-mini"

    # Remove conflicting parameters from kwargs to avoid duplication
    conflicting_params = [
        "model", "api_key", "base_url", "_priority", "hashing_kv",
        "cache_type", "mode", "keyword_extraction"
    ]
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in conflicting_params}

    result = await openrouter_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **clean_kwargs,
    )

    if keyword_extraction and isinstance(result, str):
        # Extract JSON from response for keyword extraction
        return locate_json_string_body_from_string(result)

    return result


# OpenRouter doesn't provide embedding services directly
# But we can provide a wrapper that uses OpenAI embeddings through OpenRouter
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
)
async def openrouter_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate embeddings using OpenRouter (Note: OpenRouter may not support all embedding models).
    
    Args:
        texts: List of text strings to embed
        model: Embedding model name (e.g., "text-embedding-3-small")
        base_url: OpenRouter API base URL
        api_key: OpenRouter API key
        **kwargs: Additional parameters
        
    Returns:
        numpy array of embeddings
    """
    if not texts:
        return np.array([])
    
    # Create client
    client = create_openrouter_client(api_key=api_key, base_url=base_url)
    
    logger.debug(f"OpenRouter embedding request: model={model}, texts_count={len(texts)}")
    verbose_debug(f"Texts to embed: {[text[:100] + '...' if len(text) > 100 else text for text in texts]}")
    
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
            
            logger.debug(f"OpenRouter embeddings generated: shape={result.shape}")
            verbose_debug(f"Embedding sample: {result[0][:5] if len(result) > 0 else 'empty'}")
            
            return result
            
    except Exception as e:
        logger.error(f"OpenRouter embedding request failed: {str(e)}")
        if "rate limit" in str(e).lower():
            raise RateLimitError(f"OpenRouter embedding rate limit exceeded: {e}")
        elif "connection" in str(e).lower():
            raise APIConnectionError(f"OpenRouter embedding connection error: {e}")
        elif "timeout" in str(e).lower():
            raise APITimeoutError(f"OpenRouter embedding timeout error: {e}")
        else:
            raise InvalidResponseError(f"OpenRouter embedding API error: {e}")


# Specific model functions for popular models on OpenRouter
async def openai_gpt4_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """GPT-4 model completion through OpenRouter"""
    if history_messages is None:
        history_messages = []

    # Force GPT-4 model
    kwargs["hashing_kv"] = kwargs.get("hashing_kv", {})
    if "global_config" not in kwargs["hashing_kv"]:
        kwargs["hashing_kv"]["global_config"] = {}
    kwargs["hashing_kv"]["global_config"]["llm_model_name"] = "openai/gpt-4o"

    return await openrouter_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


async def claude_3_sonnet_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Claude 3 Sonnet model completion through OpenRouter"""
    if history_messages is None:
        history_messages = []

    # Force Claude 3 Sonnet model
    kwargs["hashing_kv"] = kwargs.get("hashing_kv", {})
    if "global_config" not in kwargs["hashing_kv"]:
        kwargs["hashing_kv"]["global_config"] = {}
    kwargs["hashing_kv"]["global_config"]["llm_model_name"] = "anthropic/claude-3-sonnet"

    return await openrouter_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


async def llama_3_1_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Llama 3.1 model completion through OpenRouter"""
    if history_messages is None:
        history_messages = []

    # Force Llama 3.1 model
    kwargs["hashing_kv"] = kwargs.get("hashing_kv", {})
    if "global_config" not in kwargs["hashing_kv"]:
        kwargs["hashing_kv"]["global_config"] = {}
    kwargs["hashing_kv"]["global_config"]["llm_model_name"] = "meta-llama/llama-3.1-70b-instruct"

    return await openrouter_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


async def qwen_2_5_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Qwen 2.5 model completion through OpenRouter"""
    if history_messages is None:
        history_messages = []

    # Force Qwen 2.5 model
    kwargs["hashing_kv"] = kwargs.get("hashing_kv", {})
    if "global_config" not in kwargs["hashing_kv"]:
        kwargs["hashing_kv"]["global_config"] = {}
    kwargs["hashing_kv"]["global_config"]["llm_model_name"] = "qwen/qwen-2.5-72b-instruct"

    return await openrouter_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )
