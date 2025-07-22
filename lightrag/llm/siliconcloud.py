import sys
import asyncio
import time

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
    before_sleep_log,
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


# 全局速率限制跟踪器
class RateLimitTracker:
    def __init__(self):
        self._last_rate_limit_time = 0
        self._consecutive_rate_limits = 0
        self._lock = asyncio.Lock()

    async def record_rate_limit(self):
        """记录速率限制事件"""
        async with self._lock:
            current_time = time.time()
            # 如果距离上次速率限制不到60秒，认为是连续的
            if current_time - self._last_rate_limit_time < 60:
                self._consecutive_rate_limits += 1
            else:
                self._consecutive_rate_limits = 1
            self._last_rate_limit_time = current_time

            # 根据连续速率限制次数动态调整等待时间
            if self._consecutive_rate_limits >= 3:
                wait_time = min(120, 30 * self._consecutive_rate_limits)
                logger.warning(f"连续 {self._consecutive_rate_limits} 次速率限制，等待 {wait_time} 秒")
                await asyncio.sleep(wait_time)

    async def reset_if_success(self):
        """成功请求后重置计数器"""
        async with self._lock:
            if self._consecutive_rate_limits > 0:
                logger.info("API 请求成功，重置速率限制计数器")
                self._consecutive_rate_limits = 0


# 全局实例
_rate_limit_tracker = RateLimitTracker()


@retry(
    stop=stop_after_attempt(5),  # 增加重试次数
    wait=wait_exponential(multiplier=2, min=10, max=120),  # 增加等待时间
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
    before_sleep=before_sleep_log(logger, "WARNING"),  # 添加重试前的日志
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
    stop=stop_after_attempt(5),  # 增加重试次数
    wait=wait_exponential(multiplier=2, min=10, max=120),  # 增加等待时间，最长2分钟
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
    before_sleep=before_sleep_log(logger, "WARNING"),  # 添加重试前的日志
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

                # 成功请求后重置速率限制计数器
                await _rate_limit_tracker.reset_if_success()

                return content

    except RateLimitError as e:
        logger.error(f"SiliconFlow rate limit exceeded: {str(e)}")
        logger.warning("SiliconFlow TPM 限制已达到，正在重试...")
        # 记录速率限制事件
        await _rate_limit_tracker.record_rate_limit()
        # 重新抛出原始的 RateLimitError，让 tenacity 处理重试
        raise e
    except APIConnectionError as e:
        logger.error(f"SiliconFlow connection error: {str(e)}")
        raise e
    except APITimeoutError as e:
        logger.error(f"SiliconFlow timeout error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"SiliconFlow API request failed: {str(e)}")
        # 检查是否是速率限制错误
        if "rate limit" in str(e).lower() or "429" in str(e):
            logger.warning("检测到速率限制错误，正在重试...")
            # 记录速率限制事件
            await _rate_limit_tracker.record_rate_limit()
            # 创建一个通用的异常来触发重试
            raise InvalidResponseError(f"Rate limit error: {e}")
        elif "connection" in str(e).lower():
            raise InvalidResponseError(f"Connection error: {e}")
        elif "timeout" in str(e).lower():
            raise InvalidResponseError(f"Timeout error: {e}")
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
    stop=stop_after_attempt(5),  # 增加重试次数
    wait=wait_exponential(multiplier=2, min=10, max=120),  # 增加等待时间
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
    before_sleep=before_sleep_log(logger, "WARNING"),  # 添加重试前的日志
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

            # 成功请求后重置速率限制计数器
            await _rate_limit_tracker.reset_if_success()

            return result

    except RateLimitError as e:
        logger.error(f"SiliconFlow embedding rate limit exceeded: {str(e)}")
        logger.warning("SiliconFlow embedding TPM 限制已达到，正在重试...")
        # 记录速率限制事件
        await _rate_limit_tracker.record_rate_limit()
        raise e
    except APIConnectionError as e:
        logger.error(f"SiliconFlow embedding connection error: {str(e)}")
        raise e
    except APITimeoutError as e:
        logger.error(f"SiliconFlow embedding timeout error: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"SiliconFlow embedding request failed: {str(e)}")
        # 检查是否是速率限制错误
        if "rate limit" in str(e).lower() or "429" in str(e):
            logger.warning("检测到 embedding 速率限制错误，正在重试...")
            # 记录速率限制事件
            await _rate_limit_tracker.record_rate_limit()
            raise InvalidResponseError(f"Embedding rate limit error: {e}")
        elif "connection" in str(e).lower():
            raise InvalidResponseError(f"Embedding connection error: {e}")
        elif "timeout" in str(e).lower():
            raise InvalidResponseError(f"Embedding timeout error: {e}")
        else:
            raise InvalidResponseError(f"SiliconFlow embedding API error: {e}")


# Keep the original embedding function for backward compatibility
# This is the original siliconcloud_embedding function that was already in the project
