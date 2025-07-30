#!/usr/bin/env python3
"""
LightRAG MCP Server - 简单查询接口
将 LightRAG 查询功能通过 MCP 协议暴露给 AI 模型
"""

import asyncio
import logging
import httpx
from mcp.server.fastmcp import FastMCP
from typing import Optional

# 设置日志,输出日志文件到 mcp.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/Users/Apple/dev/LightRAG/mcp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建 MCP 服务器
mcp = FastMCP("LightRAG")

# HTTP 客户端 - 强制绕过代理配置
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=10.0),
    verify=False,  # 跳过 SSL 验证（本地环境）
    follow_redirects=True,
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    # 注意：proxies={} 在 httpx.AsyncClient 中不被支持，会导致 TypeError
    # 使用 trust_env=False 是正确的代理绕过方式
    trust_env=False  # 忽略环境变量中的代理设置（关键修复）
)


@mcp.tool()
async def query_lightrag(
    query: str,
    mode: str = "hybrid",
    top_k: Optional[int] = None,
    enable_rerank: bool = False
) -> str:
    """
    查询 LightRAG 知识库
    
    从知识库中检索相关信息并生成回答。支持多种检索模式以适应不同类型的查询。
    
    Args:
        query (str): 用户的查询问题或关键词
        mode (str): 检索模式，可选值：
            - "local": 精确的实体级别信息检索，适合查找特定概念或实体
            - "global": 抽象的概念级别信息整合，适合总结性查询
            - "hybrid": 结合local和global的混合查询，平衡性能最佳（推荐）
            - "mix": 图谱检索与向量检索的融合，适合复杂推理查询
        top_k (int, optional): 返回的最大结果数量，用于控制响应长度
        enable_rerank (bool): 是否启用重排序优化，提高结果质量（默认关闭，提高响应速度）
    
    Returns:
        str: 基于知识库内容生成的回答
        
    Examples:
        - query_lightrag("什么是人工智能？", mode="global") 
        - query_lightrag("介绍一下GPT模型", mode="local", top_k=5)
        - query_lightrag("比较深度学习和机器学习", mode="hybrid")
    """
    try:
        logger.info(f"收到查询请求: {query[:50]}...")
        
        # 先进行健康检查 - 使用 IP 地址避免代理问题
        api_base_url = "http://127.0.0.1:8000"
        logger.info("检查 API 服务器健康状态...")
        try:
            health_response = await http_client.get(f"{api_base_url}/health", timeout=5)
            logger.info(f"健康检查响应: {health_response.status_code}")
        except Exception as health_error:
            logger.warning(f"健康检查失败: {health_error}")
        
        # 构建请求数据
        request_data = {
            "query": query,
            "mode": mode,
            "enable_rerank": enable_rerank
        }
        if top_k is not None:
            request_data["top_k"] = top_k
        
        logger.info(f"请求数据: {request_data}")
        logger.info("开始发送 HTTP 请求到 LightRAG API...")
        
        # 调用 LightRAG API，增加重试机制
        for attempt in range(3):  # 增加到3次重试
            try:
                logger.info(f"发送请求尝试 {attempt + 1}/3...")
                response = await http_client.post(
                    f"{api_base_url}/query",
                    json=request_data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "LightRAG-MCP-Client",
                        "Accept": "application/json"
                    },
                    timeout=30
                )
                logger.info(f"请求成功，状态码: {response.status_code}")
                break
            except Exception as request_error:
                logger.warning(f"查询请求尝试 {attempt + 1}/3 失败: {request_error}")
                logger.warning(f"错误类型: {type(request_error).__name__}")
                if attempt < 2:  # 不是最后一次尝试
                    await asyncio.sleep(3)  # 等待3秒后重试
                else:
                    raise
        
        logger.info(f"收到响应，状态码: {response.status_code}")
        logger.info(f"响应头: {dict(response.headers)}")
        
        # 详细检查响应状态
        if response.status_code != 200:
            response_text = ""
            try:
                response_text = response.text
                logger.error(f"HTTP 错误响应内容: {response_text}")
            except Exception as text_error:
                logger.error(f"无法读取错误响应文本: {text_error}")
            
            error_msg = f"❌ LightRAG API 错误 (状态码 {response.status_code}): {response_text}"
            logger.error(error_msg)
            return error_msg
        
        # 解析 JSON 响应
        try:
            result = response.json()
            logger.info("JSON 解析成功")
            logger.debug(f"响应结果预览: {str(result)[:200]}...")
        except Exception as json_error:
            logger.error(f"JSON 解析失败: {json_error}")
            logger.error(f"原始响应内容: {response.text}")
            return f"❌ JSON 解析失败: {json_error}"
        
        response_content = result.get("response", str(result))
        logger.info(f"查询完成，响应长度: {len(response_content)} 字符")
        return response_content
        
    except httpx.RequestError as e:
        error_msg = f"❌ 无法连接到 LightRAG 服务: {str(e)}"
        logger.error(error_msg)
        logger.error(f"请求错误详情: {type(e).__name__}: {e}")
        return error_msg
    except httpx.HTTPStatusError as e:
        # 这个异常现在不会被触发，因为我们手动检查状态码
        error_msg = f"❌ LightRAG API HTTP 错误: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ 查询过程出错: {str(e)}"
        logger.error(error_msg)
        logger.error(f"未知错误详情: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return error_msg


def main():
    """启动 MCP 服务器"""
    print("🚀 LightRAG MCP 服务器启动中...")
    print("📡 连接到: http://localhost:8000")
    print("🔍 提供 query_lightrag 查询工具")
    print("📋 日志级别: INFO")
    print("⚠️  如果遇到 JSON 解析错误，请检查 Claude Desktop 配置")
    print("=" * 50)
    
    # 创建启动标记文件
    import time
    import os
    with open("mcp_startup.log", "a") as f:
        f.write(f"MCP 进程启动: {time.strftime('%Y-%m-%d %H:%M:%S')} PID: {os.getpid()}\n")
    
    try:
        logger.info("MCP 服务器启动")
        mcp.run()
    except KeyboardInterrupt:
        print("\n👋 MCP 服务器已停止")
        logger.info("MCP 服务器停止")
    except Exception as e:
        print(f"\n❌ MCP 服务器启动失败: {e}")
        logger.error(f"MCP 服务器启动失败: {e}")
        raise


if __name__ == "__main__":
    main()