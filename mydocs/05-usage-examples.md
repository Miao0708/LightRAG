# LightRAG 使用示例

## 基础使用

### 1. 快速开始示例

```python
# quick_start.py
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import os

# 设置 API 密钥
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 定义 LLM 函数
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        model="gpt-4o-mini",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        **kwargs
    )

# 定义嵌入函数
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

# 初始化 LightRAG
rag = LightRAG(
    working_dir="./ragtest",
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=embedding_func
    )
)

# 插入文档
with open("book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# 查询
result = rag.query("What are the main themes?", param=QueryParam(mode="hybrid"))
print(result)
```

### 2. 流式查询示例

```python
# streaming_example.py
import asyncio

async def streaming_query_example():
    """流式查询示例"""
    query = "Explain the key concepts in detail"
    
    print("Streaming response:")
    async for chunk in rag.aquery_stream(
        query, 
        param=QueryParam(mode="hybrid")
    ):
        print(chunk, end="", flush=True)
    print("\n")

# 运行流式查询
asyncio.run(streaming_query_example())
```

## 高级配置示例

### 1. 多存储后端配置

```python
# multi_storage_example.py
from lightrag import LightRAG
from lightrag.storage import (
    PGKVStorage, PGVectorStorage, PGGraphStorage,
    Neo4JStorage, ChromaVectorDBStorage
)

# PostgreSQL + Neo4j + Chroma 组合
def create_advanced_rag():
    # PostgreSQL 配置
    pg_config = {
        "host": "localhost",
        "port": 5432,
        "database": "lightrag",
        "user": "lightrag",
        "password": "password"
    }
    
    # Neo4j 配置
    neo4j_config = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
    }
    
    # Chroma 配置
    chroma_config = {
        "persist_directory": "./chroma_db"
    }
    
    # 创建存储实例
    kv_storage = PGKVStorage(
        namespace="lightrag_kv",
        global_config=pg_config
    )
    
    vector_storage = ChromaVectorDBStorage(
        namespace="lightrag_vector",
        global_config=chroma_config
    )
    
    graph_storage = Neo4JStorage(
        namespace="lightrag_graph",
        global_config=neo4j_config
    )
    
    # 初始化 LightRAG
    rag = LightRAG(
        working_dir="./ragtest",
        kv_storage=kv_storage,
        vector_storage=vector_storage,
        graph_storage=graph_storage,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func
    )
    
    return rag

# 使用高级配置
advanced_rag = create_advanced_rag()
```

### 2. 自定义配置示例

```python
# custom_config_example.py
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# 自定义配置
custom_config = {
    # 文本分块配置
    "chunk_token_size": 800,
    "chunk_overlap_token_size": 200,
    "tiktoken_model_name": "gpt-4o",
    
    # 实体提取配置
    "entity_extract_max_gleaning": 2,
    "entity_summary_to_max_tokens": 300,
    
    # 关系提取配置
    "relation_max_gleaning": 1,
    "relationship_summary_to_max_tokens": 200,
    
    # 性能配置
    "max_async": 8,
    "max_token_for_text_unit": 4000,
    "max_token_for_global_context": 8000,
    "max_token_for_local_context": 4000,
}

# 应用自定义配置
rag = LightRAG(
    working_dir="./ragtest",
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    **custom_config
)
```

## 不同检索模式示例

### 1. 检索模式对比

```python
# retrieval_modes_example.py
import asyncio

async def compare_retrieval_modes():
    """对比不同检索模式的效果"""
    query = "What are the relationships between key concepts?"
    
    modes = ["naive", "local", "global", "hybrid", "mix"]
    
    for mode in modes:
        print(f"\n=== {mode.upper()} MODE ===")
        result = await rag.aquery(
            query, 
            param=QueryParam(mode=mode)
        )
        print(f"Result length: {len(result)}")
        print(f"First 200 chars: {result[:200]}...")

# 运行对比
asyncio.run(compare_retrieval_modes())
```

### 2. 混合检索优化

```python
# hybrid_retrieval_example.py
from lightrag import QueryParam

def optimized_hybrid_query(question: str, context_type: str = "balanced"):
    """优化的混合检索查询"""
    
    # 根据问题类型选择参数
    if context_type == "detailed":
        param = QueryParam(
            mode="mix",
            only_need_context=False,
            response_type="Multiple Paragraphs"
        )
    elif context_type == "summary":
        param = QueryParam(
            mode="global",
            only_need_context=False,
            response_type="Single Paragraph"
        )
    else:  # balanced
        param = QueryParam(
            mode="hybrid",
            only_need_context=False,
            response_type="Multiple Paragraphs"
        )
    
    return rag.query(question, param=param)

# 使用示例
detailed_answer = optimized_hybrid_query(
    "Explain the technical architecture in detail",
    context_type="detailed"
)

summary_answer = optimized_hybrid_query(
    "Give me a brief overview",
    context_type="summary"
)
```

## 批量处理示例

### 1. 批量文档插入

```python
# batch_processing_example.py
import os
import asyncio
from pathlib import Path

async def batch_insert_documents(doc_directory: str):
    """批量插入文档"""
    doc_path = Path(doc_directory)
    
    # 支持的文件格式
    supported_formats = ['.txt', '.md', '.pdf', '.docx']
    
    # 收集所有文档
    documents = []
    for format_ext in supported_formats:
        documents.extend(doc_path.glob(f"**/*{format_ext}"))
    
    print(f"Found {len(documents)} documents to process")
    
    # 批量处理
    batch_size = 5
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        # 并行处理批次
        tasks = []
        for doc_path in batch:
            task = process_single_document(doc_path)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        print(f"Batch {i//batch_size + 1} completed")

async def process_single_document(doc_path: Path):
    """处理单个文档"""
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        await rag.ainsert(content)
        print(f"✓ Processed: {doc_path.name}")
        
    except Exception as e:
        print(f"✗ Error processing {doc_path.name}: {e}")

# 运行批量处理
asyncio.run(batch_insert_documents("./documents"))
```

### 2. 批量查询示例

```python
# batch_query_example.py
import asyncio
import json

async def batch_query_processing():
    """批量查询处理"""
    
    # 查询列表
    queries = [
        "What are the main concepts?",
        "How do the systems interact?",
        "What are the key benefits?",
        "What are the limitations?",
        "How to implement this solution?"
    ]
    
    results = {}
    
    # 并行处理查询
    tasks = []
    for query in queries:
        task = process_single_query(query)
        tasks.append(task)
    
    query_results = await asyncio.gather(*tasks)
    
    # 整理结果
    for query, result in zip(queries, query_results):
        results[query] = result
    
    # 保存结果
    with open("batch_query_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

async def process_single_query(query: str):
    """处理单个查询"""
    try:
        result = await rag.aquery(
            query, 
            param=QueryParam(mode="hybrid")
        )
        print(f"✓ Completed query: {query[:50]}...")
        return result
    except Exception as e:
        print(f"✗ Error in query '{query[:50]}...': {e}")
        return None

# 运行批量查询
results = asyncio.run(batch_query_processing())
```

## Web API 集成示例

### 1. FastAPI 集成

```python
# fastapi_integration.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="LightRAG API", version="1.0.0")

# 请求模型
class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    stream: bool = False

class InsertRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

# 响应模型
class QueryResponse(BaseModel):
    answer: str
    mode: str
    processing_time: float

# 初始化 LightRAG（假设已配置）
# rag = LightRAG(...)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """查询端点"""
    try:
        import time
        start_time = time.time()
        
        result = await rag.aquery(
            request.question,
            param=QueryParam(mode=request.mode)
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=result,
            mode=request.mode,
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """流式查询端点"""
    from fastapi.responses import StreamingResponse
    
    async def generate_stream():
        try:
            async for chunk in rag.aquery_stream(
                request.question,
                param=QueryParam(mode=request.mode)
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain"
    )

@app.post("/insert")
async def insert_endpoint(request: InsertRequest):
    """文档插入端点"""
    try:
        await rag.ainsert(request.content)
        return {"status": "success", "message": "Document inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "LightRAG API"}

# 启动命令: uvicorn fastapi_integration:app --reload
```

### 2. 聊天机器人示例

```python
# chatbot_example.py
import asyncio
from typing import List, Dict

class LightRAGChatbot:
    def __init__(self, rag_instance):
        self.rag = rag_instance
        self.conversation_history: List[Dict] = []
    
    async def chat(self, user_input: str, mode: str = "hybrid") -> str:
        """聊天功能"""
        
        # 构建上下文查询
        context_query = self._build_context_query(user_input)
        
        # 获取相关信息
        response = await self.rag.aquery(
            context_query,
            param=QueryParam(mode=mode)
        )
        
        # 记录对话历史
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "mode": mode
        })
        
        # 保持历史记录在合理范围内
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def _build_context_query(self, user_input: str) -> str:
        """构建包含上下文的查询"""
        if not self.conversation_history:
            return user_input
        
        # 获取最近的对话上下文
        recent_context = self.conversation_history[-3:]
        context_str = ""
        
        for item in recent_context:
            context_str += f"User: {item['user']}\nAssistant: {item['assistant'][:200]}...\n\n"
        
        return f"Previous conversation context:\n{context_str}\nCurrent question: {user_input}"
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []

# 使用示例
async def chatbot_demo():
    chatbot = LightRAGChatbot(rag)
    
    print("LightRAG Chatbot started! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            print("Conversation history cleared.")
            continue
        
        print("Assistant: ", end="")
        response = await chatbot.chat(user_input)
        print(response)

# 运行聊天机器人
# asyncio.run(chatbot_demo())
```

## 性能监控示例

### 1. 性能指标收集

```python
# performance_monitoring.py
import time
import psutil
import asyncio
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    query_time: float
    memory_usage: float
    cpu_usage: float
    token_count: int
    mode: str

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    async def monitored_query(self, rag, query: str, mode: str = "hybrid"):
        """监控查询性能"""
        
        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        start_cpu = psutil.cpu_percent()
        
        # 执行查询
        result = await rag.aquery(
            query,
            param=QueryParam(mode=mode)
        )
        
        # 记录结束状态
        end_time = time.time()
        end_memory = psutil.virtual_memory().percent
        end_cpu = psutil.cpu_percent()
        
        # 计算指标
        metrics = PerformanceMetrics(
            query_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=end_cpu - start_cpu,
            token_count=len(result.split()),
            mode=mode
        )
        
        self.metrics.append(metrics)
        
        return result, metrics
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.metrics:
            return {}
        
        avg_query_time = sum(m.query_time for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m.memory_usage for m in self.metrics) / len(self.metrics)
        avg_cpu = sum(m.cpu_usage for m in self.metrics) / len(self.metrics)
        
        return {
            "total_queries": len(self.metrics),
            "average_query_time": avg_query_time,
            "average_memory_usage": avg_memory,
            "average_cpu_usage": avg_cpu,
            "mode_distribution": self._get_mode_distribution()
        }
    
    def _get_mode_distribution(self) -> Dict[str, int]:
        """获取检索模式分布"""
        distribution = {}
        for metric in self.metrics:
            distribution[metric.mode] = distribution.get(metric.mode, 0) + 1
        return distribution

# 使用示例
async def performance_test():
    monitor = PerformanceMonitor()
    
    test_queries = [
        "What are the main concepts?",
        "How do systems interact?",
        "What are the benefits?",
    ]
    
    for query in test_queries:
        result, metrics = await monitor.monitored_query(rag, query, "hybrid")
        print(f"Query: {query[:30]}...")
        print(f"Time: {metrics.query_time:.2f}s, Memory: {metrics.memory_usage:.1f}%")
    
    # 打印性能摘要
    summary = monitor.get_performance_summary()
    print("\nPerformance Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

# 运行性能测试
# asyncio.run(performance_test())
```

这些示例涵盖了 LightRAG 的各种使用场景，从基础操作到高级集成，帮助用户快速掌握系统的使用方法。
