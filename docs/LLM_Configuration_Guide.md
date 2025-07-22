# LightRAG LLM配置指南

LightRAG支持多种LLM提供商，本文档详细介绍如何配置和使用不同的LLM服务。

## 支持的LLM提供商

### 1. 硅基流动 (SiliconFlow)

硅基流动提供高性价比的Qwen系列模型服务。

**配置示例**：
```bash
# .env文件配置
LLM_BINDING=siliconflow
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_BINDING_HOST=https://api.siliconflow.cn/v1
LLM_BINDING_API_KEY=sk-your-siliconflow-api-key

EMBEDDING_BINDING=siliconflow
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=https://api.siliconflow.cn/v1
EMBEDDING_BINDING_API_KEY=sk-your-siliconflow-api-key
```

**支持的模型**：
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- `BAAI/bge-m3` (embedding)

### 2. 智谱AI (ZhipuAI)

智谱AI提供GLM系列模型，支持中文对话和推理。

**配置示例**：
```bash
# .env文件配置
LLM_BINDING=zhipu
LLM_MODEL=glm-4-plus
LLM_BINDING_API_KEY=your-zhipu-api-key

EMBEDDING_BINDING=zhipu
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=1024
EMBEDDING_BINDING_API_KEY=your-zhipu-api-key
```

**支持的模型**：
- `glm-4-plus`
- `glm-4-flashx`
- `glm-4-air`
- `embedding-3` (embedding)

### 3. Google Gemini

Google的多模态大语言模型，支持文本和图像处理。

**配置示例**：
```bash
# .env文件配置
LLM_BINDING=gemini
LLM_MODEL=gemini-2.5-flash
LLM_BINDING_API_KEY=your-gemini-api-key

EMBEDDING_BINDING=gemini
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIM=768
EMBEDDING_BINDING_API_KEY=your-gemini-api-key
```

**支持的模型**：
- `gemini-2.5-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `text-embedding-004` (embedding)

### 4. OpenRouter

OpenRouter提供统一API访问多种模型，支持100+模型。

**配置示例**：
```bash
# .env文件配置
LLM_BINDING=openrouter
LLM_MODEL=openai/gpt-4o
LLM_BINDING_HOST=https://openrouter.ai/api/v1
LLM_BINDING_API_KEY=your-openrouter-api-key

EMBEDDING_BINDING=openrouter
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_BINDING_HOST=https://openrouter.ai/api/v1
EMBEDDING_BINDING_API_KEY=your-openrouter-api-key
```

**支持的模型**：
- `openai/gpt-4o`
- `anthropic/claude-3-sonnet`
- `meta-llama/llama-3.1-70b-instruct`
- `qwen/qwen-2.5-72b-instruct`
- `text-embedding-3-small` (embedding)

## 使用示例

### Python代码示例

```python
import os
from lightrag import LightRAG
from lightrag.llm.siliconcloud import siliconflow_complete, siliconflow_embed
from lightrag.utils import EmbeddingFunc

# 配置硅基流动
os.environ["LLM_BINDING_API_KEY"] = "your-api-key"

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await siliconflow_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

# 初始化LightRAG
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_model_func,
    llm_model_name="Qwen/Qwen2.5-7B-Instruct",
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: siliconflow_embed(
            texts,
            model="BAAI/bge-m3",
            api_key=os.environ["LLM_BINDING_API_KEY"]
        )
    )
)
```

### 命令行启动示例

```bash
# 使用硅基流动
lightrag-server \
  --llm-binding siliconflow \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-binding siliconflow \
  --embedding-model BAAI/bge-m3

# 使用智谱AI
lightrag-server \
  --llm-binding zhipu \
  --llm-model glm-4-plus \
  --embedding-binding zhipu \
  --embedding-model embedding-3

# 使用Google Gemini
lightrag-server \
  --llm-binding gemini \
  --llm-model gemini-2.5-flash \
  --embedding-binding gemini \
  --embedding-model text-embedding-004

# 使用OpenRouter
lightrag-server \
  --llm-binding openrouter \
  --llm-model openai/gpt-4o \
  --embedding-binding openrouter \
  --embedding-model text-embedding-3-small
```

## 配置优先级

LightRAG的配置优先级从高到低为：
1. **命令行参数** - 直接传递给程序的参数
2. **环境变量** - 系统环境变量或`.env`文件
3. **默认值** - 代码中定义的默认值

## 错误处理和重试

所有LLM实现都包含：
- **自动重试机制**：API失败时自动重试3次
- **指数退避**：重试间隔逐渐增加
- **错误分类**：区分不同类型的错误（限流、连接、认证等）
- **详细日志**：记录请求和响应详情

## 性能优化建议

1. **并发控制**：根据API限制调整`MAX_ASYNC`参数
2. **批处理**：embedding支持批量处理，提高效率
3. **缓存启用**：启用LLM缓存减少重复请求
4. **模型选择**：根据任务复杂度选择合适的模型

## 故障排除

### 常见问题

1. **API密钥错误**：检查环境变量设置
2. **网络连接问题**：检查防火墙和代理设置
3. **模型不存在**：确认模型名称正确
4. **配额超限**：检查API使用量和限制

### 调试方法

```bash
# 启用详细日志
export VERBOSE=true
export LOG_LEVEL=DEBUG

# 运行服务器
lightrag-server --verbose
```

## 扩展新的LLM提供商

如需添加新的LLM提供商，请参考现有实现：
1. 在`lightrag/llm/`目录创建新文件
2. 实现标准接口函数
3. 添加配置验证和错误处理
4. 更新服务器绑定选项
5. 创建使用示例和文档
