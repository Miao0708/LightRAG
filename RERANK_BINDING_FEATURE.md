# RERANK_BINDING 功能说明

## 🎯 功能概述

为 LightRAG 添加了 `RERANK_BINDING` 配置项，使其可以像 `LLM_BINDING` 和 `EMBEDDING_BINDING` 一样直接通过配置指定 rerank 提供商，而不需要手动编写 rerank 函数。

## 🔧 配置方式

### 环境变量配置

在 `.env` 文件中添加以下配置：

```bash
# Rerank 配置
ENABLE_RERANK=true
RERANK_BINDING=siliconflow          # 新增：指定 rerank 提供商
RERANK_MODEL=Qwen/Qwen3-Reranker-8B
RERANK_BINDING_HOST=https://api.siliconflow.cn/v1
RERANK_BINDING_API_KEY=your_api_key_here
```

### 支持的 RERANK_BINDING 选项

| 绑定值 | 提供商 | 说明 |
|--------|--------|------|
| `siliconflow` | SiliconFlow (硅基流动) | 自动添加 `/rerank` 端点 |
| `jina` | Jina AI | 使用 Jina rerank API |
| `cohere` | Cohere | 使用 Cohere rerank API |
| `custom` | 自定义 | 通用 rerank API (默认) |

## 📝 使用示例

### 1. SiliconFlow 配置

```bash
RERANK_BINDING=siliconflow
RERANK_MODEL=Qwen/Qwen3-Reranker-8B
RERANK_BINDING_HOST=https://api.siliconflow.cn/v1
RERANK_BINDING_API_KEY=sk-xxx
```

### 2. Jina AI 配置

```bash
RERANK_BINDING=jina
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=https://api.jina.ai/v1/rerank
RERANK_BINDING_API_KEY=jina_xxx
```

### 3. Cohere 配置

```bash
RERANK_BINDING=cohere
RERANK_MODEL=rerank-english-v2.0
RERANK_BINDING_HOST=https://api.cohere.ai/v1/rerank
RERANK_BINDING_API_KEY=cohere_xxx
```

### 4. 自定义 API 配置

```bash
RERANK_BINDING=custom
RERANK_MODEL=your-model
RERANK_BINDING_HOST=https://your-api.com/v1/rerank
RERANK_BINDING_API_KEY=your_key
```

## 🔄 自动处理逻辑

### SiliconFlow 特殊处理
- 自动检测 API 端点是否以 `/rerank` 结尾
- 如果没有，自动添加 `/rerank` 路径
- 例：`https://api.siliconflow.cn/v1` → `https://api.siliconflow.cn/v1/rerank`

### 其他提供商
- 直接使用配置的 `RERANK_BINDING_HOST`
- 调用对应的专用 rerank 函数

## 🚀 启动服务器

配置完成后，直接启动 LightRAG 服务器：

```bash
python -m lightrag.api.lightrag_server
```

服务器会自动根据 `RERANK_BINDING` 配置选择合适的 rerank 函数。

## 📊 测试验证

使用提供的测试脚本验证配置：

```bash
# 测试所有模型（包括 rerank）
python test_env_models.py

# 查看配置总结
python test_summary.py
```

## 🔍 日志输出

启动时会显示 rerank 配置信息：

```
INFO - Rerank model configured: siliconflow/Qwen/Qwen3-Reranker-8B (can be enabled per query)
```

## 💡 优势

1. **统一配置方式**：与 LLM 和 Embedding 配置保持一致
2. **自动适配**：根据提供商自动选择合适的 API 调用方式
3. **向下兼容**：不影响现有的手动 rerank 函数配置
4. **易于切换**：只需修改 `RERANK_BINDING` 即可切换提供商

## 🔧 代码修改

### 修改的文件

1. `lightrag/api/config.py` - 添加 `--rerank-binding` 参数
2. `lightrag/api/lightrag_server.py` - 添加基于绑定的 rerank 函数选择逻辑
3. `.env` - 添加 `RERANK_BINDING=siliconflow` 配置

### 新增功能

- 支持多种 rerank 提供商的自动配置
- SiliconFlow 端点自动处理
- 统一的配置接口

## 📋 注意事项

1. **API 密钥**：确保为选择的提供商配置正确的 API 密钥
2. **端点地址**：不同提供商的 API 端点格式可能不同
3. **模型名称**：使用提供商支持的模型名称
4. **向下兼容**：现有的手动 rerank 函数配置仍然有效

## 🎉 测试结果

✅ **所有配置的模型都正常工作！**

- **LLM**: OpenRouter (qwen/qwen3-235b-a22b-07-25:free) - 正常
- **Embedding**: SiliconFlow (Qwen/Qwen3-Embedding-0.6B) - 正常  
- **Rerank**: SiliconFlow (Qwen/Qwen3-Reranker-8B) - 正常

现在您可以通过简单的配置项直接指定 rerank 提供商，无需手动编写 rerank 函数！
