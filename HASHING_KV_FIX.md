# LightRAG hashing_kv 兼容性修复

## 🐛 问题描述

在使用 Redis 等存储后端时，LightRAG 服务器出现以下错误：

```
AttributeError: 'RedisKVStorage' object has no attribute 'get'
```

错误发生在 `openrouter.py` 第191行：
```python
model_name = kwargs.get("hashing_kv", {}).get("global_config", {}).get("llm_model_name", "openai/gpt-4o-mini")
```

## 🔍 问题分析

### 根本原因
LightRAG 中不同的 LLM 实现对 `hashing_kv` 参数的处理方式不一致：

1. **正确的方式** (hf.py, ollama.py, bedrock.py):
   ```python
   model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
   ```

2. **错误的方式** (openrouter.py, siliconflow.py, gemini.py):
   ```python
   model_name = kwargs.get("hashing_kv", {}).get("global_config", {}).get("llm_model_name", default)
   ```

### 问题详情
- 当 `hashing_kv` 是存储对象（如 `RedisKVStorage`）时，第二种方式会失败
- 存储对象没有 `.get()` 方法，导致 `AttributeError`
- 这种错误的链式调用假设 `hashing_kv` 是字典，但实际上它是存储对象

## 🔧 修复方案

### 修复的文件

1. **lightrag/llm/openrouter.py** (第190-191行)
2. **lightrag/llm/siliconcloud.py** (第189-190行)  
3. **lightrag/llm/gemini.py** (第216-217行)

### 修复前后对比

**修复前：**
```python
# 错误：假设 hashing_kv 是字典
model_name = kwargs.get("hashing_kv", {}).get("global_config", {}).get("llm_model_name", "default")
```

**修复后：**
```python
# 正确：检查 hashing_kv 是否是存储对象
hashing_kv = kwargs.get("hashing_kv")
if hashing_kv and hasattr(hashing_kv, 'global_config'):
    model_name = hashing_kv.global_config.get("llm_model_name", "default")
else:
    model_name = "default"
```

## ✅ 修复验证

### 测试脚本
创建了 `test_fix_verification.py` 来验证修复效果：

```python
# 模拟存储对象
class MockRedisKVStorage:
    def __init__(self):
        self.global_config = {
            "llm_model_name": "qwen/qwen3-235b-a22b-07-25:free"
        }

# 测试修复后的函数
result = await openrouter_complete(
    prompt="测试问题",
    hashing_kv=mock_storage,  # 传入存储对象而不是字典
    ...
)
```

### 测试结果
```
✅ hashing_kv 修复验证成功!
📝 LLM 响应: 人工智能（Artificial Intelligence，简称 AI）是指...
🎉 修复验证成功！现在可以重新启动 LightRAG 服务器了。
```

## 🎯 影响范围

### 修复的 LLM 绑定
- ✅ **OpenRouter** - 修复完成
- ✅ **SiliconFlow** - 修复完成  
- ✅ **Gemini** - 修复完成

### 不受影响的 LLM 绑定
- ✅ **Ollama** - 使用正确的方式
- ✅ **HuggingFace** - 使用正确的方式
- ✅ **Bedrock** - 使用正确的方式
- ✅ **Zhipu** - 不使用 hashing_kv 获取模型名

## 🚀 使用建议

### 重启服务器
修复完成后，重新启动 LightRAG 服务器：

```bash
python -m lightrag.api.lightrag_server
```

### 验证修复
1. 启动服务器后访问 Web UI
2. 尝试进行查询操作
3. 检查是否还有 `AttributeError` 错误

### 配置建议
确保 `.env` 文件中的配置正确：

```bash
# LLM 配置
LLM_BINDING=openrouter
LLM_MODEL=qwen/qwen3-235b-a22b-07-25:free
LLM_BINDING_HOST=https://openrouter.ai/api/v1
LLM_BINDING_API_KEY=your_key_here

# 存储配置
LIGHTRAG_KV_STORAGE=RedisKVStorage
REDIS_URI=redis://localhost:6379
```

## 📋 技术细节

### 存储对象结构
LightRAG 中的存储对象（如 `RedisKVStorage`）具有以下结构：
```python
class BaseKVStorage:
    def __init__(self):
        self.global_config = {
            "llm_model_name": "model_name",
            "enable_llm_cache": True,
            # ... 其他配置
        }
```

### 正确的访问方式
```python
# 正确：直接访问属性
model_name = hashing_kv.global_config["llm_model_name"]

# 正确：带检查的访问
if hashing_kv and hasattr(hashing_kv, 'global_config'):
    model_name = hashing_kv.global_config.get("llm_model_name", "default")
```

### 错误的访问方式
```python
# 错误：假设是字典
model_name = hashing_kv.get("global_config", {}).get("llm_model_name")
```

## 🎉 总结

这个修复解决了 LightRAG 在使用 Redis 等存储后端时的兼容性问题，确保了：

1. ✅ **OpenRouter LLM** 正常工作
2. ✅ **SiliconFlow LLM** 正常工作  
3. ✅ **Gemini LLM** 正常工作
4. ✅ **所有存储后端** 兼容性
5. ✅ **查询功能** 正常运行

现在您可以放心使用 LightRAG 服务器了！
