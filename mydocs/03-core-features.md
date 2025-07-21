# LightRAG 核心功能详解

## 功能概览

```mermaid
mindmap
  root((LightRAG 核心功能))
    文档处理
      多格式支持
      智能分块
      增量更新
      批量处理
    知识图谱
      实体提取
      关系建模
      图谱构建
      可视化
    检索系统
      双层检索
      多模式查询
      相似度计算
      结果排序
    生成增强
      上下文整合
      答案生成
      流式输出
      多轮对话
```

## 1. 文档处理系统

### 1.1 多格式文档支持

```mermaid
graph LR
    A[文档输入] --> B{文档类型}
    B -->|文本| C[TXT/MD 处理器]
    B -->|PDF| D[PDF 解析器]
    B -->|Office| E[DOC/PPT 处理器]
    B -->|网页| F[HTML 解析器]
    B -->|代码| G[代码解析器]
    
    C --> H[统一文本格式]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I[文本预处理]
    I --> J[分块处理]
```

**支持的文档格式：**
- **纯文本**：TXT, MD, CSV
- **PDF 文档**：自动提取文本和表格
- **Office 文档**：DOC, DOCX, PPT, PPTX
- **网页内容**：HTML, XML
- **代码文件**：Python, Java, JavaScript 等

### 1.2 智能文本分块

```python
# 分块配置示例
CHUNK_CONFIG = {
    "chunk_token_size": 1200,      # 每块最大 token 数
    "chunk_overlap_token_size": 100, # 重叠 token 数
    "tiktoken_model_name": "gpt-4o", # 分词模型
    "enable_smart_boundary": True,   # 智能边界检测
    "preserve_structure": True       # 保持文档结构
}
```

**分块策略特点：**
- **智能边界**：避免在句子中间分割
- **结构保持**：保留段落、章节等结构信息
- **重叠处理**：确保上下文连续性
- **动态调整**：根据内容类型调整分块大小

### 1.3 增量更新机制

```mermaid
sequenceDiagram
    participant U as User
    participant D as Document Manager
    participant S as Status Store
    participant I as Index
    participant G as Graph Store
    
    U->>D: 添加新文档
    D->>S: 检查文档状态
    S->>D: 返回状态信息
    
    alt 新文档
        D->>I: 创建新索引
        I->>G: 构建知识图谱
    else 已存在文档
        D->>I: 更新现有索引
        I->>G: 增量更新图谱
    end
    
    G->>S: 更新处理状态
    S->>U: 返回处理结果
```

## 2. 知识图谱构建

### 2.1 实体提取流程

```mermaid
flowchart TD
    A[文本块] --> B[LLM 实体提取]
    B --> C[实体标准化]
    C --> D[实体去重]
    D --> E[实体类型识别]
    E --> F[实体存储]
    
    subgraph "提取策略"
        B1[命名实体识别]
        B2[概念实体提取]
        B3[事件实体识别]
    end
    
    subgraph "标准化处理"
        C1[大小写统一]
        C2[同义词合并]
        C3[缩写展开]
    end
    
    B --> B1
    B --> B2
    B --> B3
    
    C --> C1
    C --> C2
    C --> C3
```

**实体提取示例：**
```python
# 实体提取配置
ENTITY_EXTRACT_CONFIG = {
    "entity_extract_max_gleaning": 1,  # 最大提取轮数
    "entity_summary_to_max_tokens": 500, # 实体摘要最大长度
    "entity_types": [
        "PERSON", "ORGANIZATION", "LOCATION", 
        "CONCEPT", "EVENT", "TECHNOLOGY"
    ]
}
```

### 2.2 关系建模

```mermaid
graph TD
    A[实体对] --> B[关系识别]
    B --> C[关系类型分类]
    C --> D[关系权重计算]
    D --> E[关系验证]
    E --> F[关系存储]
    
    subgraph "关系类型"
        C1[层次关系]
        C2[关联关系]
        C3[因果关系]
        C4[时序关系]
    end
    
    subgraph "权重因子"
        D1[共现频率]
        D2[语义相似度]
        D3[上下文相关性]
    end
    
    C --> C1
    C --> C2
    C --> C3
    C --> C4
    
    D --> D1
    D --> D2
    D --> D3
```

### 2.3 图谱构建算法

```python
# 图谱构建流程
def build_knowledge_graph(documents):
    """构建知识图谱"""
    entities = []
    relations = []
    
    for doc in documents:
        # 1. 实体提取
        doc_entities = extract_entities(doc)
        entities.extend(doc_entities)
        
        # 2. 关系提取
        doc_relations = extract_relations(doc, doc_entities)
        relations.extend(doc_relations)
    
    # 3. 实体去重和合并
    merged_entities = merge_entities(entities)
    
    # 4. 关系权重计算
    weighted_relations = calculate_weights(relations)
    
    # 5. 图谱构建
    graph = construct_graph(merged_entities, weighted_relations)
    
    return graph
```

## 3. 双层检索系统

### 3.1 检索模式对比

```mermaid
graph TD
    A[查询输入] --> B{选择检索模式}
    
    B -->|Local| C[局部检索]
    B -->|Global| D[全局检索]
    B -->|Hybrid| E[混合检索]
    B -->|Mix| F[图谱+向量检索]
    B -->|Naive| G[纯向量检索]
    
    C --> C1[高层关键词匹配]
    C --> C2[低层关键词匹配]
    C1 --> H[局部上下文]
    C2 --> H
    
    D --> D1[关系路径搜索]
    D --> D2[全局模式识别]
    D1 --> I[全局上下文]
    D2 --> I
    
    E --> C
    E --> D
    
    F --> J[知识图谱检索]
    F --> K[向量相似度检索]
    J --> L[混合上下文]
    K --> L
    
    G --> K
    
    H --> M[LLM 生成]
    I --> M
    L --> M
```

### 3.2 关键词提取策略

```python
# 关键词提取配置
KEYWORD_EXTRACT_CONFIG = {
    "high_level_keywords": {
        "max_count": 8,
        "min_frequency": 2,
        "semantic_threshold": 0.7
    },
    "low_level_keywords": {
        "max_count": 16,
        "include_entities": True,
        "include_concepts": True
    }
}
```

### 3.3 检索性能优化

```mermaid
graph LR
    A[查询优化] --> B[索引优化]
    A --> C[缓存策略]
    A --> D[并行检索]
    
    B --> B1[倒排索引]
    B --> B2[向量索引]
    B --> B3[图索引]
    
    C --> C1[查询缓存]
    C --> C2[结果缓存]
    C --> C3[嵌入缓存]
    
    D --> D1[多线程检索]
    D --> D2[异步处理]
    D --> D3[批量查询]
```

## 4. 生成增强功能

### 4.1 上下文整合

```mermaid
flowchart TD
    A[检索结果] --> B[相关性评分]
    B --> C[去重处理]
    C --> D[内容排序]
    D --> E[上下文构建]
    E --> F[Token 限制检查]
    F --> G{超出限制?}
    G -->|是| H[内容截断]
    G -->|否| I[完整上下文]
    H --> I
    I --> J[LLM 输入]
```

### 4.2 答案生成策略

```python
# 生成配置
GENERATION_CONFIG = {
    "max_tokens": 2000,
    "temperature": 0.1,
    "top_p": 0.9,
    "stream": True,
    "include_sources": True,
    "citation_format": "markdown"
}
```

### 4.3 多轮对话支持

```mermaid
sequenceDiagram
    participant U as User
    participant C as Conversation Manager
    participant H as History Store
    participant R as Retrieval Engine
    participant L as LLM
    
    U->>C: 发送查询
    C->>H: 获取对话历史
    H->>C: 返回历史上下文
    C->>R: 增强查询检索
    R->>C: 返回相关内容
    C->>L: 生成回答
    L->>C: 返回答案
    C->>H: 更新对话历史
    C->>U: 返回最终答案
```

## 5. 高级功能特性

### 5.1 重排序优化

```python
# 重排序配置
RERANK_CONFIG = {
    "enabled": True,
    "model": "BAAI/bge-reranker-v2-m3",
    "top_k": 10,
    "score_threshold": 0.5,
    "rerank_mode": "mix"  # 推荐使用 mix 模式
}
```

### 5.2 批量处理

```mermaid
graph TD
    A[文档批次] --> B[并发控制]
    B --> C[Worker Pool]
    C --> D[处理队列]
    D --> E[结果聚合]
    E --> F[状态更新]
    
    subgraph "处理配置"
        G[max_async: 16]
        H[batch_size: 32]
        I[timeout: 300s]
    end
    
    B --> G
    B --> H
    B --> I
```

### 5.3 API 集成

```python
# API 使用示例
from lightrag import LightRAG

# 初始化
rag = LightRAG(
    working_dir="./ragtest",
    llm_model_func=llm_model_func,
    embedding_func=embedding_func
)

# 插入文档
rag.insert("Your document content here")

# 查询
result = rag.query(
    "Your question here", 
    param=QueryParam(mode="hybrid")
)

# 流式查询
for chunk in rag.query_stream("Your question"):
    print(chunk, end="")
```

## 6. 性能监控

### 6.1 关键指标

```mermaid
graph LR
    A[性能指标] --> B[处理速度]
    A --> C[准确性]
    A --> D[资源使用]
    
    B --> B1[文档处理速度]
    B --> B2[查询响应时间]
    B --> B3[索引构建时间]
    
    C --> C1[检索准确率]
    C --> C2[答案相关性]
    C --> C3[实体识别精度]
    
    D --> D1[内存占用]
    D --> D2[存储空间]
    D --> D3[API 调用次数]
```

### 6.2 调试工具

```python
# 调试配置
DEBUG_CONFIG = {
    "log_level": "DEBUG",
    "enable_profiling": True,
    "trace_retrieval": True,
    "save_intermediate_results": True,
    "performance_metrics": True
}
```

这些核心功能构成了 LightRAG 的完整功能体系，为用户提供了强大而灵活的 RAG 解决方案。
