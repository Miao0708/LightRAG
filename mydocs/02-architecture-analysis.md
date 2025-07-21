# LightRAG 架构分析

## 整体架构

```mermaid
graph TB
    subgraph "用户层"
        A[Web UI] 
        B[REST API]
        C[Python SDK]
    end
    
    subgraph "应用层"
        D[LightRAG Core]
        E[Query Engine]
        F[Document Processor]
    end
    
    subgraph "服务层"
        G[LLM Service]
        H[Embedding Service]
        I[Rerank Service]
    end
    
    subgraph "存储层"
        J[KV Storage]
        K[Vector Storage]
        L[Graph Storage]
        M[Doc Status Storage]
    end
    
    A --> D
    B --> D
    C --> D
    
    D --> E
    D --> F
    
    E --> G
    E --> H
    E --> I
    
    F --> G
    F --> H
    
    D --> J
    D --> K
    D --> L
    D --> M
```

## 核心组件详解

### 1. 文档处理流水线

```mermaid
flowchart TD
    A[原始文档] --> B[文档解析]
    B --> C[文本分块]
    C --> D[实体提取]
    D --> E[关系识别]
    E --> F[向量化]
    F --> G[存储更新]
    
    subgraph "分块策略"
        C1[Token 大小: 1200]
        C2[重叠大小: 100]
        C3[智能边界检测]
    end
    
    subgraph "实体提取"
        D1[LLM 驱动提取]
        D2[实体消歧]
        D3[类型识别]
    end
    
    subgraph "关系建模"
        E1[实体间关系]
        E2[权重计算]
        E3[语义相似度]
    end
    
    C --> C1
    C --> C2
    C --> C3
    
    D --> D1
    D --> D2
    D --> D3
    
    E --> E1
    E --> E2
    E --> E3
```

### 2. 双层检索机制

```mermaid
graph TD
    A[用户查询] --> B[关键词提取]
    B --> C{检索模式}
    
    C -->|Local| D[实体检索]
    C -->|Global| E[关系检索]
    C -->|Hybrid| F[混合检索]
    C -->|Mix| G[图谱+向量检索]
    C -->|Naive| H[纯向量检索]
    
    D --> D1[高层关键词]
    D --> D2[低层关键词]
    D1 --> D3[实体匹配]
    D2 --> D3
    D3 --> I[上下文构建]
    
    E --> E1[关系模式识别]
    E1 --> E2[关系路径搜索]
    E2 --> I
    
    F --> D
    F --> E
    
    G --> J[知识图谱检索]
    G --> K[向量相似度检索]
    J --> I
    K --> I
    
    H --> K
    
    I --> L[LLM 生成答案]
```

### 3. 存储架构设计

```mermaid
erDiagram
    KV_STORAGE {
        string key
        json value
        timestamp created_at
        timestamp updated_at
    }
    
    VECTOR_STORAGE {
        string id
        vector embedding
        json metadata
        float similarity_score
    }
    
    GRAPH_STORAGE {
        string entity_id
        string entity_name
        string entity_type
        json properties
        string relation_id
        string source_id
        string target_id
        string relation_type
        float weight
    }
    
    DOC_STATUS {
        string doc_id
        string status
        json metadata
        timestamp processed_at
    }
    
    KV_STORAGE ||--o{ VECTOR_STORAGE : "references"
    GRAPH_STORAGE ||--o{ VECTOR_STORAGE : "embedded"
    DOC_STATUS ||--o{ KV_STORAGE : "tracks"
```

## 关键算法流程

### 1. 实体关系提取算法

```mermaid
sequenceDiagram
    participant D as Document
    participant C as Chunker
    participant L as LLM
    participant E as Entity Store
    participant R as Relation Store
    
    D->>C: 输入文档
    C->>C: 文本分块
    loop 每个文本块
        C->>L: 实体提取请求
        L->>L: 分析文本内容
        L->>E: 返回实体列表
        C->>L: 关系提取请求
        L->>R: 返回关系列表
    end
    E->>E: 实体去重合并
    R->>R: 关系权重计算
```

### 2. 查询处理算法

```mermaid
sequenceDiagram
    participant U as User
    participant Q as Query Engine
    participant K as Keyword Extractor
    participant G as Graph Store
    participant V as Vector Store
    participant L as LLM
    
    U->>Q: 用户查询
    Q->>K: 关键词提取
    K->>Q: 高低层关键词
    
    par 并行检索
        Q->>G: 图谱检索
        G->>Q: 相关实体关系
    and
        Q->>V: 向量检索
        V->>Q: 相似文本块
    end
    
    Q->>Q: 上下文整合
    Q->>L: 生成最终答案
    L->>U: 返回结果
```

## 性能优化策略

### 1. 缓存机制

```mermaid
graph LR
    A[查询请求] --> B{LLM 缓存}
    B -->|命中| C[返回缓存结果]
    B -->|未命中| D[LLM 处理]
    D --> E[更新缓存]
    E --> F[返回结果]
    
    G[嵌入请求] --> H{嵌入缓存}
    H -->|命中| I[返回缓存向量]
    H -->|未命中| J[嵌入计算]
    J --> K[更新缓存]
    K --> L[返回向量]
```

### 2. 并发处理

```mermaid
graph TD
    A[文档批处理] --> B[并发控制器]
    B --> C[Worker 1]
    B --> D[Worker 2]
    B --> E[Worker N]
    
    C --> F[实体提取]
    D --> G[关系提取]
    E --> H[向量化]
    
    F --> I[结果聚合]
    G --> I
    H --> I
    
    I --> J[存储更新]
```

## 扩展性设计

### 1. 插件化架构

```mermaid
graph TD
    A[LightRAG Core] --> B[LLM 插件接口]
    A --> C[存储插件接口]
    A --> D[嵌入插件接口]
    
    B --> B1[OpenAI Plugin]
    B --> B2[Ollama Plugin]
    B --> B3[HuggingFace Plugin]
    
    C --> C1[PostgreSQL Plugin]
    C --> C2[Neo4j Plugin]
    C --> C3[MongoDB Plugin]
    
    D --> D1[OpenAI Embedding]
    D --> D2[BGE Embedding]
    D --> D3[Custom Embedding]
```

### 2. 水平扩展支持

```mermaid
graph TB
    subgraph "负载均衡层"
        A[Load Balancer]
    end
    
    subgraph "应用层集群"
        B[LightRAG Instance 1]
        C[LightRAG Instance 2]
        D[LightRAG Instance N]
    end
    
    subgraph "存储层集群"
        E[PostgreSQL Cluster]
        F[Neo4j Cluster]
        G[Vector DB Cluster]
    end
    
    A --> B
    A --> C
    A --> D
    
    B --> E
    B --> F
    B --> G
    
    C --> E
    C --> F
    C --> G
    
    D --> E
    D --> F
    D --> G
```

## 配置管理

### 1. 环境变量配置

```mermaid
graph LR
    A[环境配置] --> B[LLM 配置]
    A --> C[存储配置]
    A --> D[性能配置]
    
    B --> B1[API_KEY]
    B --> B2[BASE_URL]
    B --> B3[MODEL_NAME]
    
    C --> C1[DATABASE_URL]
    C --> C2[VECTOR_DB_URL]
    C --> C3[GRAPH_DB_URL]
    
    D --> D1[MAX_ASYNC]
    D --> D2[BATCH_SIZE]
    D --> D3[CACHE_SIZE]
```

### 2. 运行时配置

```mermaid
graph TD
    A[LightRAG 初始化] --> B[配置验证]
    B --> C[存储初始化]
    C --> D[服务启动]
    
    B --> B1[检查必需参数]
    B --> B2[验证模型可用性]
    B --> B3[测试存储连接]
    
    C --> C1[创建存储实例]
    C --> C2[建立连接池]
    C --> C3[初始化索引]
```

## 监控与调试

### 1. 日志系统

```mermaid
graph LR
    A[应用日志] --> B[日志收集器]
    B --> C[日志处理]
    C --> D[存储/展示]
    
    A --> A1[INFO 级别]
    A --> A2[DEBUG 级别]
    A --> A3[ERROR 级别]
    
    D --> D1[文件存储]
    D --> D2[ELK Stack]
    D --> D3[监控面板]
```

### 2. 性能监控

```mermaid
graph TD
    A[性能指标] --> B[响应时间]
    A --> C[吞吐量]
    A --> D[资源使用]
    A --> E[错误率]
    
    B --> B1[查询延迟]
    B --> B2[索引时间]
    
    C --> C1[QPS]
    C --> C2[并发数]
    
    D --> D1[CPU 使用率]
    D --> D2[内存占用]
    D --> D3[存储空间]
    
    E --> E1[LLM 调用失败]
    E --> E2[存储连接错误]
```

这个架构分析展示了 LightRAG 的核心设计理念和实现细节，为深入理解和使用该系统提供了全面的技术视角。
