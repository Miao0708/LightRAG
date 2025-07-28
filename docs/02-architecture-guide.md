# LightRAG 系统架构指南

## 🏗️ 整体架构概览

LightRAG 采用现代分层架构设计，从用户接口到存储实现共分为6个主要层次，每层职责清晰，接口标准化，支持灵活的组件替换和扩展。

```mermaid
graph TB
    subgraph "🌐 用户接口层"
        A1[Web UI]
        A2[REST API]
        A3[Ollama API]
        A4[Python SDK]
    end
    
    subgraph "⚡ API服务层"
        B1[FastAPI 服务器]
        B2[认证中间件]
        B3[文档管理器]
        B4[路由处理]
    end
    
    subgraph "🧠 核心处理层"
        C1[LightRAG 核心]
        C2[文档处理管道]
        C3[查询处理引擎]
        C4[知识图谱构建]
    end
    
    subgraph "🔍 检索增强层"
        D1[Local 检索]
        D2[Global 检索]
        D3[Hybrid 检索]
        D4[Mix 检索]
    end
    
    subgraph "🤖 LLM服务层"
        E1[多模型支持]
        E2[任务特化配置]
        E3[缓存管理]
        E4[并发控制]
    end
    
    subgraph "💾 存储抽象层"
        F1[KV存储]
        F2[图数据库]
        F3[向量数据库]
        F4[文档状态]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> C1
    
    B1 --> C1
    B2 --> B1
    B3 --> B1
    B4 --> B1
    
    C1 --> D1
    C2 --> C1
    C3 --> C1
    C4 --> C1
    
    D1 --> E1
    D2 --> E1
    D3 --> E1
    D4 --> E1
    
    E1 --> F1
    E1 --> F2
    E1 --> F3
    E1 --> F4
```

## 📱 用户接口层详解

### 1.1 Web UI 界面

```mermaid
graph LR
    A[用户] --> B[React 前端]
    B --> C[文档管理]
    B --> D[图谱可视化]
    B --> E[查询测试]
    B --> F[系统监控]
    
    C --> C1[文档上传]
    C --> C2[批量导入]
    C --> C3[状态跟踪]
    
    D --> D1[实体关系图]
    D --> D2[交互式探索]
    D --> D3[图谱统计]
    
    E --> E1[多模式查询]
    E --> E2[结果对比]
    E --> E3[性能分析]
```

**功能特点**：
- 🎨 **现代化UI**: 基于React的响应式界面
- 📊 **可视化图谱**: 交互式知识图谱探索
- 📈 **实时监控**: 系统状态和性能指标
- 🔍 **查询调试**: 多种查询模式的测试界面

### 1.2 REST API 接口

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Core
    participant Storage
    
    Client->>+API: POST /documents/upload
    API->>+Core: process_document()
    Core->>+Storage: save_chunks()
    Storage-->>-Core: success
    Core-->>-API: processing_status
    API-->>-Client: 202 Accepted
    
    Client->>+API: POST /query
    API->>+Core: aquery()
    Core->>+Storage: retrieve_relevant()
    Storage-->>-Core: context_data
    Core-->>-API: generated_response
    API-->>-Client: 200 OK + response
```

**API 端点设计**：
- 📄 **文档管理**: `/documents/*` - 上传、删除、状态查询
- 🔍 **查询服务**: `/query/*` - 多模式查询、流式响应
- 🗺️ **图谱操作**: `/graphs/*` - 图谱查询、可视化数据
- 📊 **系统监控**: `/health/*` - 健康检查、性能指标

## ⚙️ 核心处理层架构

### 3.1 文档处理管道

```mermaid
flowchart TD
    A[文档输入] --> B[格式检测]
    B --> C{文件类型}
    
    C -->|Text/MD| D[直接处理]
    C -->|PDF| E[PDF解析]
    C -->|DOC/PPT| F[Office解析]
    C -->|其他| G[通用提取]
    
    D --> H[文本分块]
    E --> H
    F --> H
    G --> H
    
    H --> I[并行处理]
    I --> J[实体提取]
    I --> K[关系识别]
    I --> L[向量化]
    
    J --> M[知识图谱更新]
    K --> M
    L --> N[向量索引更新]
    M --> O[存储同步]
    N --> O
    
    O --> P[处理完成]
```

**管道特点**：
- ⚡ **异步处理**: 支持大规模文档的并发处理
- 🔄 **增量更新**: 只处理新增和修改的内容
- 🛡️ **错误恢复**: 自动重试和错误处理机制
- 📊 **进度跟踪**: 实时处理状态和进度反馈

### 3.2 查询处理引擎

```mermaid
graph TB
    A[用户查询] --> B[查询解析]
    B --> C[意图识别]
    C --> D{选择检索模式}
    
    D -->|事实查询| E[Local 检索]
    D -->|概念查询| F[Global 检索]
    D -->|复杂查询| G[Hybrid 检索]
    D -->|综合查询| H[Mix 检索]
    
    E --> I[实体邻域搜索]
    F --> J[全局推理]
    G --> K[混合策略]
    H --> L[多源融合]
    
    I --> M[上下文构建]
    J --> M
    K --> M
    L --> M
    
    M --> N[LLM生成]
    N --> O[结果后处理]
    O --> P[返回答案]
```

## 🤖 LLM服务层

### 5.1 多模型支持架构

```mermaid
graph TB
    subgraph "🎯 任务特化层"
        A1[实体提取]
        A2[关系摘要]
        A3[查询响应]
        A4[关键词提取]
    end
    
    subgraph "🔧 模型配置层"
        B1[OpenAI]
        B2[SiliconFlow]
        B3[ZhipuAI]
        B4[Gemini]
        B5[OpenRouter]
        B6[Ollama]
    end
    
    subgraph "⚡ 执行层"
        C1[负载均衡]
        C2[缓存管理]
        C3[并发控制]
        C4[错误处理]
    end
    
    A1 --> B2
    A2 --> B2
    A3 --> B1
    A4 --> B2
    
    B1 --> C1
    B2 --> C1
    B3 --> C1
    B4 --> C1
    B5 --> C1
    B6 --> C1
    
    C1 --> C2
    C1 --> C3
    C1 --> C4
```

**配置策略**：
- 💰 **成本优化**: 不同任务使用不同成本的模型
- 🎯 **性能匹配**: 根据任务复杂度选择合适模型
- 🔄 **动态切换**: 支持运行时模型切换
- 📊 **性能监控**: 实时监控各模型的性能指标

## 💾 存储抽象层

### 6.1 存储架构设计

```mermaid
graph TB
    subgraph "🏛️ 存储接口层"
        A1[KV接口]
        A2[图接口]
        A3[向量接口]
        A4[状态接口]
    end
    
    subgraph "🗄️ 本地存储"
        B1[JSON KV]
        B2[NetworkX]
        B3[NanoVectorDB]
        B4[JSON状态]
    end
    
    subgraph "☁️ 云存储"
        C1[Redis]
        C2[Neo4j/Memgraph]
        C3[Milvus/Qdrant]
        C4[PostgreSQL]
    end
    
    subgraph "🚀 生产存储"
        D1[MongoDB]
        D2[Faiss]
        D3[企业级数据库]
    end
    
    A1 --> B1
    A1 --> C1
    A1 --> D1
    
    A2 --> B2
    A2 --> C2
    
    A3 --> B3
    A3 --> C3
    A3 --> D2
    
    A4 --> B4
    A4 --> C4
```

**存储特点**：
- 🔄 **统一接口**: 所有存储后端实现相同接口
- 📈 **弹性扩展**: 支持从本地到云端的无缝迁移
- 🔒 **数据一致性**: 跨存储的事务一致性保证
- 📊 **性能优化**: 针对不同存储类型的专门优化

## 🔧 部署架构

### 单机部署

```mermaid
graph TB
    A[LightRAG Server] --> B[Local Storage]
    A --> C[Ollama Server]
    A --> D[External LLM API]
    
    B --> B1[JSON Files]
    B --> B2[NetworkX Graph]
    B --> B3[NanoVectorDB]
```

### 分布式部署

```mermaid
graph TB
    subgraph "🌐 负载均衡层"
        A[Load Balancer]
    end
    
    subgraph "⚡ 应用层"
        B1[LightRAG Server 1]
        B2[LightRAG Server 2]
        B3[LightRAG Server N]
    end
    
    subgraph "🗄️ 存储层"
        C1[Redis Cluster]
        C2[Neo4j Cluster]
        C3[Milvus Cluster]
        C4[PostgreSQL HA]
    end
    
    subgraph "🤖 AI服务层"
        D1[LLM Service Pool]
        D2[Embedding Service]
        D3[Rerank Service]
    end
    
    A --> B1
    A --> B2
    A --> B3
    
    B1 --> C1
    B1 --> C2
    B1 --> C3
    B1 --> C4
    
    B2 --> C1
    B2 --> C2
    B2 --> C3
    B2 --> C4
    
    B3 --> C1
    B3 --> C2
    B3 --> C3
    B3 --> C4
    
    B1 --> D1
    B1 --> D2
    B1 --> D3
    
    B2 --> D1
    B2 --> D2
    B2 --> D3
    
    B3 --> D1
    B3 --> D2
    B3 --> D3
```

## 📊 性能特点

- ⚡ **高并发**: 支持数千用户同时查询
- 🚀 **低延迟**: 平均响应时间 < 2秒
- 📈 **高扩展**: 水平扩展支持PB级数据
- 🛡️ **高可用**: 99.9%+ 系统可用性

---

[📚 返回文档目录](./README.md) | [🚀 下一章：核心功能](./03-core-features.md) 