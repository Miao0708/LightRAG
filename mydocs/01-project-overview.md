# LightRAG 项目概述

## 项目简介

LightRAG 是由香港大学数据科学实验室开发的轻量级检索增强生成（RAG）系统。它巧妙地结合了知识图谱和向量检索技术，在保持高效性能的同时，显著提升了复杂问答的准确性。

## 核心优势

```mermaid
graph TD
    A[LightRAG 核心优势] --> B[轻量级设计]
    A --> C[双层检索机制]
    A --> D[知识图谱增强]
    A --> E[增量更新]
    A --> F[多模态支持]
    
    B --> B1[相比 GraphRAG 更快]
    B --> B2[API 成本更低]
    B --> B3[资源占用少]
    
    C --> C1[Local 检索]
    C --> C2[Global 检索]
    C --> C3[Hybrid 混合检索]
    C --> C4[Mix 模式]
    C --> C5[Naive 基础检索]
    
    D --> D1[实体关系提取]
    D --> D2[图结构索引]
    D --> D3[上下文理解增强]
    
    E --> E1[无需重建整个索引]
    E --> E2[支持实时数据更新]
    E --> E3[降低维护成本]
    
    F --> F1[文本处理]
    F --> F2[PDF/DOC/PPT 支持]
    F --> F3[图像理解]
```

## 技术特点

### 1. 图增强文本索引
- **实体提取**：自动识别文档中的关键实体
- **关系建模**：构建实体间的语义关系
- **层次化组织**：支持多层次的知识结构

### 2. 双层检索范式
- **低层检索**：精确的实体级别信息检索
- **高层检索**：抽象的概念级别信息整合
- **自适应切换**：根据查询类型自动选择最优检索策略

### 3. 多存储后端支持
```mermaid
graph LR
    A[LightRAG 存储架构] --> B[KV 存储]
    A --> C[向量存储]
    A --> D[图存储]
    A --> E[文档状态存储]
    
    B --> B1[JsonKVStorage]
    B --> B2[PGKVStorage]
    B --> B3[RedisKVStorage]
    B --> B4[MongoKVStorage]
    
    C --> C1[NanoVectorDBStorage]
    C --> C2[PGVectorStorage]
    C --> C3[MilvusVectorDBStorage]
    C --> C4[ChromaVectorDBStorage]
    C --> C5[FaissVectorDBStorage]
    
    D --> D1[NetworkXStorage]
    D --> D2[Neo4JStorage]
    D --> D3[PGGraphStorage]
    D --> D4[MemgraphStorage]
    
    E --> E1[JsonDocStatusStorage]
    E --> E2[PGDocStatusStorage]
    E --> E3[MongoDocStatusStorage]
```

## 与其他 RAG 系统对比

| 特性 | LightRAG | GraphRAG | 传统 RAG |
|------|----------|----------|----------|
| **检索方式** | 双层检索 | 社区遍历 | 向量检索 |
| **知识表示** | 知识图谱 + 向量 | 知识图谱 | 向量嵌入 |
| **更新机制** | 增量更新 | 全量重建 | 增量更新 |
| **成本效率** | 高 | 低 | 中 |
| **复杂推理** | 强 | 强 | 弱 |
| **部署难度** | 低 | 高 | 低 |

## 应用场景

### 1. 企业知识管理
- 技术文档问答
- 政策法规查询
- 产品手册检索

### 2. 学术研究
- 文献综述生成
- 跨领域知识整合
- 研究问题探索

### 3. 智能客服
- 复杂问题解答
- 多轮对话支持
- 上下文理解

## 项目生态

```mermaid
graph TD
    A[LightRAG 生态系统] --> B[核心库]
    A --> C[API 服务]
    A --> D[Web UI]
    A --> E[扩展组件]
    
    B --> B1[lightrag-hku]
    B --> B2[Python SDK]
    
    C --> C1[REST API]
    C --> C2[Ollama 兼容接口]
    C --> C3[流式响应]
    
    D --> D1[文档管理]
    D --> D2[知识图谱可视化]
    D --> D3[查询界面]
    
    E --> E1[RAG-Anything 多模态]
    E --> E2[VideoRAG 视频理解]
    E --> E3[MiniRAG 小模型版本]
```

## 技术栈要求

### LLM 要求
- **参数规模**：建议 32B 以上
- **上下文长度**：至少 32KB，推荐 64KB
- **支持模型**：OpenAI、Ollama、HuggingFace、Azure OpenAI 等

### 嵌入模型
- **推荐模型**：`BAAI/bge-m3`、`text-embedding-3-large`
- **重要提醒**：嵌入模型需在文档索引前确定，查询时必须使用相同模型

### 重排序模型（可选）
- **推荐模型**：`BAAI/bge-reranker-v2-m3`
- **优化效果**：显著提升检索性能，建议启用 "mix" 模式

## 项目状态

- **开源协议**：MIT License
- **开发状态**：活跃开发中
- **社区支持**：Discord、微信群
- **文档完善度**：高
- **测试覆盖率**：良好

## 下一步学习

1. [架构分析](./02-architecture-analysis.md) - 深入了解系统架构
2. [核心功能](./03-core-features.md) - 掌握主要功能特性
3. [安装部署](./04-installation-deployment.md) - 快速上手部署
4. [使用示例](./05-usage-examples.md) - 实践操作指南
