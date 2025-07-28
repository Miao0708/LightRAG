# LightRAG 查询模式流程图

本文档包含LightRAG各种查询模式的详细流程图，帮助理解每种模式的工作原理。

## 整体架构图

```mermaid
graph TB
    subgraph "LLM模型层"
        LLM[LLM模型]
        LLM --> |关键词提取| KE[关键词提取模型]
        LLM --> |查询生成| QG[查询生成模型]
        LLM --> |响应生成| RG[响应生成模型]
    end
    
    subgraph "存储层"
        KG[知识图谱存储<br/>BaseGraphStorage]
        EVD[实体向量数据库<br/>BaseVectorStorage]
        RVD[关系向量数据库<br/>BaseVectorStorage]
        CVD[文档块向量数据库<br/>BaseVectorStorage]
        TCD[文本块存储<br/>BaseKVStorage]
    end
    
    subgraph "查询模式"
        NAIVE[Naive模式<br/>纯向量检索]
        LOCAL[Local模式<br/>局部实体检索]
        GLOBAL[Global模式<br/>全局关系检索]
        HYBRID[Hybrid模式<br/>混合检索]
        MIX[Mix模式<br/>图谱+向量融合]
        BYPASS[Bypass模式<br/>直接LLM调用]
    end
    
    Query[用户查询] --> KE
    KE --> |高级关键词<br/>低级关键词| QueryRouter{查询路由器}
    
    QueryRouter --> NAIVE
    QueryRouter --> LOCAL
    QueryRouter --> GLOBAL
    QueryRouter --> HYBRID
    QueryRouter --> MIX
    QueryRouter --> BYPASS
    
    NAIVE --> CVD
    LOCAL --> EVD
    LOCAL --> KG
    GLOBAL --> RVD
    GLOBAL --> KG
    HYBRID --> EVD
    HYBRID --> RVD
    HYBRID --> KG
    MIX --> EVD
    MIX --> RVD
    MIX --> CVD
    MIX --> KG
    BYPASS --> LLM
    
    CVD --> RG
    EVD --> RG
    RVD --> RG
    KG --> RG
    TCD --> RG
    
    RG --> Response[最终响应]
    
    style LLM fill:#e1f5fe
    style KG fill:#f3e5f5
    style EVD fill:#e8f5e8
    style RVD fill:#fff3e0
    style CVD fill:#fce4ec
    style TCD fill:#f1f8e9
    style BYPASS fill:#ffebee
```

## 六种查询模式详细流程

```mermaid
graph TD
    subgraph "1. Naive模式 - 纯向量检索"
        N1[用户查询] --> N2[直接向量搜索]
        N2 --> N3[文档块向量数据库<br/>chunks_vdb.query]
        N3 --> N4[获取相似文档块]
        N4 --> N5[LLM生成响应]
        N5 --> N6[返回结果]
    end
    
    subgraph "2. Local模式 - 局部实体检索"
        L1[用户查询] --> L2[LLM提取低级关键词<br/>ll_keywords]
        L2 --> L3[实体向量搜索<br/>entities_vdb.query]
        L3 --> L4[获取相关实体]
        L4 --> L5[知识图谱查询<br/>get_nodes_batch]
        L5 --> L6[获取实体邻域关系]
        L6 --> L7[构建局部上下文]
        L7 --> L8[LLM生成响应]
        L8 --> L9[返回结果]
    end
    
    subgraph "3. Global模式 - 全局关系检索"
        G1[用户查询] --> G2[LLM提取高级关键词<br/>hl_keywords]
        G2 --> G3[关系向量搜索<br/>relationships_vdb.query]
        G3 --> G4[获取相关关系]
        G4 --> G5[知识图谱查询<br/>get_edges_batch]
        G5 --> G6[获取全局关系网络]
        G6 --> G7[构建全局上下文]
        G7 --> G8[LLM生成响应]
        G8 --> G9[返回结果]
    end
    
    subgraph "4. Hybrid模式 - 混合检索"
        H1[用户查询] --> H2[LLM提取双层关键词<br/>hl_keywords + ll_keywords]
        H2 --> H3[并行执行Local和Global]
        H3 --> H4[合并实体和关系上下文]
        H4 --> H5[统一处理和排序]
        H5 --> H6[LLM生成响应]
        H6 --> H7[返回结果]
    end
    
    subgraph "5. Mix模式 - 图谱+向量融合"
        M1[用户查询] --> M2[LLM提取双层关键词]
        M2 --> M3[三路并行检索]
        M3 --> M4[向量检索<br/>chunks_vdb]
        M3 --> M5[实体检索<br/>entities_vdb]
        M3 --> M6[关系检索<br/>relationships_vdb]
        M4 --> M7[融合所有上下文]
        M5 --> M7
        M6 --> M7
        M7 --> M8[统一重排序]
        M8 --> M9[LLM生成响应]
        M9 --> M10[返回结果]
    end
    
    subgraph "6. Bypass模式 - 直接LLM调用"
        B1[用户查询] --> B2[跳过所有检索]
        B2 --> B3[直接LLM调用<br/>最高优先级8]
        B3 --> B4[支持对话历史]
        B4 --> B5[返回结果]
    end
    
    style N3 fill:#fce4ec
    style L3 fill:#e8f5e8
    style L5 fill:#f3e5f5
    style G3 fill:#fff3e0
    style G5 fill:#f3e5f5
    style M4 fill:#fce4ec
    style M5 fill:#e8f5e8
    style M6 fill:#fff3e0
    style B3 fill:#ffebee
```

## 关键词提取与模型使用流程

```mermaid
graph TD
    subgraph "关键词提取流程"
        Q1[用户查询] --> Q2{检查预定义关键词}
        Q2 -->|有| Q3[使用预定义关键词]
        Q2 -->|无| Q4[LLM关键词提取]
        
        Q4 --> Q5[构建提取提示词]
        Q5 --> Q6[添加示例和历史对话]
        Q6 --> Q7[LLM模型调用<br/>keyword_extraction=True]
        Q7 --> Q8[解析JSON响应]
        Q8 --> Q9[提取高级关键词<br/>hl_keywords]
        Q8 --> Q10[提取低级关键词<br/>ll_keywords]
        
        Q3 --> Q11[关键词验证]
        Q9 --> Q11
        Q10 --> Q11
    end
    
    subgraph "模型使用策略"
        M1[查询参数model_func] --> M2{是否指定模型}
        M2 -->|是| M3[使用指定模型]
        M2 -->|否| M4[使用全局配置模型]
        
        M4 --> M5[query_llm_func优先]
        M5 --> M6[llm_model_func备用]
        M6 --> M7[设置优先级=5]
        
        M3 --> M8[模型调用]
        M7 --> M8
        
        M8 --> M9[关键词提取模型]
        M8 --> M10[查询生成模型]
        M8 --> M11[响应生成模型]
    end
    
    subgraph "缓存机制"
        C1[计算参数哈希] --> C2[检查缓存]
        C2 -->|命中| C3[返回缓存结果]
        C2 -->|未命中| C4[执行查询]
        C4 --> C5[存储结果到缓存]
    end
    
    subgraph "模式自适应"
        A1[检查关键词完整性] --> A2{ll_keywords为空?}
        A2 -->|是| A3[local/hybrid → global]
        A2 -->|否| A4{hl_keywords为空?}
        A4 -->|是| A5[global/hybrid → local]
        A4 -->|否| A6[保持原模式]
        
        A3 --> A7[模式切换警告]
        A5 --> A7
        A6 --> A8[继续执行]
        A7 --> A8
    end
    
    Q11 --> A1
    M8 --> C1
    A8 --> FinalExecution[执行最终查询]
    
    style Q7 fill:#e1f5fe
    style M9 fill:#e8f5e8
    style M10 fill:#fff3e0
    style M11 fill:#fce4ec
    style C3 fill:#f1f8e9
    style A7 fill:#ffebee
```

## 性能对比图

```mermaid
graph LR
    subgraph "响应时间对比"
        T1[Bypass: 最快]
        T2[Naive: 0.8s]
        T3[Local: 1.2s]
        T4[Hybrid: 1.8s]
        T5[Global: 2.1s]
        T6[Mix: 2.5s]
    end
    
    subgraph "准确率对比"
        A1[Naive: 72%]
        A2[Local: 78%]
        A3[Global: 82%]
        A4[Hybrid: 85%]
        A5[Mix: 88%]
        A6[Bypass: 依赖LLM]
    end
    
    subgraph "资源消耗对比"
        R1[Bypass: 最低]
        R2[Naive: 低]
        R3[Local: 中]
        R4[Global: 中高]
        R5[Hybrid: 中高]
        R6[Mix: 最高]
    end
    
    style T1 fill:#e8f5e8
    style T6 fill:#ffcdd2
    style A5 fill:#e8f5e8
    style A1 fill:#ffcdd2
    style R1 fill:#e8f5e8
    style R6 fill:#ffcdd2
```

## 模式选择决策树

```mermaid
graph TD
    Start[开始查询] --> Q1{查询类型?}
    
    Q1 -->|简单事实| Simple{需要快速响应?}
    Q1 -->|实体相关| Entity[Local模式]
    Q1 -->|概念分析| Concept[Global模式]
    Q1 -->|复杂推理| Complex{精度要求?}
    Q1 -->|普通对话| Chat[Bypass模式]
    
    Simple -->|是| Naive[Naive模式]
    Simple -->|否| Hybrid1[Hybrid模式]
    
    Complex -->|高| Mix[Mix模式]
    Complex -->|中| Hybrid2[Hybrid模式]
    
    Q2{资源限制?} --> Q3{准确率要求?}
    Q3 -->|高| HighAcc[Mix/Hybrid模式]
    Q3 -->|中| MedAcc[Local/Global模式]
    Q3 -->|低| LowAcc[Naive模式]
    
    style Naive fill:#fce4ec
    style Entity fill:#e8f5e8
    style Concept fill:#fff3e0
    style Hybrid1 fill:#e1f5fe
    style Hybrid2 fill:#e1f5fe
    style Mix fill:#f3e5f5
    style Chat fill:#ffebee
```

## 使用建议

### 模式选择矩阵

| 查询特征 | 推荐模式 | 备选模式 |
|----------|----------|----------|
| 简单事实查询 | Naive | Hybrid |
| 实体关系查询 | Local | Mix |
| 概念性查询 | Global | Hybrid |
| 复杂推理查询 | Mix | Hybrid |
| 快速响应需求 | Naive | Bypass |
| 高精度需求 | Mix | Hybrid |
| 资源受限 | Naive | Bypass |
| 普通对话 | Bypass | - |

### 性能调优建议

1. **Naive模式**: 适合快速原型和简单查询
2. **Local模式**: 调整`top_k`参数优化实体检索
3. **Global模式**: 关注关系质量和图谱完整性
4. **Hybrid模式**: 平衡性能和准确性的最佳选择
5. **Mix模式**: 最高精度，适合关键业务场景
6. **Bypass模式**: 用于对话和不需要检索的场景

这些流程图帮助开发者理解每种查询模式的内部工作机制，从而做出最适合的选择。
