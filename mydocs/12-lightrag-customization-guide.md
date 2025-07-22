# LightRAG 定制化开发指南

## 1. 定制化开发概览

LightRAG提供了强大的定制化能力，支持在多个层面进行扩展和定制：
- **图谱抽取定制**：自定义实体和关系抽取逻辑
- **知识库构建定制**：自定义文档处理和索引策略
- **多知识库管理**：支持多个独立的知识库实例
- **存储后端扩展**：支持自定义存储实现
- **LLM集成扩展**：支持自定义LLM提供商

## 2. 定制化图谱抽取

### 2.1 自定义实体抽取

LightRAG允许通过自定义Prompt和后处理逻辑来定制实体抽取：

```python
# 自定义实体抽取Prompt
CUSTOM_ENTITY_EXTRACTION_PROMPT = """
从以下文本中抽取特定领域的实体：

文本：{input_text}

请抽取以下类型的实体：
1. 技术概念：编程语言、框架、工具等
2. 人物：开发者、作者、专家等  
3. 组织：公司、开源项目、标准组织等
4. 产品：软件产品、服务、平台等

输出格式：
{{"entities": [
    {{"name": "实体名称", "type": "实体类型", "description": "实体描述"}},
    ...
]}}
"""

# 自定义实体抽取函数
async def custom_entity_extraction(
    text_chunks: list[str],
    llm_model_func,
    **kwargs
) -> list[dict]:
    """自定义实体抽取函数"""
    entities = []
    
    for chunk in text_chunks:
        # 使用自定义Prompt
        prompt = CUSTOM_ENTITY_EXTRACTION_PROMPT.format(input_text=chunk)
        
        # 调用LLM
        response = await llm_model_func(prompt, **kwargs)
        
        # 解析响应
        try:
            result = json.loads(response)
            chunk_entities = result.get("entities", [])
            
            # 后处理：实体标准化和过滤
            for entity in chunk_entities:
                if len(entity["name"]) > 2:  # 过滤过短的实体
                    entity["source_chunk"] = chunk[:100]  # 添加来源信息
                    entities.append(entity)
                    
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entity extraction response: {response}")
    
    return entities

# 在LightRAG中使用自定义实体抽取
rag = LightRAG(
    working_dir="./custom_rag",
    entity_extract_func=custom_entity_extraction,
    # 其他配置...
)
```

### 2.2 自定义关系抽取

```python
# 自定义关系抽取Prompt
CUSTOM_RELATION_EXTRACTION_PROMPT = """
基于已识别的实体，从文本中抽取它们之间的关系：

文本：{input_text}

已识别实体：{entities}

请识别以下类型的关系：
1. 依赖关系：A依赖于B、A使用B
2. 层次关系：A是B的子类、A属于B
3. 协作关系：A与B协作、A集成B
4. 竞争关系：A与B竞争、A替代B

输出格式：
{{"relations": [
    {{"source": "源实体", "target": "目标实体", "relation": "关系类型", "description": "关系描述", "strength": 0.8}},
    ...
]}}
"""

async def custom_relation_extraction(
    text_chunks: list[str],
    entities: list[dict],
    llm_model_func,
    **kwargs
) -> list[dict]:
    """自定义关系抽取函数"""
    relations = []
    
    for chunk in text_chunks:
        # 找到该文本块中的实体
        chunk_entities = [e for e in entities if chunk in e.get("source_chunk", "")]
        
        if len(chunk_entities) < 2:
            continue
            
        entity_names = [e["name"] for e in chunk_entities]
        
        # 使用自定义Prompt
        prompt = CUSTOM_RELATION_EXTRACTION_PROMPT.format(
            input_text=chunk,
            entities=", ".join(entity_names)
        )
        
        # 调用LLM
        response = await llm_model_func(prompt, **kwargs)
        
        # 解析和后处理
        try:
            result = json.loads(response)
            chunk_relations = result.get("relations", [])
            
            for relation in chunk_relations:
                # 验证关系的实体是否存在
                if (relation["source"] in entity_names and 
                    relation["target"] in entity_names):
                    relation["source_chunk"] = chunk[:100]
                    relations.append(relation)
                    
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse relation extraction response: {response}")
    
    return relations
```

### 2.3 领域特定的图谱构建

```python
class DomainSpecificRAG(LightRAG):
    """领域特定的RAG实现"""
    
    def __init__(self, domain: str, **kwargs):
        self.domain = domain
        super().__init__(**kwargs)
        
        # 根据领域选择不同的抽取策略
        if domain == "technology":
            self.entity_extract_func = self._tech_entity_extraction
            self.relation_extract_func = self._tech_relation_extraction
        elif domain == "medical":
            self.entity_extract_func = self._medical_entity_extraction
            self.relation_extract_func = self._medical_relation_extraction
        elif domain == "legal":
            self.entity_extract_func = self._legal_entity_extraction
            self.relation_extract_func = self._legal_relation_extraction
    
    async def _tech_entity_extraction(self, text_chunks, llm_model_func, **kwargs):
        """技术领域实体抽取"""
        # 技术领域特定的实体抽取逻辑
        pass
    
    async def _medical_entity_extraction(self, text_chunks, llm_model_func, **kwargs):
        """医疗领域实体抽取"""
        # 医疗领域特定的实体抽取逻辑
        pass
    
    async def _legal_entity_extraction(self, text_chunks, llm_model_func, **kwargs):
        """法律领域实体抽取"""
        # 法律领域特定的实体抽取逻辑
        pass

# 使用示例
tech_rag = DomainSpecificRAG(
    domain="technology",
    working_dir="./tech_rag",
    llm_model_func=your_llm_function
)
```

## 3. 定制化知识库构建

### 3.1 自定义文档处理管道

```python
class CustomDocumentProcessor:
    """自定义文档处理器"""
    
    def __init__(self, rag_instance):
        self.rag = rag_instance
        self.preprocessors = []
        self.postprocessors = []
    
    def add_preprocessor(self, func):
        """添加预处理函数"""
        self.preprocessors.append(func)
    
    def add_postprocessor(self, func):
        """添加后处理函数"""
        self.postprocessors.append(func)
    
    async def process_document(self, content: str, metadata: dict = None):
        """处理单个文档"""
        # 预处理
        for preprocessor in self.preprocessors:
            content = await preprocessor(content, metadata)
        
        # 核心处理
        result = await self.rag.ainsert(content)
        
        # 后处理
        for postprocessor in self.postprocessors:
            result = await postprocessor(result, content, metadata)
        
        return result

# 预处理函数示例
async def remove_noise(content: str, metadata: dict) -> str:
    """移除文档中的噪声内容"""
    # 移除页眉页脚
    content = re.sub(r'页码：\d+', '', content)
    # 移除多余空白
    content = re.sub(r'\n\s*\n', '\n\n', content)
    return content

async def add_metadata_context(content: str, metadata: dict) -> str:
    """添加元数据上下文"""
    if metadata:
        context = f"文档信息：{metadata.get('title', '')} - {metadata.get('author', '')}\n\n"
        content = context + content
    return content

# 后处理函数示例
async def validate_extraction_quality(result, content: str, metadata: dict):
    """验证抽取质量"""
    if result and hasattr(result, 'entities'):
        entity_count = len(result.entities)
        content_length = len(content)
        
        # 检查实体密度
        if entity_count / content_length < 0.001:
            logger.warning(f"Low entity density detected: {entity_count}/{content_length}")
    
    return result

# 使用示例
processor = CustomDocumentProcessor(rag)
processor.add_preprocessor(remove_noise)
processor.add_preprocessor(add_metadata_context)
processor.add_postprocessor(validate_extraction_quality)

# 处理文档
await processor.process_document(document_content, {"title": "技术文档", "author": "张三"})
```

### 3.2 自定义分块策略

```python
class CustomChunker:
    """自定义分块器"""
    
    def __init__(self, strategy: str = "semantic"):
        self.strategy = strategy
    
    def chunk_by_semantic(self, content: str, max_tokens: int = 1000) -> list[str]:
        """基于语义的分块"""
        sentences = self._split_sentences(content)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def chunk_by_structure(self, content: str) -> list[str]:
        """基于文档结构的分块"""
        # 按标题分块
        sections = re.split(r'\n#+\s+', content)
        chunks = []
        
        for section in sections:
            if len(section.strip()) > 100:  # 过滤过短的章节
                chunks.append(section.strip())
        
        return chunks
    
    def chunk_by_topic(self, content: str, llm_func) -> list[str]:
        """基于主题的分块"""
        # 使用LLM识别主题边界
        prompt = f"""
        请将以下文本按主题分割，标出分割点：
        
        {content}
        
        在每个主题分割点前添加标记：[TOPIC_BREAK]
        """
        
        response = llm_func(prompt)
        sections = response.split("[TOPIC_BREAK]")
        
        return [section.strip() for section in sections if section.strip()]

# 集成到LightRAG
class CustomLightRAG(LightRAG):
    def __init__(self, chunking_strategy: str = "semantic", **kwargs):
        super().__init__(**kwargs)
        self.chunker = CustomChunker(chunking_strategy)
    
    async def ainsert(self, content: str, **kwargs):
        # 使用自定义分块
        if self.chunker.strategy == "semantic":
            chunks = self.chunker.chunk_by_semantic(content)
        elif self.chunker.strategy == "structure":
            chunks = self.chunker.chunk_by_structure(content)
        elif self.chunker.strategy == "topic":
            chunks = self.chunker.chunk_by_topic(content, self.llm_model_func)
        else:
            chunks = [content]  # 默认不分块
        
        # 处理每个分块
        results = []
        for chunk in chunks:
            result = await super().ainsert(chunk, **kwargs)
            results.append(result)
        
        return results
```

## 4. 多知识库管理实现

### 4.1 多知识库管理器

```python
class MultiKnowledgeBaseManager:
    """多知识库管理器"""
    
    def __init__(self, base_config: dict = None):
        self.knowledge_bases = {}
        self.base_config = base_config or {}
        self.file_manager = FileManager()
    
    def create_knowledge_base(
        self, 
        name: str, 
        config: dict = None,
        domain: str = None
    ) -> LightRAG:
        """创建新的知识库"""
        if name in self.knowledge_bases:
            raise ValueError(f"Knowledge base '{name}' already exists")
        
        # 合并配置
        kb_config = {**self.base_config, **(config or {})}
        kb_config["working_dir"] = f"./kb_{name}"
        
        # 根据领域创建特定的RAG实例
        if domain:
            kb = DomainSpecificRAG(domain=domain, **kb_config)
        else:
            kb = LightRAG(**kb_config)
        
        self.knowledge_bases[name] = {
            "instance": kb,
            "config": kb_config,
            "domain": domain,
            "created_at": datetime.now(),
            "file_refs": set()
        }
        
        return kb
    
    def get_knowledge_base(self, name: str) -> LightRAG:
        """获取知识库实例"""
        if name not in self.knowledge_bases:
            raise ValueError(f"Knowledge base '{name}' not found")
        return self.knowledge_bases[name]["instance"]
    
    def list_knowledge_bases(self) -> list[dict]:
        """列出所有知识库"""
        return [
            {
                "name": name,
                "domain": info["domain"],
                "created_at": info["created_at"],
                "file_count": len(info["file_refs"])
            }
            for name, info in self.knowledge_bases.items()
        ]
    
    async def add_file_to_kb(self, file_id: str, kb_name: str, metadata: dict = None):
        """将文件添加到知识库"""
        kb = self.get_knowledge_base(kb_name)
        file_content = await self.file_manager.get_file_content(file_id)
        
        # 添加文件引用
        self.knowledge_bases[kb_name]["file_refs"].add(file_id)
        
        # 插入内容到知识库
        result = await kb.ainsert(file_content)
        
        # 记录文件-知识库关联
        await self._record_file_kb_association(file_id, kb_name, metadata)
        
        return result
    
    async def query_knowledge_base(
        self, 
        kb_name: str, 
        query: str, 
        **kwargs
    ) -> str:
        """查询特定知识库"""
        kb = self.get_knowledge_base(kb_name)
        return await kb.aquery(query, **kwargs)
    
    async def query_multiple_kbs(
        self, 
        kb_names: list[str], 
        query: str, 
        merge_strategy: str = "weighted"
    ) -> str:
        """查询多个知识库并合并结果"""
        results = []
        
        for kb_name in kb_names:
            try:
                result = await self.query_knowledge_base(kb_name, query)
                results.append({
                    "kb_name": kb_name,
                    "result": result,
                    "domain": self.knowledge_bases[kb_name]["domain"]
                })
            except Exception as e:
                logger.warning(f"Failed to query KB {kb_name}: {e}")
        
        # 合并结果
        if merge_strategy == "weighted":
            return await self._weighted_merge_results(results, query)
        elif merge_strategy == "ranked":
            return await self._ranked_merge_results(results, query)
        else:
            return await self._simple_merge_results(results)
    
    async def _weighted_merge_results(self, results: list[dict], query: str) -> str:
        """加权合并结果"""
        # 根据领域相关性和结果质量进行加权合并
        merged_content = []
        
        for result in results:
            domain = result["domain"]
            content = result["result"]
            
            # 计算领域权重（示例）
            domain_weight = self._calculate_domain_weight(domain, query)
            
            merged_content.append(f"[来源: {result['kb_name']}] (权重: {domain_weight:.2f})\n{content}")
        
        return "\n\n".join(merged_content)
    
    def _calculate_domain_weight(self, domain: str, query: str) -> float:
        """计算领域权重"""
        # 简单的关键词匹配权重计算
        domain_keywords = {
            "technology": ["技术", "开发", "编程", "软件", "系统"],
            "medical": ["医疗", "健康", "疾病", "治疗", "药物"],
            "legal": ["法律", "法规", "合同", "诉讼", "权利"]
        }
        
        if domain not in domain_keywords:
            return 1.0
        
        keywords = domain_keywords[domain]
        matches = sum(1 for keyword in keywords if keyword in query)
        
        return 1.0 + (matches * 0.2)  # 基础权重1.0，每个匹配关键词增加0.2

# 使用示例
manager = MultiKnowledgeBaseManager({
    "llm_model_func": your_llm_function,
    "embedding_func": your_embedding_function
})

# 创建不同领域的知识库
tech_kb = manager.create_knowledge_base("tech_docs", domain="technology")
medical_kb = manager.create_knowledge_base("medical_docs", domain="medical")

# 添加文件到知识库
await manager.add_file_to_kb("tech_file_001", "tech_docs")
await manager.add_file_to_kb("medical_file_001", "medical_docs")

# 查询特定知识库
tech_result = await manager.query_knowledge_base("tech_docs", "什么是微服务架构？")

# 查询多个知识库
multi_result = await manager.query_multiple_kbs(
    ["tech_docs", "medical_docs"], 
    "人工智能在医疗中的应用",
    merge_strategy="weighted"
)
```

## 5. 存储后端扩展

### 5.1 自定义存储实现

```python
from lightrag.storage import BaseKVStorage, BaseVectorStorage, BaseGraphStorage

class CustomKVStorage(BaseKVStorage):
    """自定义键值存储实现"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = self._create_client()

    def _create_client(self):
        # 创建自定义存储客户端
        pass

    async def all_keys(self) -> list[str]:
        """获取所有键"""
        # 实现获取所有键的逻辑
        pass

    async def get_by_id(self, id: str):
        """根据ID获取值"""
        # 实现根据ID获取值的逻辑
        pass

    async def get_by_ids(self, ids: list[str]) -> dict:
        """批量获取值"""
        # 实现批量获取的逻辑
        pass

    async def filter_keys(self, filter_func) -> list[str]:
        """过滤键"""
        # 实现键过滤逻辑
        pass

    async def upsert(self, data: dict):
        """插入或更新数据"""
        # 实现数据插入/更新逻辑
        pass

    async def drop(self):
        """删除存储"""
        # 实现存储删除逻辑
        pass

class CustomVectorStorage(BaseVectorStorage):
    """自定义向量存储实现"""

    def __init__(self, connection_string: str, embedding_dim: int):
        self.connection_string = connection_string
        self.embedding_dim = embedding_dim
        self.client = self._create_client()

    async def upsert(self, data: dict):
        """插入或更新向量数据"""
        # 实现向量数据插入/更新逻辑
        pass

    async def query(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """向量相似度查询"""
        # 实现向量查询逻辑
        pass

# 使用自定义存储
rag = LightRAG(
    working_dir="./custom_storage_rag",
    kv_storage=CustomKVStorage("your_connection_string"),
    vector_storage=CustomVectorStorage("your_vector_db_string", 1024),
    # 其他配置...
)
```

### 5.2 分布式存储支持

```python
class DistributedKVStorage(BaseKVStorage):
    """分布式键值存储"""

    def __init__(self, nodes: list[str], replication_factor: int = 3):
        self.nodes = nodes
        self.replication_factor = replication_factor
        self.clients = [self._create_client(node) for node in nodes]
        self.consistent_hash = ConsistentHash(nodes)

    def _get_nodes_for_key(self, key: str) -> list[str]:
        """获取键对应的存储节点"""
        return self.consistent_hash.get_nodes(key, self.replication_factor)

    async def upsert(self, data: dict):
        """分布式插入数据"""
        tasks = []
        for key, value in data.items():
            nodes = self._get_nodes_for_key(key)
            for node in nodes:
                client = self._get_client_for_node(node)
                tasks.append(client.upsert({key: value}))

        await asyncio.gather(*tasks)

    async def get_by_id(self, id: str):
        """分布式获取数据"""
        nodes = self._get_nodes_for_key(id)

        for node in nodes:
            try:
                client = self._get_client_for_node(node)
                result = await client.get_by_id(id)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Failed to get data from node {node}: {e}")

        return None
```

## 6. 高级查询和检索定制

### 6.1 自定义查询策略

```python
class CustomQueryEngine:
    """自定义查询引擎"""

    def __init__(self, rag_instance):
        self.rag = rag_instance
        self.query_strategies = {
            "precise": self._precise_query,
            "exploratory": self._exploratory_query,
            "analytical": self._analytical_query,
            "creative": self._creative_query
        }

    async def query(self, query: str, strategy: str = "precise", **kwargs):
        """根据策略执行查询"""
        if strategy not in self.query_strategies:
            raise ValueError(f"Unknown query strategy: {strategy}")

        return await self.query_strategies[strategy](query, **kwargs)

    async def _precise_query(self, query: str, **kwargs):
        """精确查询：注重准确性和相关性"""
        # 使用更严格的相似度阈值
        kwargs["similarity_threshold"] = kwargs.get("similarity_threshold", 0.8)
        kwargs["top_k"] = kwargs.get("top_k", 3)

        return await self.rag.aquery(query, param=QueryParam(mode="local", **kwargs))

    async def _exploratory_query(self, query: str, **kwargs):
        """探索性查询：获取更广泛的信息"""
        kwargs["similarity_threshold"] = kwargs.get("similarity_threshold", 0.3)
        kwargs["top_k"] = kwargs.get("top_k", 10)

        return await self.rag.aquery(query, param=QueryParam(mode="global", **kwargs))

    async def _analytical_query(self, query: str, **kwargs):
        """分析性查询：结合多种检索模式"""
        # 先进行局部查询
        local_result = await self.rag.aquery(
            query,
            param=QueryParam(mode="local", top_k=5)
        )

        # 再进行全局查询
        global_result = await self.rag.aquery(
            query,
            param=QueryParam(mode="global", top_k=5)
        )

        # 合并结果
        return f"局部分析：\n{local_result}\n\n全局分析：\n{global_result}"

    async def _creative_query(self, query: str, **kwargs):
        """创造性查询：鼓励发散思维"""
        # 使用混合模式，降低相似度阈值
        kwargs["similarity_threshold"] = kwargs.get("similarity_threshold", 0.2)
        kwargs["temperature"] = kwargs.get("temperature", 0.8)

        return await self.rag.aquery(query, param=QueryParam(mode="hybrid", **kwargs))

# 使用示例
query_engine = CustomQueryEngine(rag)

# 不同策略的查询
precise_answer = await query_engine.query("什么是微服务架构？", strategy="precise")
exploratory_answer = await query_engine.query("微服务相关技术", strategy="exploratory")
analytical_answer = await query_engine.query("微服务的优缺点", strategy="analytical")
creative_answer = await query_engine.query("如何创新微服务架构？", strategy="creative")
```

### 6.2 智能查询路由

```python
class IntelligentQueryRouter:
    """智能查询路由器"""

    def __init__(self, knowledge_bases: dict, llm_func):
        self.knowledge_bases = knowledge_bases
        self.llm_func = llm_func
        self.domain_classifier = self._build_domain_classifier()

    def _build_domain_classifier(self):
        """构建领域分类器"""
        return {
            "technology": ["技术", "开发", "编程", "软件", "系统", "架构", "算法"],
            "medical": ["医疗", "健康", "疾病", "治疗", "药物", "症状", "诊断"],
            "legal": ["法律", "法规", "合同", "诉讼", "权利", "义务", "条款"],
            "finance": ["金融", "投资", "股票", "基金", "保险", "银行", "理财"],
            "education": ["教育", "学习", "课程", "教学", "培训", "考试", "学校"]
        }

    async def route_query(self, query: str) -> dict:
        """智能路由查询"""
        # 1. 领域分类
        domains = await self._classify_query_domain(query)

        # 2. 查询意图识别
        intent = await self._identify_query_intent(query)

        # 3. 选择合适的知识库和策略
        selected_kbs = self._select_knowledge_bases(domains)
        query_strategy = self._select_query_strategy(intent)

        # 4. 执行查询
        results = {}
        for kb_name in selected_kbs:
            if kb_name in self.knowledge_bases:
                kb = self.knowledge_bases[kb_name]
                result = await kb.aquery(query, param=QueryParam(mode=query_strategy))
                results[kb_name] = result

        # 5. 合并和排序结果
        final_result = await self._merge_and_rank_results(results, query, intent)

        return {
            "answer": final_result,
            "domains": domains,
            "intent": intent,
            "knowledge_bases_used": selected_kbs,
            "strategy": query_strategy
        }

    async def _classify_query_domain(self, query: str) -> list[str]:
        """分类查询领域"""
        domains = []

        for domain, keywords in self.domain_classifier.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > 0:
                domains.append((domain, score))

        # 按分数排序，返回前3个领域
        domains.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, _ in domains[:3]]

    async def _identify_query_intent(self, query: str) -> str:
        """识别查询意图"""
        intent_prompt = f"""
        请分析以下查询的意图类型：

        查询：{query}

        意图类型：
        1. factual - 寻求事实信息
        2. analytical - 需要分析和比较
        3. procedural - 寻求操作步骤
        4. creative - 需要创意和建议
        5. troubleshooting - 解决问题

        请只返回意图类型（如：factual）
        """

        response = await self.llm_func(intent_prompt)
        return response.strip().lower()

    def _select_knowledge_bases(self, domains: list[str]) -> list[str]:
        """选择合适的知识库"""
        selected = []

        for domain in domains:
            # 根据领域映射到知识库
            if domain == "technology":
                selected.extend(["tech_kb", "dev_kb"])
            elif domain == "medical":
                selected.extend(["medical_kb", "health_kb"])
            # 添加更多映射...

        # 去重并限制数量
        return list(set(selected))[:3]

    def _select_query_strategy(self, intent: str) -> str:
        """选择查询策略"""
        strategy_mapping = {
            "factual": "local",
            "analytical": "global",
            "procedural": "hybrid",
            "creative": "mix",
            "troubleshooting": "hybrid"
        }

        return strategy_mapping.get(intent, "local")

# 使用示例
router = IntelligentQueryRouter(knowledge_bases, llm_function)
result = await router.route_query("如何在Python中实现微服务架构？")

print(f"答案：{result['answer']}")
print(f"识别领域：{result['domains']}")
print(f"查询意图：{result['intent']}")
print(f"使用的知识库：{result['knowledge_bases_used']}")
```

## 7. 性能监控和优化

### 7.1 性能监控系统

```python
import time
import psutil
from dataclasses import dataclass
from typing import Dict, List
import asyncio

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: str = None

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, dict] = {}

    async def monitor_operation(self, operation_name: str, func, *args, **kwargs):
        """监控操作性能"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()

        success = True
        error_message = None
        result = None

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Operation {operation_name} failed: {e}")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_cpu = psutil.cpu_percent()

        # 记录指标
        metrics = PerformanceMetrics(
            timestamp=start_time,
            operation=operation_name,
            duration=end_time - start_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=end_cpu - start_cpu,
            success=success,
            error_message=error_message
        )

        self.metrics.append(metrics)
        self._update_operation_stats(operation_name, metrics)

        return result

    def _update_operation_stats(self, operation: str, metrics: PerformanceMetrics):
        """更新操作统计"""
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                "count": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "max_duration": 0,
                "min_duration": float('inf'),
                "success_rate": 0,
                "total_memory": 0,
                "avg_memory": 0
            }

        stats = self.operation_stats[operation]
        stats["count"] += 1
        stats["total_duration"] += metrics.duration
        stats["avg_duration"] = stats["total_duration"] / stats["count"]
        stats["max_duration"] = max(stats["max_duration"], metrics.duration)
        stats["min_duration"] = min(stats["min_duration"], metrics.duration)

        if metrics.success:
            stats["success_rate"] = (stats.get("success_count", 0) + 1) / stats["count"]
            stats["success_count"] = stats.get("success_count", 0) + 1

        stats["total_memory"] += metrics.memory_usage
        stats["avg_memory"] = stats["total_memory"] / stats["count"]

    def get_performance_report(self) -> dict:
        """获取性能报告"""
        return {
            "total_operations": len(self.metrics),
            "operation_stats": self.operation_stats,
            "recent_metrics": self.metrics[-10:] if self.metrics else []
        }

# 性能监控装饰器
def monitor_performance(operation_name: str):
    """性能监控装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = getattr(args[0], '_performance_monitor', None)
            if monitor:
                return await monitor.monitor_operation(operation_name, func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            monitor = getattr(args[0], '_performance_monitor', None)
            if monitor:
                return asyncio.run(monitor.monitor_operation(operation_name, func, *args, **kwargs))
            else:
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# 集成到LightRAG
class MonitoredLightRAG(LightRAG):
    """带性能监控的LightRAG"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._performance_monitor = PerformanceMonitor()

    @monitor_performance("document_insert")
    async def ainsert(self, content: str, **kwargs):
        return await super().ainsert(content, **kwargs)

    @monitor_performance("query")
    async def aquery(self, query: str, **kwargs):
        return await super().aquery(query, **kwargs)

    def get_performance_report(self):
        """获取性能报告"""
        return self._performance_monitor.get_performance_report()

# 使用示例
rag = MonitoredLightRAG(working_dir="./monitored_rag")

# 执行操作
await rag.ainsert("测试文档内容")
result = await rag.aquery("测试查询")

# 获取性能报告
report = rag.get_performance_report()
print(f"总操作数：{report['total_operations']}")
print(f"查询平均耗时：{report['operation_stats'].get('query', {}).get('avg_duration', 0):.2f}秒")
```

这个扩展的定制化开发指南展示了LightRAG在存储扩展、查询定制、性能监控等方面的强大定制能力，为构建企业级RAG应用提供了全面的技术指导。
