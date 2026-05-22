## 第三章：QueryEngine 与 ChatEngine —— RAG 的最终交互层

前两章讲了检索的理论（三阶段漏斗）和组件（Retriever / Reranker / ResponseSynthesizer）。本章把它们串起来，讲解 LlamaIndex 中面向用户的最终交互层——QueryEngine（单轮问答）和 ChatEngine（多轮对话），并完成一个端到端的完整实战。

### 3.1 QueryEngine 是什么

```
═══════════════════════════════════════════════════════════════════════
          QueryEngine 在 RAG 系统中的位置
═══════════════════════════════════════════════════════════════════════

  用户 ──▶ QueryEngine.query("请假需要提前多久？") ──▶ 自然语言回答
                │
                │  内部自动执行:
                │
                ├──▶ ① Retriever.retrieve(query)
                │        └── 从索引中召回相关 Node (Top-K 候选)
                │
                ├──▶ ② NodePostprocessor.postprocess_nodes(nodes)
                │        └── 过滤 + Reranker 精排 + 替换 (Top-N 精选)
                │
                └──▶ ③ ResponseSynthesizer.synthesize(query, nodes)
                         └── 组织上下文 → 填入 Prompt → LLM 生成 → 返回

  QueryEngine 的本质:
    它是 Retriever + NodePostprocessor + ResponseSynthesizer 的"封装器"。
    用户只需要调用 .query("问题")，内部三步自动串联。
```

**QueryEngine 内部调用流程（源码逻辑简化）：**

```python
# 这是 QueryEngine 内部的 query() 方法的简化版逻辑:
def query(self, query_str: str) -> Response:
    # Step 1: 检索 (Retriever)
    nodes_with_scores = self.retriever.retrieve(query_str)
    
    # Step 2: 后处理 (NodePostprocessor) —— 可选
    for postprocessor in self._node_postprocessors:
        nodes_with_scores = postprocessor.postprocess_nodes(nodes_with_scores, query_str)
    
    # Step 3: 合成 (ResponseSynthesizer)
    response = self._response_synthesizer.synthesize(query_str, nodes_with_scores)
    
    return response
```

**三个关键概念：**

| 概念 | 含义 |
|------|------|
| `retriever` | 决定"从哪找、找多少"——向量/BM25/混合, Top-20/50/100 |
| `node_postprocessors` | 决定"怎么筛、怎么排"——过滤阈值、Reranker精排、元数据替换 |
| `response_synthesizer` | 决定"怎么组织、怎么生成"——拼接上下文、填充Prompt、调用LLM |

---

### 3.2 as_query_engine() —— 工厂方法完整解析

`index.as_query_engine()` 是实际开发中最常用的创建 QueryEngine 的方式。和 `as_retriever()` 一样，它是工厂方法——从 Index 实例中提取组件，自动装配为 QueryEngine。

**内部做了什么：**

```
  index.as_query_engine(similarity_top_k=5, ...)
          │
          ▼
  Step 1: 如果没有传 retriever → 自动调用 index.as_retriever(**kwargs)
          将 similarity_top_k, filters, alpha 等参数传给 Retriever 工厂
          → 返回 VectorIndexRetriever (默认)

  Step 2: 如果没有传 node_postprocessors → 默认为空列表 []
          (不在工厂方法里自动加 Reranker, 需要显式传入)

  Step 3: 如果没有传 response_synthesizer → 自动创建
          默认 response_mode = "compact"

  Step 4: 组装 RetrieverQueryEngine(retriever, postprocessors, synthesizer)
          → 返回
```

**完整参数表：**

| 参数 | 默认值 | 传给谁 | 说明 |
|------|--------|--------|------|
| `similarity_top_k` | 2 | Retriever | 粗排取多少个候选 |
| `retriever` | None(自动) | — | 如果传了就跳过自动创建 |
| `filters` | None | Retriever | metadata 过滤条件 |
| `node_postprocessors` | [] | NodePostprocessor | Reranker / 过滤器的列表 |
| `response_mode` | "compact" | ResponseSynthesizer | 合成策略 |
| `streaming` | False | ResponseSynthesizer | 是否流式输出 |
| `text_qa_template` | None | ResponseSynthesizer | 自定义 Prompt 模板 |
| `alpha` | None | Retriever | 混合检索权重 |
| `vector_store_query_mode` | "default" | Retriever | 向量库查询模式 |

**三种典型用法：**

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# ═══════════════════════════════════════════════════════════════
# 用法 1: 最简原型 —— 只设 top_k，其他全用默认
# ═══════════════════════════════════════════════════════════════
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("事假需要提前多久申请？")
# 内部: Retriever(向量, Top-3) → 无后处理 → Compact合成 → 回答

# ═══════════════════════════════════════════════════════════════
# 用法 2: 加 Reranker —— 生产环境的标准配置
# ═══════════════════════════════════════════════════════════════
# 粗排 Top-20 → Reranker → 精排 Top-3 → 合成
query_engine = index.as_query_engine(
    similarity_top_k=20,               # 粗排多取
    node_postprocessors=[
        SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=3,                    # 精排后只取 3 个
        ),
    ],
    response_mode="compact",           # 合成策略
    streaming=True,                    # 流式输出
)
response = query_engine.query("加班费怎么算的？")

# ═══════════════════════════════════════════════════════════════
# 用法 3: 完整自定义 —— Retriever + Reranker + 自定义 Prompt
# ═══════════════════════════════════════════════════════════════
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts import PromptTemplate

# 自定义 Prompt
custom_template = PromptTemplate(
    "你是星辰科技的员工助手。请用简洁的中文回答，不要编造信息。\n"
    "参考资料:\n{context_str}\n"
    "用户问题: {query_str}\n"
    "回答:"
)

# 显式创建组件
retriever = index.as_retriever(
    similarity_top_k=30,
    filters=MetadataFilters(
        filters=[MetadataFilter(key="status", value="active")]
    ),
)
reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=3)
synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=custom_template,
    streaming=True,
)

query_engine = index.as_query_engine(
    retriever=retriever,
    node_postprocessors=[reranker],
    response_synthesizer=synthesizer,
)
```

#### 3.2.1 QueryEngine 的响应模式

QueryEngine 支持与 ResponseSynthesizer 相同的五种响应模式。通过 `response_mode` 参数设置：

| response_mode | 何时选用 |
|---------------|----------|
| `"compact"` | **默认推荐**——先尝试拼接所有Node，超窗口就压缩后一次LLM调用 |
| `"refine"` | 需要极致质量（法律/医疗），逐条迭代精炼 |
| `"tree_summarize"` | 检索到8+个Node时需要综合大量信息 |
| `"simple_summarize"` | 1-3个Node，追求最快速度 |
| `"no_text"` | 测试基线，不使用检索结果 |

---

### 3.3 ChatEngine —— 多轮对话

QueryEngine 处理的是"一问一答"。但真实用户不会只说一句话——他们会在对话中引用前文（"那个规定具体是什么？"），纠正错误（"不是病假，是事假"），或者追问细节（"那最少要提前几天？"）。

ChatEngine 就是为多轮对话设计的。它在 QueryEngine 的基础上增加了**对话记忆（ChatMemory）**。

```
═══════════════════════════════════════════════════════════════════════
          ChatEngine 的多轮对话处理
═══════════════════════════════════════════════════════════════════════

  第 1 轮:
    User: "请假需要提前多久？"
    ChatEngine:
      1. 查询 ChatMemory → 无历史记录
      2. 调用 QueryEngine.query("请假需要提前多久？") → "事假需提前1个工作日"
      3. 将 (User, Assistant) 存入 ChatMemory
      4. 返回 "事假需提前1个工作日"

  第 2 轮:
    User: "那病假呢？"  ← "那" 指代不明, "病假"是关键
    ChatEngine:
      1. 查询 ChatMemory → [上轮: User"请假需要提前多久", Assistant"事假需..."]
      2. Condense (压缩): 将历史 + 当前问题 压缩为一个独立查询
         → "病假需要提前多久申请？"  ← 消解了"那"的指代
      3. 调用 QueryEngine.query("病假需要提前多久申请？") → "病假应在8:30前通知"
      4. 将 (User, Assistant) 存入 ChatMemory
      5. 返回 "病假应在当日8:30前通知部门主管"

  第 3 轮:
    User: "如果连续超过2天呢？"
    ChatEngine:
      1. Condense: 历史 + 当前问题 → "连续病假超过2天有什么规定？"
      2. QueryEngine → "连续超过2天须提供二级及以上医院证明"
      3. 存入 ChatMemory → 返回结果
```

#### 3.3.1 as_chat_engine() 工厂方法

```python
from llama_index.core.memory import ChatMemoryBuffer

# ═══════════════════════════════════════════════════════════════
# 用法 1: 最简 —— 使用默认 ChatMemory
# ═══════════════════════════════════════════════════════════════
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",  # 聊天模式
    similarity_top_k=5,
    verbose=True,                   # 打印内部处理过程
)

response = chat_engine.chat("请假需要提前多久？")
response = chat_engine.chat("那病假呢？")       # 自动理解指代
response = chat_engine.chat("年假有多少天？")    # 继续积累上下文

# ═══════════════════════════════════════════════════════════════
# 用法 2: 自定义 ChatMemory —— 控制记忆大小
# ═══════════════════════════════════════════════════════════════
memory = ChatMemoryBuffer.from_defaults(
    token_limit=4000,   # 最多保留 4000 token 的历史 (超出自动丢弃旧的)
)

chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=memory,
    similarity_top_k=5,
)

# ═══════════════════════════════════════════════════════════════
# 用法 3: 完全自定义 —— Retriever + Reranker + 记忆
# ═══════════════════════════════════════════════════════════════
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=ChatMemoryBuffer.from_defaults(token_limit=4000),
    retriever=index.as_retriever(similarity_top_k=20),
    node_postprocessors=[
        SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=3),
    ],
    response_mode="compact",
)
```

#### 3.3.2 聊天模式详解

| chat_mode | 工作原理 | 适用场景 |
|-----------|---------|----------|
| **`condense_question`** | 每次chat时，把历史+当前问题压缩为一个独立查询。然后用这个独立查询调用QueryEngine。 | 多轮对话中，用户经常使用指代（"那个""它"） |
| **`context`** | 把历史+检索结果一起拼接为上下文。LLM直接基于完整上下文回答。 | 对话轮次少，历史不长 |
| **`react`** | 使用 ReAct Agent 模式。Agent决定"是否需要检索"、"什么时候检索"。 | 需要Agent自主判断的复杂对话 |
| **`best`** | 自动选择最合适的模式。 | 不确定时用这个 |

**condense_question 的内部逻辑（最重要的模式）：**

```
  用户当前问题: "那最少要提前几天？"
  历史记录:
    User: "请假怎么申请？"
    Assistant: "事假需提前1个工作日向部门主管申请。病假应在8:30前通知。"

  Condense 阶段:
    构造一个 LLM Prompt:
      "以下是用户和助手的对话历史:
       User: 请假怎么申请？
       Assistant: 事假需提前1个工作日...
       
       基于上述历史，将用户的后续问题改写为一个独立的、完整的查询。
       后续问题: 那最少要提前几天？
       独立查询:"

     LLM 输出: "事假最少需要提前几天申请？"  ← 消解了"那"的指代

  然后用这个"独立查询"去调用 QueryEngine.retrieve() → 和和单轮完全一样!
```

---

### 3.4 端到端实战：向量检索 + BM25 + RRF 融合 + LLM 回答

```python
"""
═══════════════════════════════════════════════════════════════════════════
  完整实战: 向量检索 + BM25 + RRF融合 + LLM回答
═══════════════════════════════════════════════════════════════════════════

  流程:
    1. 加载已有 Chroma 向量库 (数据已在第二阶段第五章入库)
    2. 创建向量检索器 + BM25 检索器
    3. 用 HybridRetriever (RRF模式) 融合两路结果
    4. 加 Cross-Encoder Reranker 精排
    5. 用 Compact 合成策略生成最终回答
    6. 展示来源溯源

  召回与精排的推荐比值:
    粗排 (召回): 20-50 个候选   →  高召回, 确保不漏
    精排 (Rerank): 3-5 个最终   →  高精度, 确保前三最相关
    比值: 粗排:精排 ≈ 5:1 ~ 10:1

    为什么是这个比值:
    · 粗排太少 (< 10): 可能漏掉正确答案
    · 粗排太多 (> 100): Reranker 处理太慢, 成本太高
    · 精排太多 (> 5): LLM 上下文太大, 成本增加, 噪声增加
    · 精排太少 (= 1): 只有一个候选, 没有选择余地
    
    推荐起步值: 粗排 30 → 精排 3 (10:1 比值)
    追求高召回: 粗排 50 → 精排 5
    追求低成本: 粗排 15 → 精排 3
"""
import chromadb
from llama_index.core import (
    Settings, VectorStoreIndex, StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import FlatReader
from pathlib import Path

# ── 全局配置 ──────────────────────────────────────────────────
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", dimensions=512
)
Settings.llm = OpenAI(model="gpt-4o", temperature=0)


# ═══════════════════════════════════════════════════════════════
# Step 1: 加载文档 + 切分 (如果没有现成 Chroma 库)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("  端到端实战: 向量 + BM25 混合检索 + Rerank + LLM")
print("=" * 70)

# 加载文档
reader = FlatReader()
documents = reader.load_data(Path("星辰科技员工手册.md"))
document = documents[0]

# 注入元数据
document.metadata.update({
    "category": "人事政策",
    "department": "HR",
    "version": "V3.0",
    "status": "active",
    "created_date": "2024-03-15",
})

# 切分
splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents([document])
print(f"  [加载] 1 个 Document → {len(nodes)} 个 Node")

# ═══════════════════════════════════════════════════════════════
# Step 2: 构建向量索引
# ═══════════════════════════════════════════════════════════════
print(f"\n  [索引] 构建向量索引...")

# 创建 Chroma 向量库 (内存模式, 演示用)
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("end_to_end_demo")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
print(f"  [索引] 完成, 共 {len(nodes)} 个向量")

# ═══════════════════════════════════════════════════════════════
# Step 3: 创建双路检索器
# ═══════════════════════════════════════════════════════════════
from llama_index.core.retrievers import (
    VectorIndexRetriever, BM25Retriever, HybridRetriever,
)

# 向量检索器: 取 Top-30 (粗排, 高召回)
vector_retriever = index.as_retriever(similarity_top_k=30)
# ↑ 为什么是 30?
#   需要给 Reranker 足够的候选去精排。
#   30 个候选能覆盖绝大多数情况下 Top-3 正确答案。

# BM25 检索器: 也取 Top-30
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=30,
)

# ═══════════════════════════════════════════════════════════════
# Step 4: 混合检索器 (RRF 融合)
# ═══════════════════════════════════════════════════════════════
hybrid_retriever = HybridRetriever.from_defaults(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rerank",    # RRF 融合
    k=60,
    top_k=30,                    # 融合后仍保留 30 个候选
)

# 演示: 看看混合检索的中间结果
print(f"\n  [混合检索] 中间结果展示")
query = "事假需要提前多久申请？"
hybrid_results = hybrid_retriever.retrieve(query)
print(f"  Query: {query}")
print(f"  融合后候选数: {len(hybrid_results)}")
for i, node_result in enumerate(hybrid_results[:5]):
    print(f"    [{i+1}] RRF={node_result.score:.4f} | "
          f"{node_result.text[:60]}...")

# ═══════════════════════════════════════════════════════════════
# Step 5: 加 Reranker 精排
# ═══════════════════════════════════════════════════════════════
from llama_index.core.postprocessor import SentenceTransformerRerank

try:
    # 本地 Cross-Encoder 精排
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=3,                     # 从 30 个候选中选最好的 3 个
        device="cpu",                # 如果没有 GPU 改用 "cpu"
    )
    print(f"  [Reranker] Cross-Encoder (bge-reranker-v2-m3), Top-30 → Top-3")
except:
    # 如果没有装 bge-reranker, 用相似度后处理器模拟
    from llama_index.core.postprocessor import SimilarityPostprocessor
    reranker = SimilarityPostprocessor(similarity_cutoff=0.3)
    print(f"  [Reranker] 降级为相似度过滤 (similarity_cutoff=0.3)")

# ═══════════════════════════════════════════════════════════════
# Step 6: 组装 QueryEngine
# ═══════════════════════════════════════════════════════════════
query_engine = index.as_query_engine(
    retriever=hybrid_retriever,               # 混合检索
    node_postprocessors=[reranker],            # Reranker 精排
    response_mode="compact",                   # 合成策略
    streaming=False,                           # 非流式
)

print(f"  [QueryEngine] 组装完成:")
print(f"    搜索器: HybridRetriever (向量+BM25, RRF融合)")
print(f"    后处理: Reranker (Top-30→Top-3)")
print(f"    合成: Compact")

# ═══════════════════════════════════════════════════════════════
# Step 7: 测试查询
# ═══════════════════════════════════════════════════════════════

test_queries = [
    "事假需要提前多久申请？",
    "病假和年假分别有什么规定？",         # 需要多个 Node 的信息
    "加班费怎么计算的？法定节假日呢？",     # 需要精确数字
    "五险一金包括哪些？公司和个人的比例？",  # 表格数据
    "报销流程是怎样的？超过5000元怎么办？", # 多步骤流程
]

for i, query in enumerate(test_queries, 1):
    print(f"\n  {'─'*60}")
    print(f"  查询 {i}: {query}")
    print(f"  {'─'*60}")

    response = query_engine.query(query)

    print(f"  回答: {response}")
    print(f"  来源 ({len(response.source_nodes)} 个):")
    for j, src in enumerate(response.source_nodes, 1):
        source = src.metadata.get("file_name", "?")
        page = src.metadata.get("chunk_index", "?")
        print(f"    [{j}] {source} chunk#{page} "
              f"score={src.score:.3f}")
        print(f"        内容: {src.text[:80]}...")
    print()

# ═══════════════════════════════════════════════════════════════
# Step 8: 召回与精排比值的实际验证
# ═══════════════════════════════════════════════════════════════
print("  ── 召回与精排比值说明 ──")
print(f"""
  本示例使用的比值:
    粗排(向量):      Top-30
    粗排(BM25):      Top-30
    RRF 融合后:       Top-30 (两路去重+融合)
    Reranker 精排:    Top-3
    粗排:精排 = 30:3 = 10:1

  为什么 10:1 是好的起步值:
    · 30 个候选中, 正确答案几乎一定在其中 (Recall > 98%)
    · Reranker 处理 30 个候选 = 30 × 5ms = 150ms (可接受)
    · 精排 Top-3 → LLM 输入 3 个 Chunk (约 1200 tokens)
      → 成本合理, 信息充足

  不同场景的推荐比值:
    通用知识库:     粗排 30 → 精排 3  (10:1)  ← 起步
    高精度(法律):   粗排 50 → 精排 5  (10:1)  ← 更多候选+更多输出
    低成本(FAQ):    粗排 15 → 精排 3  (5:1)   ← 候选少代价小
    极致召回(医疗): 粗排 100 → 精排 5 (20:1)  ← 宁可多找也不能漏
""")
```

---

### 3.5 总结：查询阶段的组件选型

```
═══════════════════════════════════════════════════════════════════════
              Query Engine / Chat Engine 选型速查
═══════════════════════════════════════════════════════════════════════

  单轮还是多轮?
    ├─ 单轮 (一问一答) → index.as_query_engine()
    │
    └─ 多轮 (有上下文) → index.as_chat_engine(chat_mode="condense_question")
                          + ChatMemoryBuffer(token_limit=4000)

  检索策略?
    ├─ 纯语义查询 → as_retriever(similarity_top_k=30)  (纯向量)
    ├─ 精确编码多 → + BM25Retriever                         (加关键词)
    ├─ 两者都需要 → HybridRetriever(mode="reciprocal_rerank") (RRF融合)
    └─ 有元数据过滤 → MetadataFilters(...)                  (缩小范围)

  需要 Reranker 吗?
    ├─ 追求精度 → 加 Cross-Encoder (bge-reranker-v2-m3, top_n=3)
    └─ 成本优先 → 不加, 只用 similarity_top_k=3

  合成策略?
    ├─ 1-3个结果 → response_mode="simple_summarize"  (最快)
    ├─ 4-8个结果 → response_mode="compact"            (推荐默认)
    └─ 8+个结果 → response_mode="tree_summarize"      (大量信息综合)

  召回与精排比值?
    ├─ 通用起步 → 30:3 (10:1)
    ├─ 高精度 → 50:5 (10:1)
    └─ 低成本 → 15:3 (5:1)
═══════════════════════════════════════════════════════════════════════
```


---

