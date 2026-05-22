## 第二章：LlamaIndex 检索与生成核心组件 —— 从 Retriever 到 ResponseSynthesizer 的完整拆解

前一章从理论层面讲解了检索的三阶段漏斗。本章从 LlamaIndex 代码实现层面，把每一个组件拆开来看：它们内部做了什么、怎么调用、参数怎么选、什么时候该用哪个。

### 2.1 检索-生成组件全景：四层架构

```
═══════════════════════════════════════════════════════════════════════
              LlamaIndex 检索-生成四层架构
═══════════════════════════════════════════════════════════════════════

  用户 Query (自然语言)
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  封装层: QueryEngine / ChatEngine                                │
  │  ┌───────────────────────────────────────────────────────────┐ │
  │  │  QueryEngine: 处理单轮问答                                 │ │
  │  │    · query(str) → Response                                │ │
  │  │    · 内部串联: Retriever → Postprocessor → Synthesizer    │ │
  │  │                                                           │ │
  │  │  ChatEngine: 处理多轮对话                                   │ │
  │  │    · chat(str) → Response                                 │ │
  │  │    · 额外: 对话历史管理、指代消解、上下文压缩               │ │
  │  └───────────────────────────────────────────────────────────┘ │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │  ① Retriever │───▶│ ② NodePostproc   │───▶│ ③ Response       │
  │   (检索器)    │    │    (后处理器)     │    │    Synthesizer   │
  │              │    │                  │    │    (响应合成器)    │
  │  召回候选     │    │  过滤 + 重排 +   │    │  组织上下文 +     │
  │  Node 列表    │    │  压缩 + 替换     │    │  Prompt + LLM    │
  └──────────────┘    └──────────────────┘    └──────────────────┘
```

**每一层做了什么，数据在它们之间怎么流动：**

```
  [Retriever]                     [NodePostprocessor]              [ResponseSynthesizer]
  
  输入: Query 字符串               输入: List[NodeWithScore]       输入: List[NodeWithScore]
  处理: Query→向量/分词→ANN/倒排   处理: 阈值过滤→Rerank→替换      处理: 拼接上下文→填充Prompt→LLM
  输出: List[NodeWithScore]       输出: List[NodeWithScore]       输出: Response (text + sources)
        (~20-100个)                     (~3-10个，精排后)                 (自然语言答案)
```

---

### 2.2 Retriever 深度拆解 —— 检索器的统一抽象、工厂模式与具体实现

Retriever 是 LlamaIndex 检索体系的核心抽象层。它解决了一个关键设计问题：**你不需要知道底层是用向量、关键词还是知识图谱做检索——调用方式完全一样。**

#### 2.2.1 BaseRetriever —— 统一接口解决的问题

```
  没有统一接口时 (各检索器各自为政):

    向量检索:  vector_store.search(query_vec, k=5)         ← 需要把Query先向量化
    关键词检索: bm25_index.search(query_tokens, k=5)       ← 需要把Query先分词
    图谱检索:  kg_index.query_entities(entities, k=5)      ← 需要先做NER提取实体

    → 每种检索器的调用方式不同，换一个检索器就要重写调用代码

  有了 BaseRetriever 统一接口后:

    vector_retriever.retrieve("请假规定")     ← 内部自己处理向量化
    bm25_retriever.retrieve("请假规定")       ← 内部自己处理分词
    kg_retriever.retrieve("请假规定")         ← 内部自己处理NER

    → 所有检索器的调用方式完全一样，可以任意替换
    → 这就是"面向接口编程"的力量
```

**BaseRetriever 定义的契约：**

```python
# BaseRetriever 的核心方法 (简化):
class BaseRetriever:
    def retrieve(self, str_or_query_bundle: str, **kwargs) -> List[NodeWithScore]:
        """同步检索。输入查询字符串，返回带分数的Node列表。"""
        pass

    async def aretrieve(self, str_or_query_bundle: str, **kwargs) -> List[NodeWithScore]:
        """异步检索。不阻塞事件循环。"""
        pass

    # 所有子类必须实现 _retrieve (内部方法)，retrieve 会自动处理:
    #   1. 将 str 转为 QueryBundle (包含 embedding + query_str)
    #   2. 调用 callback_manager 触发检索事件
    #   3. 调用子类的 _retrieve 方法
    #   4. 将结果包装为 NodeWithScore 列表
```

**NodeWithScore 的结构：**

```python
class NodeWithScore:
    node: BaseNode          # Node 对象 (text + metadata + relationships)
    score: Optional[float]  # 相似度分数 (余弦值 or BM25分 or RRF融合分)
                            # score 的含义取决于检索器类型
                            # 对向量检索: score = 余弦相似度 [0, 1]
                            # 对 BM25:     score = BM25 分数 [0, ∞)
                            # 对 RRF:      score = RRF 融合分 [0, 0.03左右]
```

#### 2.2.2 as_retriever() —— 工厂方法的完整解析

`index.as_retriever()` 是实际开发中最常用的获取 Retriever 的方式。它的本质是一个**工厂方法**——根据 Index 的类型自动选择对应的 Retriever 类，并把 Index 的组件注入进去。

**四步完整流程：**

```
  Step 1: 提取 Index 的内部组件
  ┌─────────────────────────────────────────────────────────────┐
  │  VectorStoreIndex 持有以下组件:                               │
  │    · self._vector_store   → 向量数据库连接                    │
  │    · self._docstore       → Node 文本/元数据存储              │
  │    · self._index_store    → 索引元数据存储                    │
  │    · self._embed_model    → Settings.embed_model (全局)      │
  │    · self._callback_manager → 事件追踪器                     │
  │                                                             │
  │  as_retriever() 把这些组件全部提取出来                          │
  └─────────────────────────────────────────────────────────────┘

  Step 2: 延迟导入对应的 Retriever 类
  ┌─────────────────────────────────────────────────────────────┐
  │  根据 Index 的类型选择:                                       │
  │    VectorStoreIndex      → VectorIndexRetriever              │
  │    SummaryIndex           → SummaryIndexRetriever             │
  │    TreeIndex              → TreeIndexRetriever                │
  │    KeywordTableIndex      → KeywordTableRetriever             │
  │    KnowledgeGraphIndex    → KnowledgeGraphRAGRetriever        │
  │                                                             │
  │  为什么是"延迟导入": 避免循环依赖。只有调用了 as_retriever()     │
  │  才会 import 具体的 Retriever 类。                              │
  └─────────────────────────────────────────────────────────────┘

  Step 3: 依赖注入 —— 把 Index 的组件传给 Retriever 构造器
  ┌─────────────────────────────────────────────────────────────┐
  │  VectorIndexRetriever 构造器需要:                              │
  │    index          ← self (Index 实例本身)                    │
  │    vector_store   ← self._vector_store                      │
  │    embed_model    ← self._embed_model                       │
  │    docstore       ← self._docstore                          │
  │    callback_manager ← self._callback_manager                │
  │                                                             │
  │  Retriever.retrieve() 内部:                                   │
  │    1. 用 embed_model 将 Query 向量化                          │
  │    2. 用 vector_store 在向量库中检索                           │
  │    3. 用 docstore 将检索到的 ID 映射回 Node 对象              │
  │    4. 用 callback_manager 触发事件 (用于监控和追踪)            │
  └─────────────────────────────────────────────────────────────┘

  Step 4: 传递 **kwargs —— 让调用方控制检索行为
  ┌─────────────────────────────────────────────────────────────┐
  │  similarity_top_k=5    → Retriever 每次检索返回几个结果        │
  │  alpha=0.5             → 混合检索权重 (0=纯BM25, 1=纯向量)    │
  │  filters=MetadataFilters(...) → 元数据过滤条件                │
  │  node_ids=[id1, id2]   → 限定只在指定 Node 中检索             │
  └─────────────────────────────────────────────────────────────┘
```

**as_retriever 的三种典型用法：**

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# ═══════════════════════════════════════════════════════════════
# 用法 1: 最简原型 —— 只设 top_k
# ═══════════════════════════════════════════════════════════════
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("事假需要提前多久申请？")
# 适用: 刚跑通流程，还没调参数

# ═══════════════════════════════════════════════════════════════
# 用法 2: 元数据过滤 —— 限定检索范围
# ═══════════════════════════════════════════════════════════════
filters = MetadataFilters(filters=[
    MetadataFilter(key="status", value="active"),
    MetadataFilter(key="version", value="V3.0"),
])
retriever = index.as_retriever(
    similarity_top_k=10,    # 在过滤后的子集中取 Top-10
    filters=filters,
)
# 适用: 只查当前生效的 V3.0 政策，排除旧版本和草稿

# ═══════════════════════════════════════════════════════════════
# 用法 3: 配合 Reranker —— 粗排 + 精排
# ═══════════════════════════════════════════════════════════════
from llama_index.core.postprocessor import SentenceTransformerRerank

# 粗排: 取 30 个候选 (高召回)
retriever = index.as_retriever(similarity_top_k=30)
# 精排: 用 Cross-Encoder 从 30 个中挑出最好的 3 个
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=3,
)

query_engine = index.as_query_engine(
    retriever=retriever,
    node_postprocessors=[reranker],
)
# 适用: 生产环境，追求高精度
```

**as_retriever 适合与不适合的场景：**

| 适合 | 不适合 |
|------|--------|
| 快速原型——一行代码拿到检索器 | 需要 BM25 检索——as_retriever 默认是向量检索 |
| 标准 RAG 查询——默认向量检索足够 | 需要多路检索——需要显式创建 HybridRetriever |
| 元数据过滤——传入 filters 即可 | 需要精细控制检索参数——显式创建更灵活 |
| 配合 Reranker——设大 top_k 做粗排 | 需要自定义检索逻辑——显式创建 |

#### 2.2.3 显式创建检索器 —— 完全自由的控制

```python
from llama_index.core.retrievers import VectorIndexRetriever

# 显式创建，可以传入 as_retriever() 不支持的高级参数
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    # ── 以下参数 as_retriever 也支持但通过 **kwargs 隐式传递 ──
    vector_store_query_mode="hybrid",  # "default"(纯向量) / "sparse" / "hybrid"
                                       # "hybrid" 需要向量库支持混合检索
    alpha=0.7,          # 混合检索权重: 0.7向量 + 0.3关键词
    filters=filters,
    # ── 以下参数 as_retriever 不支持 ──
    node_ids=["node_001", "node_005"],  # 限定只在这两个 Node 中检索
    # 这在你只想"在特定章节范围内检索"时非常有用
)

# 适用场景:
#   1. 需要 vector_store_query_mode="hybrid" (向量库原生混合检索)
#   2. 需要限定特定 node_ids (只查某几个章节)
#   3. 需要精细控制 alpha 参数
```

---

### 2.3 BM25Retriever —— 关键词统计算法的完整解析

#### 2.3.1 BM25 的数学直觉

向量检索找的是"意思相近"的文档，BM25 找的是"包含相同词"的文档。两者互补，因为有些查询（如精确编号"ERP-2025-BJ-001"）没有"意思"可言——它就是一段精确的字符串。

**BM25 公式的直观理解：**

```
  对于查询 Q = "请假 申请 流程" 和 文档 D:

  Score(Q, D) = IDF("请假") × TF_norm("请假", D) + 
                IDF("申请") × TF_norm("申请", D) + 
                IDF("流程") × TF_norm("流程", D)

  三个部分拆解:

  1. IDF(q) —— 这个词"有多稀有"？
     = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
     
     N = 总文档数 = 1000
     "的": 出现在 998 个文档中 → IDF ≈ ln(2.5/998.5 + 1) ≈ 0.002  ← 几乎为0
     "请假": 出现在 15 个文档中 → IDF ≈ ln(985.5/15.5 + 1) ≈ 4.2  ← 很高!
     "竞业限制": 出现在 2 个文档 → IDF ≈ ln(998.5/2.5 + 1) ≈ 6.0  ← 极高!

     → 稀有词获得更高权重，常见词被自动压制

  2. TF_norm(q, D) —— 这个词在文档中出现几次？(长度归一化后)
     = f(q,D) / (f(q,D) + k₁ × (1 - b + b × |D|/avgDL))
     
     其中:
       f(q,D): 词在文档D中出现的次数
       |D|:    文档D的长度
       avgDL:  所有文档的平均长度
       k₁:     词频饱和参数 (通常 1.5)
       b:      长度归一化参数 (通常 0.75)

     文档A (500字，出现"请假" 3次): TF_norm ≈ 3/(3+1.5×0.775) ≈ 0.72
     文档B (5000字，出现"请假" 3次): TF_norm ≈ 3/(3+1.5×1.525) ≈ 0.57
     
     → 同样的词频，短文档比长文档得分更高
     → 这很make sense: 500字文章出现3次"请假"说明文档核心就是请假;
       5000字文章出现3次可能只是顺带提了一下
```

**k₁ 和 b 的调参影响：**

| 参数 | 作用 | 值小 → | 值大 → | 默认值 |
|:---:|------|------|------|:---:|
| k₁ | 词频饱和控制 | TF增长慢(出现几次就饱和) | TF增长快(出现很多次仍加分) | 1.5 |
| b | 长度归一化强度 | 不惩罚长文档 | 强力惩罚长文档 | 0.75 |

#### 2.3.2 BM25Retriever 使用示例

```python
from llama_index.core.retrievers import BM25Retriever
from llama_index.core import VectorStoreIndex

# ── 从 Node 列表构建 BM25 检索器 ─────────────────────────
# 注意: BM25 不需要 Embedding! 它直接从 Node 的纯文本构建倒排索引
# 所以节点数越多，构建时间越长，但检索速度不受影响

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,               # 所有 Node (用于构建倒排索引)
    similarity_top_k=10,       # 每次返回 Top-10
    # BM25 参数:
    k1=1.5,                    # 词频饱和 (默认 1.5)
    b=0.75,                    # 长度归一化 (默认 0.75)
)

# ── 检索 ─────────────────────────────────────────────────
# BM25 对精确编码查询特别有效
results = bm25_retriever.retrieve("ERP-2025-BJ-001 审批状态")
for node in results:
    print(f"BM25={node.score:.2f} | {node.text[:60]}...")

# ── BM25 也可配合元数据过滤 ─────────────────────────────
# 先用元数据缩小范围，再在范围内做 BM25
bm25_retriever = BM25Retriever.from_defaults(
    nodes=[n for n in nodes if n.metadata.get("status") == "active"],
    # ↑ 只对 active 的 Node 建倒排索引
    similarity_top_k=5,
)
```

---

### 2.4 混合检索策略与 RRF 融合机制 —— 深入剖析

#### 2.4.1 为什么 RRF 而不简单加权

将向量检索和 BM25 的结果融合，最朴素的想法是加权求和：0.7 × 向量分 + 0.3 × BM25 分。但这里有两个问题：

```
  问题 1: 两种分数的量纲完全不同
    向量相似度: [0, 1]  (0=无关, 1=几乎相同)
    BM25 分数:  [0, ∞)  (可能几百也可能几千)
    
    如果你不归一化直接加权:
      0.7 × 0.92 + 0.3 × 243.5 = 0.644 + 73.05 = 73.7
      → BM25 完全主导了融合结果，向量检索等于没参与

    归一化后 (min-max):
      0.7 × 0.92 + 0.3 × 0.65 = 0.644 + 0.195 = 0.839
      → 看起来合理了

    但归一化需要先拿到所有分数才能算 min/max，增加了计算开销。

  问题 2: 某种检索器偶尔产出一个异常高分
    如果 BM25 对某个文档打出特别高的分 (因为查询词在文档中出现了很多次)，
    归一化后那个文档的分数会接近 1.0，其他文档都接近 0.0
    → 一种检索器的"异常"主导了整个融合

  RRF 没有这两个问题:
    1. 它只看排名，完全不关心原始分数是多少
    2. 排名天然是均匀分布 (1, 2, 3, ...)，没有异常值
    3. 不需要先计算 min/max 再归一化 —— 直接融合
```

#### 2.4.2 RRF 公式的完整分析

```
  RRF_score(d) = Σ 1 / (k + rank_i(d))

  假设有 2 个检索器，k=60:

  文档A: 检索器1排第1，检索器2排第5
    RRF = 1/(60+1) + 1/(60+5) = 0.01639 + 0.01538 = 0.03177

  文档B: 检索器1排第3，检索器2排第2
    RRF = 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200  ← B > A!

  直觉: A在一路拿了第1，B在每路都是前3。RRF 认为"两路都看好"的B
        比"一路特别看好"的A更可靠。这减小了单路排名波动的风险。

  k 值的影响:
    k=0:   RR 差距大 → 第1名 = 1.0, 第10名 = 0.1 → 极端
    k=60:  RR 差距小 → 第1名 ≈ 0.016, 第10名 ≈ 0.014 → 平滑
    k=∞:   所有人得分一样 → 失去排名信息 → 无意义

    k=60 是经验值，平衡了"保留排名信息"和"给排名靠后的文档机会"。
```

#### 2.4.3 HybridRetriever 三种融合模式完整对比

```python
from llama_index.core.retrievers import (
    VectorIndexRetriever, BM25Retriever, HybridRetriever,
)

# ── 基础检索器 ────────────────────────────────────────────
vec = index.as_retriever(similarity_top_k=20)
bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)

# ═══════════════════════════════════════════════════════════════
# 模式 1: reciprocal_rerank (RRF, 默认)
# ═══════════════════════════════════════════════════════════════
hybrid_rrf = HybridRetriever.from_defaults(
    retrievers=[vec, bm25],
    mode="reciprocal_rerank",
    k=60,                # RRF 平滑常数
    top_k=10,            # 最终返回 Top-10
)
# 
# 工作流: 两路各自检索 → RRF融合 → Top-10
# 适用: 检索器类型不同 (向量 vs BM25)，分数不可比
# 优点: 不依赖分数量纲、对异常值不敏感
# 缺点: 丢弃了原始分数中的"置信度"信息
# 典型场景: 向量 + BM25 混合，最常用的设置

# ═══════════════════════════════════════════════════════════════
# 模式 2: relative_score_fusion (相对分数融合)
# ═══════════════════════════════════════════════════════════════
hybrid_rel = HybridRetriever.from_defaults(
    retrievers=[vec, bm25],
    mode="relative_score_fusion",
    top_k=10,
    weights=[0.7, 0.3],  # 向量 70%，BM25 30%
)
#
# 工作流: 各自归一化 → 加权求和 → Top-10
# 适用: 检索器分数体系相似，可以对分数做公平归一化
# 优点: 保留原始分数的置信度信息，可以给不同检索器不同权重
# 缺点: 需要选择归一化方法，对异常值敏感
# 典型场景: 多个向量检索器 (不同 Embedding) 的融合

# ═══════════════════════════════════════════════════════════════
# 模式 3: simple (简单合并)
# ═══════════════════════════════════════════════════════════════
hybrid_simple = HybridRetriever.from_defaults(
    retrievers=[vec, bm25],
    mode="simple",
    top_k=10,
)
#
# 工作流: 收集所有结果 → 去重 → 按原始分数排序 → Top-10
# 适用: 所有检索器使用相同类型的分数 (如都是余弦相似度)
# 优点: 最简单，开销最小
# 缺点: 不做任何融合优化，不同分数量纲下排序不公平
# 典型场景: 同一索引的不同检索参数 (如: K=20 + K=30 两路)
```

**三种融合模式对比表：**

| 维度 | reciprocal_rerank | relative_score_fusion | simple |
|------|:---:|:---:|:---:|
| 是否保留原始分数 | 否 (只看排名) | 是 (归一化后加权) | 是 (直接用) |
| 是否需要分数归一化 | 否 | 是 | 否 |
| 对异常值敏感度 | 低 | 高 | 高 |
| 是否可以加权 | 否 (排名天然等权重) | 是 (weights参数) | 否 |
| 不同类检索器融合 | ★★★★★ (最适合) | ★★★ (需归一化) | ★ (不推荐) |
| 同类检索器融合 | ★★★ | ★★★★ | ★★★★★ |
| 额外计算开销 | 无 | 需计算min/max | 无 |
| 推荐场景 | 向量+BM25 混合 | 多Embedding融合 | 同索引不同参数 |

---

### 2.5 Reranker 类型详解与选型

#### 2.5.1 三种 Reranker 的完整工作原理

**类型 1: Cross-Encoder Reranker**

```
  输入: Query="请假需要提前多久", Candidate Docs = [Doc1, Doc2, ..., Doc50]

  Step 1: Tokenization (逐个处理或批处理)
    Doc1: [CLS] 请假 需要 提前 多久 [SEP] 事假 需 提前 1 个 工作日 ... [SEP]
    Doc2: [CLS] 请假 需要 提前 多久 [SEP] 年假 天数 按 工龄 计算 ... [SEP]
    ...

  Step 2: Transformer 编码
    12 层 BERT-based 模型，每层 Self-Attention 在 Query 和 Doc 
    的所有 token 之间做完整的交叉注意力。
    
    关键: Query的token可以看到Doc的所有token，Doc的token也可以看到Query的所有token
    → 这就是"Cross-Encoder"的含义 —— Query和Doc不是独立编码的

  Step 3: Pooling + 线性层 → 分数
    [CLS] token的最终隐藏状态 → h_cls (768 or 1024维)
    → sigmoid(W · h_cls + b) → 0.93 (高分 = 高度相关)
    
  Step 4: 按分数降序排列 → 取 Top-N
    Doc1: 0.93  Doc2: 0.12  Doc3: 0.87  ...
    → 精排后: Doc1 > Doc3 > Doc5 > ...
    → 取前 3 个返回
```

**类型 2: LLM-as-Judge Reranker**

```
  工作原理:
    不是用专门的 Cross-Encoder 模型，而是用 GPT-4 这类 LLM 直接做裁判。

    Prompt 模板:
      "下面是一个用户问题和若干个候选文档片段。
       请判断每个片段能否帮助回答用户问题。

       用户问题: {query}
       
       候选片段 1: {doc1_text}
       候选片段 2: {doc2_text}
       ...

       请选择最相关的 3 个片段，并给出你的理由。"

    LLM 回复 → 解析出选择的片段 → 返回

  与 Cross-Encoder 的对比:
    ✅ 可解释性强 (LLM 能说出为什么选这个)
    ✅ 不需要专门的 reranker 模型
    ❌ 极慢 (一次推理 2-5s, 比 Cross-Encoder 慢 10-50 倍)
    ❌ 成本高 (GPT-4 API 费用)
    ❌ 分数不稳定 (同一输入两次可能给出不同排列)
  
  适用场景: 评测/调参阶段，需要理解"为什么这个doc被选中"
```

**类型 3: API-based Reranker**

```
  调用 Cohere Rerank API / Jina Rerank API:
  
    POST https://api.cohere.ai/v1/rerank
    {
      "query": "请假需要提前多久",
      "documents": ["事假需提前1个工作日...", "年假按工龄计算...", ...],
      "top_n": 3
    }
    
    响应:
    {
      "results": [
        {"index": 0, "relevance_score": 0.97},
        {"index": 5, "relevance_score": 0.82},
        ...
      ]
    }
  
  适用场景: 不想部署模型，愿意为便利付费
  代价: 数据发送到第三方，有合规风险
```

#### 2.5.2 Reranker 选型决策

```
  你对"精度"的要求有多高?
    ├─ 极高 (法律/医疗) → Cross-Encoder (bge-reranker-v2-m3)
    │   必须用本地部署的最强模型
    │
    ├─ 高 (企业客服) → Cohere Rerank API 或 bge-reranker-base
    │   精度和成本的平衡
    │
    └─ 一般 (内部工具) → 甚至可以不用 Reranker，直接用向量检索 Top-3

  你有 GPU 吗?
    ├─ 有 → bge-reranker-v2-m3 (本地部署，零 API 成本)
    └─ 没有 → Cohere Rerank API 或 Jina Rerank API

  你需要"可解释性"吗?
    ├─ 是 → LLM-as-Judge (评测阶段，需要理解排名逻辑)
    └─ 否 → Cross-Encoder (开箱即用)
```

---

### 2.6 ResponseSynthesizer —— 从检索结果到最终回答的完整链路

#### 2.6.1 内部四步骤完整解析

```
═══════════════════════════════════════════════════════════════════════
        ResponseSynthesizer.get_response() 内部全过程
═══════════════════════════════════════════════════════════════════════

  输入:
    query_str = "请假需要提前多久申请？"
    nodes = [
      NodeWithScore(node=Node(text="事假需提前1个工作日..."), score=0.93),
      NodeWithScore(node=Node(text="病假应在8:30前通知..."), score=0.81),
      NodeWithScore(node=Node(text="年假按工龄计算..."), score=0.75),
    ]

  Step 1: 上下文组织 —— 把 Node 列表变成一段"参考资料"
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  输入: List[NodeWithScore]  (n=3)                           │
  │                                                             │
  │  处理:                                                      │
  │    1. 按 score 降序排列                                      │
  │    2. 每个 Node 标记序号: [1], [2], [3]                     │
  │    3. 可选: 在每个 Node 前补充 metadata 信息                  │
  │       如: [1] 来源: 考勤管理制度 (V3.0, 第2页)               │
  │    4. Node 之间用分隔符连接: "\n\n---\n\n"                  │
  │    5. 控制总长度 (如果超过 max_tokens→截断最后面的 Node)     │
  │                                                             │
  │  输出: context_str                                          │
  │    "[1] 事假需提前1个工作日向部门主管申请，经审批后交HR备案。 │
  │     事假期间不发放工资。\n\n---\n\n                          │
  │     [2] 病假应在当日8:30前通知部门主管。连续超过2天须提供    │
  │     二级及以上医院证明。\n\n---\n\n                           │
  │     [3] 年假天数按工龄计算：入职1-5年5天，5-10年10天，       │
  │     10年以上15天。年假最小单位为半天。"                        │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  Step 2: 提示工程与填充 —— 构造完整的 LLM Prompt
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  LlamaIndex 默认的 Prompt 模板 (可自定义):                     │
  │                                                             │
  │  System:                                                     │
  │    "你是一个智能助手。请根据以下参考资料回答用户问题。         │
  │     如果参考资料中没有相关信息，请明确说明。                   │
  │     回答时请标注引用的资料编号，如 [1]。"                      │
  │                                                             │
  │  Context:                                                    │
  │    {context_str}          ← Step 1 产出的上下文              │
  │                                                             │
  │  Query:                                                      │
  │    {query_str}            ← 用户的原始问题                   │
  │                                                             │
  │  填充 → 完整的 Prompt 字符串                                  │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  Step 3: LLM 推理生成
  ┌─────────────────────────────────────────────────────────────┐
  │  · 将完整 Prompt 发送给 Settings.llm                         │
  │  · 参数: temperature=0.3 (忠实于资料), max_tokens=1024       │
  │  · 如果是 streaming=True → 流式返回 token                     │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
  Step 4: 后处理 —— 构建 Response 对象
  ┌─────────────────────────────────────────────────────────────┐
  │  Response:                                                   │
  │    .response        → "根据规定，事假需提前1个工作日[1]..."    │
  │    .source_nodes    → [NodeWithScore, ...] (原始检索结果)    │
  │    .metadata        → {synthesizer, query, ...}              │
  └─────────────────────────────────────────────────────────────┘

  输出: Response 对象
═══════════════════════════════════════════════════════════════════════
```

#### 2.6.2 五种合成策略完整详解

**① Simple Summarize —— 一次性拼接，最快**

```
  工作原理:
    所有 Node 文本 → 拼成一段 → 填入 Prompt → 一次性发给 LLM

  内部代码等价于:
    context = "\n\n---\n\n".join([n.text for n in nodes])
    prompt = f"参考资料:\n{context}\n\n用户问题: {query}"
    answer = llm(prompt)

  优点: 只调用一次 LLM，最快最省钱
  缺点: 如果所有 Node 拼接后超过 LLM 上下文窗口 → 后面的内容被截断
  适用: 1-3 个 Node，总文本不长
  不适用: 5+ 个长 Node (可能超过 8K/32K tokens)

  典型耗时: 1-2 秒 (1 次 LLM 调用)
  典型成本: 1 次 LLM API 调用
```

**② Refine —— 逐条迭代精炼，质量最高**

```
  工作原理:
    Step 1: Node[0] → Prompt → LLM → answer_0 (基于第 1 条资料的回答)
    Step 2: answer_0 + Node[1] → Prompt → LLM → answer_1 (融合第 2 条资料)
    Step 3: answer_1 + Node[2] → Prompt → LLM → answer_2 (融合第 3 条资料)
    ...
    最终返回 answer_n

  Prompt 结构 (第二步及之后):
    "我们已经有了一个部分回答: {existing_answer}
     
     下面是一条新的参考信息。请更新上面的回答来包含这条新信息。
     如果新信息与已有回答有矛盾，请指出矛盾。
     
     新信息: {next_node_text}
     
     用户问题: {query}"

  优点: 
    · 不会因为内容太多超过窗口 (每次只处理 1 个 Node)
    · 每条信息都被 LLM 充分"咀嚼"过
    · 如果前后信息有矛盾，LLM 会注意到
  缺点: 
    · N 次 LLM 调用 (N = Node 数量) → 最慢、最贵
    · 如果前几步理解有误，后续纠正困难
  适用: 
    · 需要极致质量 (法律文件、医疗报告)
    · 信息之间可能存在矛盾 (多个来源)
  不适用: 
    · 对延迟敏感的场景
    · 成本敏感的场景

  典型耗时: N × 1-2s (N 次 LLM 调用)
  典型成本: N 次 LLM API 调用
```

**③ Compact —— 先压缩再生成，平衡之选**

```
  工作原理:
    Step 1: 尝试把所有 Node 拼接 → 检查 token 数
    Step 2: 如果 ≤ max_tokens → 直接用 Simple Summarize 模式 (一次调用)
    Step 3: 如果 > max_tokens → 对 Node 文本做"压缩":
      把每个长 Node 拆分为更小的片段
      对每个片段做摘要 (用 LLM 生成短摘要)
      用摘要替代原始文本
      重复检查 token 数 → 直到合适
    Step 4: 将压缩后的上下文 + Query → 一次 LLM 调用

  优点:
    · 自适应 —— 内容少时和 Simple 一样快，内容多时自动压缩
    · 不会超出上下文窗口
    · 成本可控 (最多 2 次 LLM 调用: 压缩 + 生成)
  缺点:
    · 压缩阶段可能丢失细节
    · 不如 Refine 精确 (细节被摘要替代了)
  适用: 
    · 检索结果多 (5-10 个 Node) 且总文本长
    · 生产环境的默认推荐
  不适用:
    · 需要精确还原原文细节 (法律条款)

  典型耗时: 2-4 秒 (1-2 次 LLM 调用)
  典型成本: 1-2 次 LLM API 调用
```

**④ Tree Summarize —— 树形归纳，处理大量 Node**

```
  工作原理:
    
    Node[0]  Node[1]  Node[2]  Node[3]  Node[4]  Node[5]  Node[6]  Node[7]
       │        │        │        │        │        │        │        │
       └───┬────┘        └───┬────┘        └───┬────┘        └───┬────┘
           │                 │                 │                 │
        LLM调用1          LLM调用2          LLM调用3          LLM调用4
           │                 │                 │                 │
       摘要 A             摘要 B             摘要 C             摘要 D
           │                 │                 │                 │
           └───────┬─────────┘                 └───────┬─────────┘
                   │                                   │
                LLM调用5                             LLM调用6
                   │                                   │
               摘要 AB                              摘要 CD
                   │                                   │
                   └──────────────┬────────────────────┘
                                  │
                               LLM调用7
                                  │
                              最终回答

  一共 7 次 LLM 调用 (对于 8 个 Node)
  
  优点:
    · 能处理任意数量的 Node
    · 层级推理 —— 先局部总结再全局综合
    · 适合"总结整份文档"类的查询
  缺点:
    · 多次 LLM 调用 → 慢、贵
    · 每一层都可能丢失信息
    · 对简单的、答案单一的事实查询过度设计
  适用:
    · 用户问"给我总结一下"、"主要内容是什么"等需要综合大量信息的查询
    · 检索到 8+ 个 Node 时
  不适用:
    · 简单事实查询 ("加班费多少")
    · 答案只在一个 Node 中的查询

  典型耗时: log₂(N) × 1-2s (分层调用)
  典型成本: N-1 次 LLM API 调用
```

**⑤ No Text —— 不使用检索结果，纯 LLM**

```
  工作原理: LLM 直接回答，不给任何上下文

  用途:
    · 测试基线: RAG 到底有没有提升? (Compare RAG vs NoText)
    · 简单问候: "你好" "谢谢" 等不需要检索的对话
    · 对比实验: 检验检索到的内容是否真的被用上了
```

**五种策略汇总对比表：**

| 策略 | LLM调用次数 | 延迟 | 质量 | 适用Node数 | 适用场景 |
|------|:---:|:---:|:---:|:---:|----------|
| Simple | 1 | ~1s | 中 | 1-3 | 快速原型、简单查询 |
| Compact | 1-2 | ~2s | 中高 | 4-8 | **生产环境默认推荐** |
| Tree Summarize | N-1 | ~N×1s | 中 | 8+ | 总结类查询、大量结果 |
| Refine | N | ~N×2s | 极高 | 2-5 | 法律/医疗、不能丢信息 |
| No Text | 1 | ~0.5s | — | 0 | 测试基线、简单问候 |

#### 2.6.3 流式响应与答案溯源

```python
# ── 流式输出 ─────────────────────────────────────────────
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("公司有哪些福利？")
# 用户看到回答逐字出现
for token in streaming_response.response_gen:
    print(token, end="", flush=True)

# ── 答案溯源 ─────────────────────────────────────────────
response = query_engine.query("五险一金怎么交的？")
print(f"回答: {response}\n")
for i, src in enumerate(response.source_nodes, 1):
    print(f"[来源{i}] {src.metadata.get('title')}")
    print(f"  版本: {src.metadata.get('version')}")
    print(f"  页码: {src.metadata.get('page_number', '?')}")
    print(f"  相似度: {src.score:.3f}")
    print(f"  内容: {src.text[:100]}...\n")
```

---

### 2.7 组件选型速查

```
═══════════════════════════════════════════════════════════════════════
            完全选型指南
═══════════════════════════════════════════════════════════════════════

  Q1: 检索器选什么?
    ├─ 快速原型 → as_retriever(similarity_top_k=5)
    ├─ 精确编码查询多 → BM25Retriever
    ├─ 语义查询多 → VectorIndexRetriever
    └─ 两者都多 → HybridRetriever(retrievers=[vec, bm25], mode="reciprocal_rerank")

  Q2: 需要 Reranker 吗?
    ├─ 高Hit+低MRR → 加 Reranker (效果最明显)
    ├─ 高Hit+高MRR → 不需要 Reranker
    └─ 低Hit → 先检查 Recall 问题 (Embedding/切分), 别怪 Reranker

  Q3: 合成策略选什么?
    ├─ 1-3个Node → Simple (最快)
    ├─ 4-8个Node → Compact (推荐默认)
    ├─ 8+个Node (总结) → Tree Summarize
    └─ 极致质量 → Refine (代价: N次LLM调用)
═══════════════════════════════════════════════════════════════════════
```


---

