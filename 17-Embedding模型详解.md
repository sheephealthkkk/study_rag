## 第二章：Embedding 模型 —— 检索质量的"天花板"

第一章讲了向量数据库的选型。但数据库只是一个"容器"——真正决定检索质量上限的，是**往容器里装什么质量的向量**。而向量的质量，由 Embedding 模型决定。

### 2.1 Embedding 模型的重要性：它直接决定检索质量的上限

```
═══════════════════════════════════════════════════════════════════════
        Embedding 模型是 RAG 检索质量的"天花板"
═══════════════════════════════════════════════════════════════════════

  后续所有优化（Reranker 精排、查询改写、HyDE）都是在 Embedding 的
  基础上做"修正"——但如果 Embedding 本身就没把语义编码好（同义词在
  向量空间中距离很远），再多的后处理也弥补不了。

  一个类比:
    Embedding 模型 = 地基
    Reranker/QueryRewrite = 装修

    地基歪了 → 装修再好 → 房子还是会塌
    Embedding 不准 → 后处理再强 → 相关信息排不进 Top-K

  具体影响:
    1. 检索精度上限: Embedding 的语义编码能力几乎是 RAG 检索的"理论最优值"
       → 如果两个语义相同的文本在向量空间中距离很远,
          没有任何检索算法能把它们"拉近"

    2. 系统扩展性: Embedding 模型决定了索引的维度
       → 维度 × 向量数 = 存储成本
       → 维度 × 向量数 = 检索时的计算开销
       → MRL (Matryoshka) 技术支持动态截断维度, 灵活控制扩展成本

    3. 跨语言能力: 如果知识库有中英文混排, Embedding 模型必须支持多语言
       → 否则中文查询永远找不到英文文档, 反之亦然
═══════════════════════════════════════════════════════════════════════
```

### 2.2 铁律：索引和查询必须使用同一个 Embedding 模型

```
═══════════════════════════════════════════════════════════════════════
          为什么必须用同一个 Embedding 模型
═══════════════════════════════════════════════════════════════════════

  索引阶段:
    文档 Chunk → Embedding 模型 A → 向量 V_doc_A → 存入向量库
    文档 Chunk → Embedding 模型 B → 向量 V_doc_B → 存入向量库

  查询阶段:
    用户 Query → Embedding 模型 A → 向量 V_query_A
    用户 Query → Embedding 模型 B → 向量 V_query_B

  如果索引用模型 A, 查询用模型 B:
    V_query_B 和 V_doc_A 不在同一个向量空间!
    → 两者的余弦相似度是随机值 (约等于 0)
    → 检索结果完全随机, 等于没检索

  类比: 用中文发音去匹配英文发音的相似度 —— 坐标系不同, 比较毫无意义。

  生产环境的约束:
    · 切换 Embedding 模型 = 全量重新入库 (re-index)
    · 全量 re-index 的成本 = 文档量 × API 调用成本 + 索引重建时间
    · 百万级文档的 re-index 可能耗时数小时到数天

  → 所以: Embedding 模型一选定终身, 选之前必须充分测试。
═══════════════════════════════════════════════════════════════════════
```

---

### 2.3 主流 Embedding 模型全方位对比

#### 2.3.1 云端 API vs 本地开源：两种部署模式

```
═══════════════════════════════════════════════════════════════════════
              云端 API  vs  本地开源 —— 全景对比
═══════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────┐
  │  云端 API (OpenAI / Cohere / VoyageAI / Jina AI)                 │
  │                                                                  │
  │  配置要点:                                                        │
  │    · 只需 API Key, 不需要 GPU 服务器                              │
  │    · 一行代码调用: client.embeddings.create(input=texts)          │
  │    · 自动扩容, 不需要关心并发                                     │
  │                                                                  │
  │  优势:                                                            │
  │    ✅ 零运维 —— 不需要管理 GPU 服务器/模型更新                     │
  │    ✅ 多语言能力强 —— OpenAI/Cohere 在 100+ 语言上训练             │
  │    ✅ 持续更新 —— 模型升级透明, 无需手动部署新版本                  │
  │    ✅ 弹性伸缩 —— 从 1 条到 100万条, API 自动处理并发              │
  │    ✅ MRL 支持 —— text-embedding-3 系列支持动态截断维度             │
  │                                                                  │
  │  注意事项:                                                        │
  │    ⚠ 成本随规模增长 —— 100万条 × 512 token × $0.02/M = ~$10      │
  │       看似便宜, 但每日增量 + 查询量累加后可达数千美元/月             │
  │    ⚠ 网络延迟 —— 每次调用有 50-200ms 的网络往返                   │
  │    ⚠ 数据合规 —— 文本发送到第三方服务器 (GDPR/数据安全)            │
  │    ⚠ API 限流 —— 免费 tier 通常 3-60 RPM, 大规模索引需付费 tier   │
  ├──────────────────────────────────────────────────────────────────┤
  │  本地开源 (bge / m3e / gte / jina-embeddings / stella)            │
  │                                                                  │
  │  配置要点:                                                        │
  │    · 需要 GPU 服务器 (推荐 A10/A100/T4)                           │
  │    · 模型加载到 GPU 显存中                                         │
  │    · 部署方式: HuggingFace Transformers / sentence-transformers   │
  │              / vLLM / TEI (Text Embeddings Inference)             │
  │                                                                  │
  │  优势:                                                            │
  │    ✅ 数据安全 —— 文本不出内网                                     │
  │    ✅ 长期 TCO 低 —— 固定 GPU 成本, 不按 token 计费                │
  │    ✅ 完全可控 —— 可微调 / 可量化 / 可剪枝 / 可定制                │
  │    ✅ 零延迟网络开销 —— 本地推理 < 10ms vs API 50-200ms           │
  │    ✅ 无 API 限流 —— 吞吐量只受 GPU 算力限制                       │
  │                                                                  │
  │  注意事项:                                                         │
  │    ⚠ 需要工程投入 —— GPU 采购/租赁, 模型部署, 监控告警              │
  │    ⚠ 需要运维能力 —— GPU 驱动/CUDA/显存管理/服务高可用              │
  │    ⚠ 模型更新需手动操作 —— 新模型发布 → 下载 → 替换 → re-index     │
  │    ⚠ 多语言能力取决于具体模型 —— bge-large-zh 对英文支持有限        │
  │    ⚠ 初期成本高 —— GPU 服务器租金 $300-3000/月                     │
  └──────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 具体模型对比

| 模型 | 维度 | 最大输入 | 中文 | 英文 | 多语言 | 成本 | 部署 | 亮点 |
|------|------|----------|:---:|:---:|:---:|------|:---:|------|
| **text-embedding-3-small** | 512-1536* | 8192t | ★★★ | ★★★★★ | ★★★★★ | $0.02/Mt | API | MRL节省存储，性价比首选 |
| **text-embedding-3-large** | 1024-3072* | 8192t | ★★★★ | ★★★★★ | ★★★★★ | $0.13/Mt | API | 精度最高API，MRL |
| **Cohere embed-v3** | 1024 | 512t | ★★★ | ★★★★★ | ★★★★★ | $0.10/Mt | API | 压缩输出，分类优化 |
| **bge-large-zh-v1.5** | 1024 | 512t | ★★★★★ | ★★★ | ★★ | 免费 | 本地 | 中文检索首选开源 |
| **bge-m3** | 1024 | 8192t | ★★★★★ | ★★★★ | ★★★★★ | 免费 | 本地 | 多语言+稀疏+稠密混合 |
| **m3e-base** | 768 | 512t | ★★★★ | ★★ | ★ | 免费 | 本地 | 轻量级，CPU也可用 |
| **m3e-large** | 1024 | 512t | ★★★★½ | ★★ | ★ | 免费 | 本地 | m3e的大尺寸版 |
| **gte-large-zh** | 1024 | 512t | ★★★★½ | ★★★ | ★★ | 免费 | 本地 | 阿里达摩院出品 |
| **jina-embeddings-v3** | 1024 | 8192t | ★★★★ | ★★★★ | ★★★★★ | 免费 | 本地 | 长文本(8K)，任务特定嵌入 |
| **stella-base-zh-v3-1792d** | 1792 | 512t | ★★★★½ | ★ | ★ | 免费 | 本地 | 高维中文专用 |

> \* MRL 支持 = 可动态截断到更小维度而不严重损失精度。512维是默认值，1536维是最大。

#### 2.3.3 LlamaIndex 一键全局设置

LlamaIndex 提供了全局 `Settings` 对象，可以一次性设置 Embedding 模型，后续所有索引和查询自动使用同样的模型——从根本上避免了"索引用一个模型、查询用另一个模型"的错误：

```python
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── 方式 1: 用 OpenAI 云端 API ───────────────────────────
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    dimensions=512,      # MRL 截断到 512 维, 节省 67% 存储
    api_key="sk-...",    # 或从环境变量读取
)

# ── 方式 2: 用 HuggingFace 本地模型 ─────────────────────
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5",
    device="cuda",       # GPU 推理
    max_length=512,      # bge 的最大输入
)

# ── 设置后, 所有索引创建和查询自动使用这个模型 ──────────
from llama_index.core import VectorStoreIndex

# 建索引 → 自动用 Settings.embed_model
index = VectorStoreIndex.from_documents(documents)

# 查询 → 也自动用 Settings.embed_model (同一个!)
query_engine = index.as_query_engine()
response = query_engine.query("事假提前多久？")
# ↑ 索引和查询用同一个模型, 保证了向量空间的一致性
```

---

### 2.4 成本计算与优化策略

#### 2.4.1 成本公式

```
═══════════════════════════════════════════════════════════════════════
              Embedding 成本计算公式
═══════════════════════════════════════════════════════════════════════

  总成本 = 索引成本 + 查询成本

  索引成本 (一次性):
    总 Chunk 数 × 平均每个 Chunk 的 token 数 × 单价

    示例: 10,000 个 Chunk, 平均 350 token/Chunk, text-embedding-3-small($0.02/Mt)
      → 10,000 × 350 = 3,500,000 token
      → 3,500,000 / 1,000,000 × $0.02 = $0.07
      → 一次性成本极低!

  查询成本 (持续):
    日均查询次数 × 平均每个 Query 的 token 数 × 单价 × 30 天

    示例: 日均 10,000 次查询, 平均 15 token/Query, $0.02/Mt
      → 10,000 × 15 = 150,000 token/天
      → 150,000 × 30 / 1,000,000 × $0.02 = $0.09/月
      → 查询成本也极低!

  结论: 对 text-embedding-3-small 来说, Embedding 成本通常不是瓶颈。
        真正的成本来自 LLM 生成 (GPT-4o 生成 1K token ≈ $5-15/Mt vs embed $0.02/Mt)

═══════════════════════════════════════════════════════════════════════
```

#### 2.4.2 基于成本的策略选择

| 策略 | 如何降低成本 | 代价 |
|------|------------|------|
| **减小维度 (MRL)** | 1536→512维, 存储和检索成本降低 67% | 检索精度损失 < 3% |
| **减小 top-K** | top_k=3 vs top_k=10, LLM 输入 token 减少 70% | 可能漏掉相关信息 |
| **缓存热门查询** | 相同 Query → 返回缓存结果, 不调 Embedding API | 需要缓存系统, 缓存失效策略 |
| **本地部署** | 固定 GPU 成本, 不限调用量 | 需要 GPU 服务器和运维 |

---

### 2.5 本地部署详解：以 HuggingFace 为主

#### 2.5.1 部署方式概览

| 方式 | 适用场景 | 复杂度 | 吞吐量 |
|------|----------|:---:|:---:|
| **sentence-transformers** | 单机小规模, Python 原生 | 极低 | 低(~50条/s) |
| **HuggingFace Transformers** | 需要自定义 Pooling 策略 | 低 | 低(~30条/s) |
| **TEI (Text Embeddings Inference)** | 生产级, Rust+GPU 优化 | 中 | 高(~2000条/s) |
| **vLLM** | 如果模型是 LLM-based(如 gte-qwen2) | 中 | 高 |
| **Infinity (BAAI)** | bge 系列官方推荐 | 低 | 高 |

**最常用的三种部署代码：**

```python
# ═══════════════════════════════════════════════════════════════
# 方式 1: sentence-transformers (最简单, 原型首选)
# ═══════════════════════════════════════════════════════════════
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device="cuda")
# 注意: bge 模型的 query 需要加前缀 "为这个句子生成表示以用于检索相关文章："
embeddings = model.encode(
    ["为这个句子生成表示以用于检索相关文章：事假需提前多久申请？"],
    normalize_embeddings=True,   # 归一化后内积=余弦相似度
    batch_size=32,               # 批处理
)
print(f"向量维度: {embeddings.shape}")  # (1, 1024)


# ═══════════════════════════════════════════════════════════════
# 方式 2: LlamaIndex 的 HuggingFaceEmbedding (集成最方便)
# ═══════════════════════════════════════════════════════════════
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5",
    device="cuda",
    max_length=512,
    normalize=True,
    # bge 模型的特殊配置
    query_instruction="为这个句子生成表示以用于检索相关文章：",
    embed_batch_size=32,
)
# 设置全局后，所有索引自动使用
Settings.embed_model = embed_model


# ═══════════════════════════════════════════════════════════════
# 方式 3: TEI (生产环境高吞吐)
# ═══════════════════════════════════════════════════════════════
# 启动: docker run -p 8080:80 --gpus all \
#         ghcr.io/huggingface/text-embeddings-inference:latest \
#         --model-id BAAI/bge-large-zh-v1.5

# Python 调用:
from openai import OpenAI
tei_client = OpenAI(base_url="http://localhost:8080", api_key="not-needed")
response = tei_client.embeddings.create(
    model="bge-large-zh-v1.5",
    input=["事假需提前1个工作日申请"],
)
```

---

### 2.6 评估指标详解：Hit Rate@K 与 MRR@K

#### 2.6.1 指标的定义与计算

```
═══════════════════════════════════════════════════════════════════════
           Hit Rate@K 与 MRR@K —— 公式与含义
═══════════════════════════════════════════════════════════════════════

  前提条件:
    有 N 条测试查询 (如 20 条)。
    每条查询有一个或多个"正确答案"的 Chunk ID (人工标注)。
    检索系统对每条查询返回 Top-K 个 Chunk。

  ═══════════════════════════════════════════════════════════

  Hit Rate@K (命中率@K)
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  Hit Rate@K = 至少命中 1 个正确答案的查询数 / 总查询数    │
  │                                                         │
  │  计算方式:                                               │
  │    for each query:                                       │
  │      检索返回 Top-K 个 Chunk                              │
  │      if 这些 Chunk 中至少有一个是"正确答案":               │
  │        hits += 1                                        │
  │    Hit Rate@K = hits / total_queries                    │
  │                                                         │
  │  含义: "在 Top-K 个结果中, 有多少比例的查询找到了        │
  │         至少一个相关答案?"                                │
  │                                                         │
  │  例子: 20 条查询, 其中 18 条的 Top-5 结果中包含了正确答案 │
  │        → Hit Rate@5 = 18/20 = 90%                       │
  │                                                         │
  │  Hit Rate 高 → 大部分查询能找到相关结果 (查全好)          │
  │  Hit Rate 低 → 很多查询找不到相关结果 (漏检多)            │
  └─────────────────────────────────────────────────────────┘

  MRR@K (Mean Reciprocal Rank @K)
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  MRR@K = (1/N) × Σ (1 / rank_of_first_correct)          │
  │                                                         │
  │  计算方式:                                               │
  │    for each query:                                       │
  │      找到第一个正确答案在 Top-K 结果中的排名 rank_i       │
  │      如果 K 个结果中都没有正确答案 → 该查询的 RR = 0      │
  │    MRR@K = (Σ 1/rank_i) / N                             │
  │                                                         │
  │  含义: "正确答案在检索结果中排第几位? 排名越靠前, MRR越高" │
  │                                                         │
  │  例子: 3 条查询                                          │
  │    Query 1: 正确答案排第 1  → RR = 1/1 = 1.000          │
  │    Query 2: 正确答案排第 3  → RR = 1/3 = 0.333          │
  │    Query 3: 正确答案排第 2  → RR = 1/2 = 0.500          │
  │    MRR = (1.0 + 0.333 + 0.5) / 3 = 0.611               │
  │                                                         │
  │  MRR 高 → 正确答案排名普遍靠前 (排序质量好)               │
  │  MRR 低 → 正确答案排名靠后, 即使命中了也排在后面          │
  └─────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
```

#### 2.6.2 K 值的选择与含义

```
═══════════════════════════════════════════════════════════════════════
              不同 K 值的含义
═══════════════════════════════════════════════════════════════════════

  K=1:
    最严格 —— 只有排在第 1 位的才计入命中
    Hit Rate@1 = 50% → 一半查询的最佳结果是正确答案
    意义: 衡量的是"检索系统能不能一次给对"
    典型值: 简单场景 60-80%, 复杂场景 30-50%

  K=5:
    最常用 —— Top-5 是 LLM 生成时最常使用的上下文数量
    Hit Rate@5 = 90% → 90% 的查询在 Top-5 中至少有一个正确答案
    意义: 衡量的是"检索到的信息是否足以支撑 LLM 生成"
    典型值: 80-95%

  K=10:
    宽松 —— Top-10 是 Rerank 前粗排的典型值
    Hit Rate@10 = 97% → 几乎不会漏掉正确答案
    意义: 衡量的是"粗排的召回能力"
    典型值: 90-99%

  为什么看多个 K:
    Hit Rate@1 vs Hit Rate@10 的差距告诉我们:

    · 差距小 (90% vs 92%): 正确答案通常排得很靠前
      → 说明 Embedding 质量高, 检索排序准确

    · 差距大 (40% vs 95%): 正确答案存在, 但排名不靠前
      → 说明 Embedding 能区分"相关vs不相关", 但不擅长精细排序
      → 需要加 Reranker 做精排

  ───────────────────────────────────────────────────────────

  1/2, 1/5, 1/10 的区别 (这是 MRR 的分母部分):

    如果正确答案排在第 1 位 → MRR 贡献 = 1/1 = 1.000
    如果正确答案排在第 2 位 → MRR 贡献 = 1/2 = 0.500
    如果正确答案排在第 5 位 → MRR 贡献 = 1/5 = 0.200
    如果正确答案排在第 10 位 → MRR 贡献 = 1/10 = 0.100

    注意: 1/2 = 0.5 是 1/1 = 1.0 的一半! 排第 2 的"价值"只有排第 1 的一半。
          排第 10 的"价值"只有排第 1 的十分之一。

    这说明 MRR 非常强调"第一个正确答案的排名"——排名越靠后, 贡献急剧下降。
    MRR 对 Reranker 的效果非常敏感 —— Reranker 把排名从 5→1, MRR 从 0.2 跳到 1.0。
═══════════════════════════════════════════════════════════════════════
```

#### 2.6.3 诊断模式：Hit Rate 与 MRR 的四种组合

```
═══════════════════════════════════════════════════════════════════════
          Hit Rate vs MRR 诊断矩阵
═══════════════════════════════════════════════════════════════════════

  高 Hit Rate + 高 MRR (理想状态)
  ┌─────────────────────────────────────────────────────────┐
  │ Hit Rate@5=95%, MRR=0.85                                │
  │                                                         │
  │ 含义: 大部分查询能找到正确答案, 且排名靠前                  │
  │ 评估: ★★★★★ 完美! Embedding 模型和切分策略都很好            │
  │ 操作: 保持现状, 可以考虑减小 top_k 以降低成本               │
  └─────────────────────────────────────────────────────────┘

  高 Hit Rate + 低 MRR (常见问题)
  ┌─────────────────────────────────────────────────────────┐
  │ Hit Rate@5=90%, MRR=0.30                                │
  │                                                         │
  │ 含义: 正确答案在 Top-5 里, 但通常排在 3-5 位              │
  │        → "找到了, 但没放在最前面"                         │
  │                                                         │
  │ 原因分析:                                                │
  │   · Embedding 模型能区分相关/不相关, 但精细排序能力弱     │
  │   · 多个 Chunk 语义相近, 正确答案被其他类似的 Chunk      │
  │     "挤"到后面                                          │
  │   · 查询太短/模糊, 多个 Chunk 得分接近, 排序不稳定        │
  │                                                         │
  │ 调优方案:                                                │
  │   1. 加 Reranker (Cross-Encoder 精排) —— 最有效的方案    │
  │   2. 增大 top_k, 让更多候选进入精排阶段                   │
  │   3. 查询改写 (Query Rewrite/Expansion), 让 query 更精确 │
  └─────────────────────────────────────────────────────────┘

  低 Hit Rate + 高 MRR (需警惕)
  ┌─────────────────────────────────────────────────────────┐
  │ Hit Rate@5=50%, MRR=0.70                                │
  │                                                         │
  │ 含义: 排名前几的结果质量高, 但只有一半查询能找到正确答案    │
  │        → "找到的都排对了, 但很多查询根本找不到"           │
  │                                                         │
  │ 原因分析:                                                │
  │   · 某些类别的查询没有对应的 Chunk (知识库覆盖不全)       │
  │   · chunk_size 太小, 部分关键词信息被分散到多个 Chunk     │
  │   · Embedding 模型对某类领域术语的理解差                  │
  │                                                         │
  │ 调优方案:                                                │
  │   1. 检查未命中的查询 → 补充知识库文档                    │
  │   2. 增大 chunk_size, 让每个 Chunk 包含更完整的信息       │
  │   3. 用查询分解 (Sub-question) 拆分长查询为多个短查询     │
  │   4. 混合检索 (向量 + BM25) 补充纯向量的盲区              │
  └─────────────────────────────────────────────────────────┘

  低 Hit Rate + 低 MRR (需要根本性调整)
  ┌─────────────────────────────────────────────────────────┐
  │ Hit Rate@5=30%, MRR=0.15                                │
  │                                                         │
  │ 含义: 大部分查询找不到正确答案, 找到了也排在很后面         │
  │                                                         │
  │ 原因分析:                                                │
  │   · Embedding 模型完全不合适 (如用英文模型做中文检索)     │
  │   · 切分策略有严重问题 (如 chunk 太碎或太大)             │
  │   · 知识库内容与用户查询完全不匹配                        │
  │                                                         │
  │ 调优方案:                                                │
  │   1. 换 Embedding 模型 (中文→bge, 英文→text-embedding-3) │
  │   2. 重新设计切分策略 (改 chunk_size, 换切分器)           │
  │   3. 重新审核知识库质量和覆盖范围                          │
  └─────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
```

#### 2.6.4 后续关联指标：Faithfulness / Answer Relevance / Context Relevance

这些指标在第二十二章已经讲过，这里补充它们与 Embedding 模型的关系：

| 指标 | 含义 | 与 Embedding 的关系 |
|------|------|-------------------|
| **Faithfulness (忠实度)** | LLM 回答是否基于检索到的 Chunk？ | Embedding 检索不到正确 Chunk → LLM 只能编造 → 忠实度低 |
| **Answer Relevance (回答相关性)** | 回答是否直接回应用户问题？ | 检索到的 Chunk 不相关 → LLM 被噪声引导 → 回答跑题 |
| **Context Relevance (上下文相关性)** | 检索到的 Chunk 是否与问题相关？ | **直接衡量 Embedding 检索质量** — 与 Precisio@K 高度相关 |

---

### 2.7 评估实战：代码演示

```python
"""
═══════════════════════════════════════════════════════════════════════════
  Embedding 模型评估 —— Hit Rate@K 与 MRR@K 完整实战
═══════════════════════════════════════════════════════════════════════════

  评估流程:
    1. 准备测试数据集: 每条 = (问题, 正确答案 Chunk ID 列表)
    2. 用待评估的 Embedding 模型对所有 Chunk 向量化
    3. 对每条测试查询, 用同样的 Embedding 模型向量化
    4. 计算余弦相似度 → 排序 → 取 Top-K
    5. 检查正确答案是否在 Top-K 中 → 计算 Hit Rate
    6. 检查正确答案的排名 → 计算 MRR

  依赖: pip install numpy openai scikit-learn
"""
import numpy as np
from typing import List, Dict, Tuple
import time


# ═══════════════════════════════════════════════════════════════
# Step 1: 准备测试数据
# ═══════════════════════════════════════════════════════════════

# 模拟知识库: 10 个 Chunk
# 实际场景中, 这些是你切分好的文档片段
knowledge_chunks = [
    # ID:0  "事假需提前1个工作日向部门主管申请..."
    "事假需提前1个工作日向部门主管申请，经审批后交HR备案。事假期间不发放工资。",
    # ID:1  "病假应在当日8:30前通知部门主管..."
    "病假应在当日8:30前通知部门主管。连续病假超过2天须提供二级及以上医院证明。",
    # ID:2  "年假天数按工龄计算：1-5年5天，5-10年10天..."
    "年假天数按工龄计算：入职1-5年员工享有5天年假，5-10年10天，10年以上15天。",
    # ID:3  "公司为所有员工缴纳五险一金..."
    "公司依法为所有员工缴纳五险一金：养老保险、医疗保险、失业保险、工伤保险、生育保险和住房公积金。",
    # ID:4  "工作日加班按1.5倍工资计算，休息日加班按2倍..."
    "加班费计算标准：工作日加班按1.5倍工资计算，休息日加班按2倍，法定节假日按3倍。",
    # ID:5  "住宿标准：一线城市不超过500元/天..."
    "出差住宿标准：一线城市不超过500元/天，二线城市不超过350元/天，其他城市不超过200元/天。",
    # ID:6  "报销需在费用发生后30日内提交..."
    "报销需在费用发生后30日内提交申请。单次超过5000元的报销需总经理审批。",
    # ID:7  "公司实行弹性工作制，核心工作时间10:00-17:00..."
    "公司实行弹性工作制。核心工作时间为上午10:00至17:00，午休1.5小时。",
    # ID:8  "婚假为3天，产假按国家规定执行..."
    "婚假为3天，需在结婚登记之日起1年内休完。产假按国家规定执行，陪产假为7天。",
    # ID:9  "试用期为3个月，试用期工资为转正工资的80%..."
    "公司标准试用期为3个月，优秀者可提前转正。试用期工资为转正工资的80%。",
]

# 测试查询 + 人工标注的正确答案 Chunk ID
test_queries = [
    {"query": "事假需要提前多久申请？",        "relevant_ids": [0]},
    {"query": "病假怎么规定的？",               "relevant_ids": [1]},
    {"query": "年假有多少天？",                  "relevant_ids": [2]},
    {"query": "公司给员工交什么保险？",          "relevant_ids": [3]},
    {"query": "加班怎么算钱？",                  "relevant_ids": [4]},
    {"query": "出差住宿标准多少钱？",            "relevant_ids": [5]},
    {"query": "报销要多久之内提交？",            "relevant_ids": [6]},
    {"query": "公司几点上班？",                  "relevant_ids": [7]},
    {"query": "婚假可以休几天？",                "relevant_ids": [8]},
    {"query": "试用期多长？试用期工资多少？",     "relevant_ids": [9]},
    {"query": "请假和加班分别是怎样的？",        "relevant_ids": [0, 4]},  # 多正确答案
    {"query": "保险和住宿怎么规定的？",          "relevant_ids": [3, 5]},  # 多正确答案
]


# ═══════════════════════════════════════════════════════════════
# Step 2: 向量化
# ═══════════════════════════════════════════════════════════════

def build_embeddings_openai(chunks: List[str], model: str, dims: int):
    """用 OpenAI 做 Embedding (需要 API Key)"""
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(
        model=model, input=chunks, dimensions=dims
    )
    vectors = [np.array(d.embedding, dtype=np.float32) for d in response.data]
    return np.stack(vectors)


def build_embeddings_simulated(chunks: List[str], dims: int = 128,
                                seed: int = 42) -> np.ndarray:
    """
    用随机向量模拟 Embedding (用于在没有 API Key 时演示评估逻辑)。

    模拟策略: 为每个 Chunk 生成一个"主题向量"(3维)，
    相同主题的 Chunk 的向量方向会更接近。
    这模拟了真实 Embedding 的语义聚合行为。
    """
    rng = np.random.default_rng(seed)

    # 手动定义每个 Chunk 的"主题" (用于模拟相似度)
    # 0=请假, 1=保险, 2=加班, 3=报销, 4=考勤, 5=婚假产假, 6=入职
    topic_vectors = {
        0: np.array([0.9, 0.1, 0.0]),  # 请假
        1: np.array([0.8, 0.2, 0.0]),  # 请假
        2: np.array([0.7, 0.3, 0.0]),  # 请假(年假)
        3: np.array([0.0, 0.9, 0.1]),  # 保险
        4: np.array([0.1, 0.0, 0.9]),  # 加班
        5: np.array([0.2, 0.1, 0.8]),  # 报销(住宿)
        6: np.array([0.3, 0.1, 0.7]),  # 报销(流程)
        7: np.array([0.0, 0.0, 1.0]),  # 考勤
        8: np.array([0.5, 0.0, 0.5]),  # 婚假产假
        9: np.array([0.0, 0.5, 0.5]),  # 入职
    }

    vectors = np.zeros((len(chunks), dims), dtype=np.float32)
    for i in range(len(chunks)):
        # 用主题向量作为"种子"，扩展到完整维度 + 噪声
        topic = topic_vectors.get(i, np.zeros(3))
        base = np.zeros(dims)
        base[:3] = topic * 5.0  # 放大信号
        base[3:] = rng.normal(0, 0.3, dims - 3)  # 噪声维度
        vectors[i] = base

    # L2 归一化
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════
# Step 3: 评估函数
# ═══════════════════════════════════════════════════════════════

def evaluate_retrieval(
    query_vectors: np.ndarray,       # (N_queries, D)
    chunk_vectors: np.ndarray,       # (N_chunks, D)
    test_queries: List[Dict],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict:
    """
    评估检索系统的 Hit Rate@K 和 MRR@K。

    参数:
      query_vectors:  每个测试查询的向量
      chunk_vectors:  所有知识库 Chunk 的向量
      test_queries:   测试数据 (含 query 文本和 relevant_ids)
      k_values:       评估哪些 K 值

    返回:
      每个 K 值对应的 Hit Rate 和 MRR
    """
    results = {k: {"hit_rate": 0.0, "mrr": 0.0, "detail": []}
               for k in k_values}

    # ── 对每个查询计算相似度并排序 ──────────────────────────
    # 所有向量已经 L2 归一化过 → 内积 = 余弦相似度
    sim_matrix = np.dot(query_vectors, chunk_vectors.T)  # (Q, C)
    # 按相似度降序排列, 取每个 Query 的排名
    ranked_indices = np.argsort(-sim_matrix, axis=1)     # (Q, C)

    for q_idx, test_item in enumerate(test_queries):
        relevant = set(test_item["relevant_ids"])  # 正确答案集合
        ranking = ranked_indices[q_idx]            # 这个 Query 的排序结果

        for k in k_values:
            top_k = ranking[:k]  # Top-K 个 Chunk 的 ID

            # ── Hit Rate: Top-K 中是否有正确答案? ──────────
            hit = bool(set(top_k) & relevant)
            if hit:
                results[k]["hit_rate"] += 1

            # ── MRR: 第一个正确答案的排名 ─────────────────
            first_rank = None
            for rank, chunk_id in enumerate(top_k, start=1):
                if chunk_id in relevant:
                    first_rank = rank
                    break

            if first_rank is not None:
                rr = 1.0 / first_rank   # Reciprocal Rank
                results[k]["mrr"] += rr
                results[k]["detail"].append({
                    "query": test_item["query"],
                    "first_rank": first_rank,
                    "rr": round(rr, 4),
                    "hit": True,
                })
            else:
                results[k]["detail"].append({
                    "query": test_item["query"],
                    "first_rank": None,
                    "rr": 0.0,
                    "hit": False,
                })

    # ── 归一化 ────────────────────────────────────────────────
    N = len(test_queries)
    for k in k_values:
        results[k]["hit_rate"] = round(results[k]["hit_rate"] / N, 4)
        results[k]["mrr"] = round(results[k]["mrr"] / N, 4)

    return results


# ═══════════════════════════════════════════════════════════════
# Step 4: 执行评估
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  Embedding 检索评估 —— Hit Rate@K 与 MRR@K")
print("=" * 70)

# 向量化
print("\n[1] 向量化 Chunk 和 Query...")
chunk_vecs = build_embeddings_simulated(knowledge_chunks, dims=128, seed=42)

query_texts = [q["query"] for q in test_queries]
# Query 向量也用同样的模拟方法
query_vecs = np.array([
    chunk_vecs[q["relevant_ids"][0]] + np.random.normal(0, 0.15, 128)
    for q in test_queries
], dtype=np.float32)
query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)

# 评估
print("[2] 评估中...")
results = evaluate_retrieval(query_vecs, chunk_vecs, test_queries,
                              k_values=[1, 3, 5, 10])

# ═══════════════════════════════════════════════════════════════
# Step 5: 输出报告
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  评估报告")
print("=" * 70)

print(f"\n  {'K':<6} {'Hit Rate':<12} {'MRR':<12} {'解读'}")
print(f"  {'─' * 50}")
for k in [1, 3, 5, 10]:
    hr = results[k]["hit_rate"]
    mrr = results[k]["mrr"]
    if k == 1:
        note = "最严格——第1名必须是正确答案"
    elif k == 5:
        note = "最常用——Top-5 是 LLM 输入的典型值"
    elif k == 10:
        note = "宽松——衡量粗排的召回能力"
    else:
        note = ""
    print(f"  K={k:<3}  {hr:<12.2%}  {mrr:<12.4f}  {note}")

# ── 详细结果 ────────────────────────────────────────────────
print(f"\n  ── 每条查询的详细结果 (K=5) ──")
k = 5
for i, detail in enumerate(results[k]["detail"]):
    rank_str = f"第{detail['first_rank']}位" if detail['first_rank'] else "未命中"
    status = "✓" if detail['hit'] else "✗"
    print(f"  [{status}] {detail['query'][:30]:<35s} → RR={detail['rr']:.4f} ({rank_str})")

# ── 诊断分析 ────────────────────────────────────────────────
print(f"\n  ── 诊断分析 ──")
hr5 = results[5]["hit_rate"]
mrr5 = results[5]["mrr"]

if hr5 > 0.8 and mrr5 > 0.6:
    print(f"  高 Hit Rate + 高 MRR → ★ 理想状态! Embedding 质量好, 排序准确")
elif hr5 > 0.8 and mrr5 <= 0.4:
    print(f"  高 Hit Rate + 低 MRR → 正确答案在 Top-5 但排名靠后")
    print(f"    调优建议: 加 Reranker 精排; 查询改写提高精确度")
elif hr5 <= 0.6 and mrr5 > 0.5:
    print(f"  低 Hit Rate + 高 MRR → 已命中质量高, 但漏检多")
    print(f"    调优建议: 补充知识库; 增大 chunk_size; 混合检索(BM25)")
else:
    print(f"  低 Hit Rate + 低 MRR → 需要根本性调整")
    print(f"    调优建议: 更换 Embedding 模型; 重新设计切分策略")


# ═══════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  评估总结")
print("=" * 70)
print("""
  Key Takeaways:
    1. Hit Rate@K → 衡量"找不找得到"(召回能力)
    2. MRR@K     → 衡量"排得对不对"(排序质量)
    3. 不同 K 值揭示不同的问题:
       K=1   → 理想排序能力
       K=5   → 实用生成质量
       K=10  → 粗排召回能力
    4. 高 Hit + 低 MRR → 加 Reranker 效果最明显
    5. 低 Hit + 高 MRR → 补充知识库或切分策略
    6. 评估不是一次性的——持续监控, 持续优化
""")
```

---

### 2.8 总结：Embedding 模型选型思考框架

```
═══════════════════════════════════════════════════════════════════════
            Embedding 模型选型决策树
═══════════════════════════════════════════════════════════════════════

  Q1: 延迟敏感吗? (需要 < 50ms 的响应时间?)
    ├─ 是 → 本地部署 (sentence-transformers / TEI)
    │       不用等网络往返, 本地推理 5-20ms
    │
    └─ 否 → API 或本地都可以, 看其他因素

  Q2: 文档是什么语言?
    ├─ 纯中文 → bge-large-zh-v1.5 (开源首选)
    │           text-embedding-3-large (API 精度最高)
    │
    ├─ 纯英文 → text-embedding-3-small/large
    │           Cohere embed-v3
    │
    └─ 中英混合 / 多语言 → bge-m3 (开源多语言)
                          text-embedding-3-large (API 多语言)
                          jina-embeddings-v3 (开源长文本)

  Q3: 文档长度? (Chunk 的 token 数)
    ├─ 短 chunk (< 512t) → 大多数模型都支持
    │
    ├─ 中长 chunk (512-2000t) → bge-large-zh / text-embedding-3 都 OK
    │
    └─ 长 chunk (2000-8192t) → text-embedding-3 系列 (8192t)
                                jina-embeddings-v3 (8192t)
                                bge-m3 (8192t)
                                注意: 切分策略需匹配模型最大输入

  Q4: 成本预算?
    ├─ 无 GPU / 小规模 (< 10万条) → API (text-embedding-3-small)
    │                               → 总成本通常 < $10
    │
    ├─ 有 GPU / 大规模 (> 100万条) → 本地部署 (bge / m3e)
    │                                 长期 TCO 更低
    │
    └─ 不想管运维 → API 或 Pinecone (自带 Embedding)

  Q5: 数据合规要求?
    ├─ 数据不能出内网 → 本地部署 (bge / m3e / gte)
    │
    └─ 无合规要求 → API 或本地都可以

═══════════════════════════════════════════════════════════════════════
```

**最终建议：**

> 1. **先用 token-embedding-3-small 做原型。** 成本极低 ($0.02/M)，中文能力中上，MRL 支持维度截断。快速跑通流程后再根据评估结果决定是否替换。
> 2. **中文生产环境首选 bge-large-zh-v1.5。** C-MTEB 排名第一，开源免费，社区活跃。配合 TEI 部署可以达到生产级吞吐。
> 3. **跨语言场景用 bge-m3 或 text-embedding-3-large。** bge-m3 同时支持稠密+稀疏检索，是混合检索的理想 Embedding 底座。
> 4. **用 Hit Rate@5 和 MRR@5 作为核心评估指标。** 它们是 RAG 生成质量的直接预测因子。不要只看一个——高 Hit + 低 MRR 和高 MRR + 低 Hit 需要不同的优化方向。


---

