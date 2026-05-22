## 十五、LlamaIndex 在 RAG 中的应用 —— 知识管理的核心框架

在前面章节中，我们用手写代码和 Unstructured 解决了"如何把文档变成可检索的数据"。但当 RAG 系统愈发复杂——多个数据源、多种索引策略、需要对话记忆、需要动态路由——纯手写代码的维护成本就会急剧上升。

**LlamaIndex 就是为这个阶段设计的。** 它是一个专门为 LLM 应用构建"外部知识管理层"的框架。如果说 Unstructured 专注于"读懂文档"，LlamaIndex 专注于的是**"组织知识、高效检索、精准回答"**。

> 官方定义："LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data."
>
> 通俗理解：LlamaIndex 就是大模型的"外部大脑"——模型负责推理，LlamaIndex 负责告诉模型"你应该参考哪些信息来回答"。

它同时提供了 **Python** 和 **TypeScript (LlamaIndexTS)** 两个版本，分别用于后端服务和前端/边缘侧集成。

---

### 15.1 核心架构：用户-索引-LLM 的三角闭环

LlamaIndex 的整个 RAG 架构可以抽象为一个以**索引为中心**的三角闭环：

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LlamaIndex RAG 三角闭环架构                               ║
║                                                                              ║
║                              ┌──────────────────┐                           ║
║                              │     索引 (Index)   │                          ║
║                              │                  │                           ║
║                              │  所有知识的"目录"  │                           ║
║                              │  · 向量索引        │                           ║
║                              │  · 树形索引        │                           ║
║                              │  · 关键词索引      │                           ║
║                              │  · 知识图谱索引    │                           ║
║                              └────────┬─────────┘                           ║
║                                       │                                      ║
║                    ┌──────────────────┼──────────────────┐                   ║
║                    │                  │                  │                   ║
║                    ▼                  │                  ▼                   ║
║          ┌─────────────────┐          │        ┌─────────────────┐         ║
║          │   用户 (User)    │          │        │    LLM (大模型)  │         ║
║          │                 │          │        │                 │         ║
║          │ "事假要提前      │          │        │  GPT-4o / Claude│         ║
║          │  多久申请？"    │          │        │  DeepSeek / Qwen│         ║
║          └────────┬────────┘          │        └────────┬────────┘         ║
║                   │                   │                  │                   ║
║                   │   ① Query         │                  │                   ║
║                   │ ──────────────────▶                  │                   ║
║                   │   用户提出自然语言查询                 │                   ║
║                   │                   │                  │                   ║
║                   │                   │  ② Augmented Prompt                 ║
║                   │                   │  (System指令 + Query + Relevant Data)│
║                   │                   │ ─────────────────▶                  ║
║                   │                   │                  │                   ║
║                   │                   │                  │  ③ Response       ║
║                   │                   │                  │ ◀────────────────  │
║                   │                   │                  │                   ║
║                   │   ④ 答案 + 溯源   │                  │                   ║
║                   │ ◀─────────────────┘                  │                   ║
║                   │                   │                  │                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

数据流动详解：

  ① User → Index：用户自然语言问题被向量化后查询索引
     内部：Query → Embedding → FAISS/Milvus 检索 → Top-K Node 列表

  ② Index → LLM：检索到的 Node 拼入 System Prompt，形成增强 Prompt
     格式：System (角色+规则) + 参考资料(检索到的 Node) + 用户问题

  ③ LLM → Index/User：LLM 阅读增强 Prompt，生成回答
     约束：temperature=0.3 保证低幻觉、依据参考资料而非"编造"

  ④ Index/QueryEngine → User：回到答案 + 可选的引用来源 (source_nodes)
```

**这个架构的核心思想：**

- **索引是中心枢纽。** 所有数据被加载、切分、向量化后存入索引。查询时，索引是唯一的检索入口。你可以有不同的索引策略（向量/树形/关键词/图谱），但它们都在这个"三角"中扮演相同的角色。
- **LLM 不直接接触原始数据。** LLM 看到的是"经过索引筛选后的、最相关的那几条信息"，而不是全量文档。这从根本上解决了上下文窗口限制。
- **用户提问驱动整个流程。** 每次查询都是一次完整的"检索 → 增强 → 生成"循环。

---

### 15.2 六阶段管线：从数据到答案

LlamaIndex 将一个完整的 RAG 流程抽象为六个阶段，每个阶段有对应的组件，LLM 在其中的三个阶段扮演不同角色：

```
═══════════════════════════════════════════════════════════════════════════════
                LlamaIndex 六阶段 RAG 管线（附 LLM 连接点）
═══════════════════════════════════════════════════════════════════════════════

  阶段 1: 数据源 (Data Sources)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PDF · Word · HTML · Markdown · SQL · API · CSV · 代码仓库 · Slack · 邮件│
  │  任何能被"读取"的数据都可以作为数据源                                       │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  阶段 2: 加载 (Loading) —— 不进 LLM
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LlamaHub 提供 300+ Reader：                                             │
  │    SimpleDirectoryReader  → 读取整个目录（自动检测格式）                   │
  │    PDFReader / DocxReader  → 逐格式专用 Reader                           │
  │    DatabaseReader          → SQL 查询结果作为 Document                    │
  │    GithubRepositoryReader  → 代码仓库                                    │
  │                                                                         │
  │  输出：List[Document]    {text: str, metadata: dict}                     │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  阶段 3: 切分 (Chunking / Node Parsing) —— 不进 LLM
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Document → Node 的转换过程。Node = Document + node_id + relationships   │
  │                                                                         │
  │  切分策略：                                                              │
  │    SentenceSplitter          → 按句子边界切分                            │
  │    TokenTextSplitter         → 按 token 数精确控制                       │
  │    SemanticSplitterNodeParser→ 按语义相似度切分（需要 Embedding 模型）    │
  │    SentenceWindowNodeParser  → 保留上下文窗口（检索后自动扩展相邻句子）   │
  │                                                                         │
  │  关键概念：Node 之间有 relationships 属性，构成关系图                     │
  │    node.relationships = {                                                │
  │        NodeRelationship.SOURCE: SourceNode,  # 父文档                    │
  │        NodeRelationship.PREVIOUS: PrevNode,  # 前一个 Node               │
  │        NodeRelationship.NEXT: NextNode       # 后一个 Node               │
  │    }                                                                    │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  阶段 4: 索引构建 (Indexing) ← LLM 用于 Embedding
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  将 Node 向量化后存入索引结构。                                           │
  │                                                                         │
  │  内置索引类型：                                                          │
  │    VectorStoreIndex      → 向量索引（最常用，对接 FAISS/Milvus/Qdrant）  │
  │    SummaryIndex          → 摘要索引（所有 Node → 摘要 → 检索时合成）     │
  │    TreeIndex             → 树形索引（自顶向下分层检索）                   │
  │    KeywordTableIndex     → 关键词索引（每个关键词映射到包含它的 Node）    │
  │    KnowledgeGraphIndex   → 知识图谱索引（实体+关系三元组）                │
  │                                                                         │
  │  ┌─ LLM 连接点 1：Embedding 模型 ─────────────────────────────────────┐ │
  │  │  Node.text → text-embedding-3-small / bge-large-zh → 向量          │ │
  │  │  这里的 "LLM" 是编码器（Encoder），把文字转换为数字向量              │ │
  │  │  不生成任何自然语言，只输出 float32 数组                             │ │
  │  └────────────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  阶段 5: 检索 + 前后处理 (Retrieval & Processing) ← LLM 用于查询理解
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  ┌─ 检索前处理 (Pre-Retrieval) ───────────────────────────────────┐    │
  │  │  QueryTransform     → 查询改写（模糊→精确）                      │    │
  │  │  RouterQueryEngine  → 路由：根据问题类型选不同的索引/检索器       │    │
  │  │  SubQuestionQueryEngine → 复杂问题分解为子问题，逐个子问题检索   │    │
  │  │  HyDE (Hypothetical Document Embedding) → 先构造假设文档再检索  │    │
  │  └────────────────────────────────────────────────────────────────┘    │
  │                              │                                          │
  │                              ▼                                          │
  │  ┌─ 检索 (Retrieval) ─────────────────────────────────────────────┐   │
  │  │  VectorIndexRetriever  → 向量检索（稠密）                       │   │
  │  │  BM25Retriever         → 关键词检索（稀疏）                     │   │
  │  │  QueryFusionRetriever  → 混合检索（多路融合）                   │   │
  │  │  RecursiveRetriever    → 递归检索（检索 Node → 再去检索其父节点）│   │
  │  │  AutoMergingRetriever  → 自动合并检索（小 chunk 检索 → 大 chunk 输出）│ │
  │  └────────────────────────────────────────────────────────────────┘   │
  │                              │                                          │
  │                              ▼                                          │
  │  ┌─ 检索后处理 (Post-Retrieval) ──────────────────────────────────┐   │
  │  │  SimilarityPostprocessor      → 相似度阈值过滤                  │   │
  │  │  KeywordNodePostprocessor     → 关键词必须/排除过滤             │   │
  │  │  MetadataReplacementPostprocessor → 用父 Node 的 metadata 替换  │   │
  │  │  SentenceTransformerRerank    → Cross-Encoder 精排              │   │
  │  │  LongContextReorder           → 长上下文：相关→中间，不相关→边缘 │   │
  │  └────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌─ LLM 连接点 2：查询理解 ────────────────────────────────────────┐   │
  │  │  "那个请假政策是什么" → LLM 改写 → "公司员工请假申请规定和流程"    │   │
  │  │  这里的 LLM 扮演"翻译官"：把口语化的模糊问题                                   │   │
  │  │  翻译成适合检索的精确查询（如 HyDE、QueryRewrite）                           │   │
  │  └────────────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  阶段 6: 生成回答 (Response Synthesis) ← LLM 用于回答生成
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  ResponseSynthesizer 的工作流程：                                         │
  │    1. 接收检索到的 List[NodeWithScore]（Node + 相似度分数）               │
  │    2. 将 Node 文本 + 用户问题 + System Prompt 拼成完整 Prompt             │
  │    3. 调用 LLM 生成回答                                                   │
  │    4. 可选：在回答中附带引用来源 (source_nodes)                            │
  │                                                                         │
  │  内置合成模式：                                                           │
  │    CompactAndRefine → 逐块压缩+精炼（适合小块数据）                       │
  │    TreeSummarize    → 自底向上树形总结（适合多块数据）                    │
  │    SimpleSummarize  → 直接拼接+一次生成（适合少量块）                     │
  │                                                                         │
  │  ┌─ LLM 连接点 3：回答生成 ────────────────────────────────────────┐   │
  │  │  Prompt + 参考资料 → GPT-4o / Claude / DeepSeek → 自然语言回答    │   │
  │  │  这里的 LLM 是"生成器"：阅读、理解、综合、输出                      │   │
  │  │  使用低 temperature (0.3) 保证回答忠实于参考资料                                   │   │
  │  └────────────────────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                   LLM 在三阶段中的角色总结
═══════════════════════════════════════════════════════════════════════════════

  ┌────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │                │  阶段 4: 索引构建  │  阶段 5: 查询理解  │  阶段 6: 回答生成  │
  ├────────────────┼──────────────────┼──────────────────┼──────────────────┤
  │ LLM 扮演角色    │  编码器 (Encoder) │  翻译官 (Translator)│ 生成器 (Generator) │
  │ 输入            │  文本字符串       │  模糊的自然语言    │  增强后的 Prompt   │
  │ 输出            │  浮点数向量       │  精确的查询文本    │  自然语言回答      │
  │ 典型模型        │  text-embedding-3 │  GPT-4o-mini      │  GPT-4o / Claude   │
  │                │  bge-large-zh     │  DeepSeek-V3      │  DeepSeek-V3       │
  │ 是否生成自然语言 │  ❌ 否            │  ✅ 是             │  ✅ 是             │
  └────────────────┴──────────────────┴──────────────────┴──────────────────┘

  关键理解：阶段 4 的 Embedding 模型 和 阶段 5/6 的生成式 LLM 通常不是同一个模型。
           Embedding 模型（如 text-embedding-3-small）是专门训练用来做语义表示的编码器，
           LLM（如 GPT-4o）是专门训练用来做文本生成的解码器。
           两者在 RAG 中各司其职，不可混用。
```

---

### 15.3 LangChain vs LlamaIndex 深度对比

虽然两者在 RAG 生态中有重叠，但定位和设计哲学有本质区别：

#### 15.3.1 定位差异

```
  LangChain   → 通用 LLM 应用框架
                "Everything is a Chain"（一切皆链式编排）

                核心场景：Agent 编排 · 工具调用 · 对话管理 · Workflow 自动化
                RAG 只是它的一个模块，不是唯一重心
                通过 LangGraph 提供了目前业界最强的 Agent 编排能力

  LlamaIndex  → 数据索引与 LLM 知识管理框架
                "Everything is an Index"（一切皆索引）

                核心场景：RAG · 文档问答 · 知识库检索 · 数据增强生成
                RAG 是它的"天职"——所有设计围绕"如何高效检索外部知识"
                提供最丰富、最专业的索引和检索策略体系
```

#### 15.3.2 多维度能力对比

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **核心理念** | 链式编排 (Chain/LCEL)——组件串联形成 Pipeline | 索引中心 (Index)——一切围绕数据索引构建 |
| **数据结构** | Document → `{page_content, metadata}` | Document → Node → `{text, metadata, node_id, relationships}` |
| **节点关系** | Document 之间无关系 | Node 之间有 `PREVIOUS / NEXT / SOURCE / PARENT / CHILD` 关系图 |
| **索引结构** | VectorStore 抽象层，对接外部向量库 | **6+ 种内置索引**：VectorStoreIndex / SummaryIndex / TreeIndex / KeywordTableIndex / KnowledgeGraphIndex / 等 |
| **检索能力** | 简单向量检索（依赖 VectorStore 实现） | **多策略检索**：递归检索 / 路由检索 / 混合检索 / 子问题分解 / 自动合并检索 |
| **检索前处理** | 需手动编排（或通过 Hub 拉取 Prompt） | 内置 QueryTransform / Router / SubQuestion / HyDE |
| **检索后处理** | 需手动实现 | 内置 NodePostprocessor 系列：相似度过滤 / 关键词过滤 / 元数据替换 / Rerank |
| **上下文分析** | 无内置能力 | SentenceWindowNodeParser（检索后自动扩展上下文窗口）<br>MetadataReplacementPostProcessor（用父节点元数据替换子节点） |
| **文档加载** | 200+ DocumentLoader，与 Unstructured 集成 | 300+ Reader (LlamaHub)，与 Unstructured 深度集成 |
| **多模态** | 较弱，需手动处理 | 内置 ImageReader / AudioReader / VideoReader |
| **对话记忆** | Memory 组件 (ConversationBufferMemory 等) | ChatEngine + CondenseQuestionChatEngine |
| **Agent 编排** | **LangGraph**（业界最强、有状态多步骤编排） | 内置 Agent（ReAct/OpenAIAgent），编排弱于 LangGraph |
| **评估体系** | LangSmith (追踪+评估) | 内置 Evaluation 模块 (FaithfulnessEvaluator, RelevancyEvaluator) |
| **部署** | LangServe (一键 API) | LlamaCloud (托管索引服务) |
| **学习曲线** | 中等偏高 (概念多：Chain/Agent/Tool/Memory/LCEL) | 较低 (概念聚焦：Index/Node/Retriever/QueryEngine) |
| **生态大小** | GitHub Stars ~95K+，社区更大 | GitHub Stars ~38K+，增长迅速 |

#### 15.3.3 场景选型决策树

```
你的项目核心是什么？

  ┌─ 主要是 RAG / 知识库问答 / 文档检索
  │   → LlamaIndex
  │     理由：索引策略丰富（6+种）、检索优化内置（递归/路由/混合）
  │           上下文分析独特（SentenceWindow/自动元数据替换）
  │           5 行代码搭建 RAG 原型
  │
  ├─ 主要是 Agent 编排 / 多工具调用 / 复杂 Workflow
  │   → LangChain + LangGraph
  │     理由：LangGraph 的 Agent 编排和状态管理能力业界最强
  │           生态更大，第三方集成更全面
  │
  ├─ RAG + Agent 都需要
  │   → 两者混用
  │     LlamaIndex 负责：文档加载 → 索引构建 → 检索
  │     LangChain 负责：对话管理 → Agent 调度 → 工具编排
  │     两者有官方集成，LlamaIndex 的 QueryEngine 可以封装为 LangChain Tool
  │
  └─ 快速原型验证 / 小规模知识库
      → LlamaIndex
        理由：上手最快，概念最聚焦，代码量最少
```

---

### 15.4 核心组件：按文档格式分类的 Reader

LlamaIndex 通过 **LlamaHub** 提供了 300+ 开箱即用的数据加载器，覆盖几乎所有常见数据源。以下是按格式分类的核心组件：

#### 15.4.1 简单格式 —— SimpleDirectoryReader

```python
from llama_index.core import SimpleDirectoryReader

# 一行代码读取整个目录，自动检测格式
documents = SimpleDirectoryReader(
    input_dir="./docs",
    recursive=True,                # 递归子目录
    required_exts=[".md", ".txt"], # 只读指定格式
    exclude=["temp", ".git"],      # 排除目录
    file_metadata=lambda fp: {"category": "policies"},  # 自定义元数据
).load_data()
# 返回 List[Document]，每个 Document 有 text + metadata
```

**底层原理：** SimpleDirectoryReader 内部维护了一个文件后缀→Reader 的映射表。读取每个文件时，根据后缀自动选择对应的 Reader（PDF → PDFReader、.docx → DocxReader 等）。

#### 15.4.2 PDF 专用 —— PDFReader

```python
from llama_index.readers.file import PDFReader

# LlamaIndex 的 PDFReader 会自动降级选择解析引擎：
# 优先级：Unstructured → PyMuPDF → PyPDF2（按可用性自动降级）
documents = PDFReader().load_data(file="员工手册.pdf")
```

#### 15.4.3 多格式混合 —— file_extractor 配置

```python
from llama_index.core import SimpleDirectoryReader

# 为不同格式指定不同解析引擎
documents = SimpleDirectoryReader(
    input_dir="./mixed_docs",  # 包含 PDF、Word、HTML 的混合目录
    file_extractor={
        ".pdf": "unstructured",   # PDF 用 Unstructured 解析
        ".docx": "unstructured",  # Word 用 Unstructured 解析
        ".html": "default",       # HTML 用默认（内置 BeautifulSoup）
        ".pptx": "unstructured",  # PPT 用 Unstructured 解析
        ".csv": "default",        # CSV 用内置解析
    }
).load_data()
```

#### 15.4.4 表格专用 —— CSVReader / PandasExcelReader

```python
from llama_index.readers.file import CSVReader, PandasExcelReader

# CSV —— 每行作为一个独立的 Document
csv_docs = CSVReader(concat_rows=False).load_data(file="数据.csv")

# Excel —— 每个 Sheet 页作为一个 Document
excel_docs = PandasExcelReader(
    concat_rows=False,
    sheet_name=None,          # None = 所有 Sheet
).load_data(file="财务报表.xlsx")
```

#### 15.4.5 其他数据源

```python
# 数据库
from llama_index.readers.database import DatabaseReader
db_docs = DatabaseReader(
    sqlalchemy_uri="postgresql://user:pass@localhost/db"
).load_data(query="SELECT * FROM policies WHERE active=TRUE")

# 网页
from llama_index.readers.web import SimpleWebPageReader
web_docs = SimpleWebPageReader(html_to_text=True).load_data(
    urls=["https://example.com/policy"])

# 代码仓库
from llama_index.readers.github import GithubRepositoryReader
github_docs = GithubRepositoryReader(
    owner="your-org", repo="your-repo"
).load_data(branch="main")

# Slack / Notion / Google Docs / Jira / Confluence ...
# 在 LlamaHub (llamahub.ai) 上搜索对应的 Reader
```

---

### 15.5 LlamaIndex 与 Unstructured 的集成

#### 15.5.1 两者的天然分工

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    RAG 技术栈中的分工                                       ║
║                                                                          ║
║  原始文档                                                                ║
║  (PDF / Word / HTML / PPT / 图片 / 邮件 / ...)                           ║
║     │                                                                    ║
║     ▼                                                                    ║
║  ┌──────────────────────────┐                                           ║
║  │      Unstructured         │  ← 文档提取引擎                           ║
║  │                          │                                           ║
║  │  核心能力：               │   解决的问题：                              ║
║  │  · 格式解析               │   "PDF 里的表格怎么转成结构化的文字？"       ║
║  │  · 布局检测 (视觉模型)     │   "扫描件怎么识别出文字？"                  ║
║  │  · OCR 文字识别           │   "多栏排版的阅读顺序是什么？"               ║
║  │  · 表格结构还原           │   "页眉页脚怎么去掉？"                      ║
║  │  · 元数据提取             │   "每段文字在页面上的什么位置？"             ║
║  └──────────┬───────────────┘                                           ║
║             │                                                            ║
║             │ Elements 列表                                              ║
║             │ (每个 Element 带类型标签 + 坐标 + 层级 + 字体等元数据)        ║
║             │                                                            ║
║             ▼                                                            ║
║  ┌──────────────────────────┐                                           ║
║  │      LlamaIndex          │  ← 知识管理引擎                            ║
║  │                          │                                           ║
║  │  核心能力：               │   解决的问题：                              ║
║  │  · 索引构建 (6+ 种索引)   │   "如何高效组织这些文字以便快速检索？"       ║
║  │  · 检索策略 (多路/递归)   │   "复杂的跨文档查询怎么拆解？"               ║
║  │  · 查询增强 (HyDE等)     │   "模糊问题怎么准确找到相关内容？"           ║
║  │  · Prompt 合成           │   "如何把检索到的信息交给 LLM？"            ║
║  │  · 对话管理              │   "多轮对话中怎么记住上下文？"               ║
║  └──────────┬───────────────┘                                           ║
║             │                                                            ║
║             │ 检索到的相关 Node + Prompt                                   ║
║             │                                                            ║
║             ▼                                                            ║
║  ┌──────────────────────────┐                                           ║
║  │         LLM              │  ← 推理生成引擎                             ║
║  │  (GPT-4o / Claude / ...) │   解决："给定参考资料，如何生成准确答案？"   ║
║  └──────────────────────────┘                                           ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**一句话总结：** Unstructured 把"难读的格式"变成"可理解的数据 (Elements)"，LlamaIndex 把"可理解的数据"变成"可问答的知识 (Index + QueryEngine)"。

#### 15.5.2 集成方式

**方式一：通过 SimpleDirectoryReader 的 file_extractor 配置**

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir="./docs",
    file_extractor={
        ".pdf": "unstructured",
        ".docx": "unstructured",
        ".html": "unstructured",
        ".pptx": "unstructured",
    },
).load_data()
# LlamaIndex 内部自动调用 Unstructured 的 partition()
# 然后将 Elements 转换为 LlamaIndex 的 Document 对象
```

**方式二：直接使用 UnstructuredReader**

```python
from llama_index.readers.unstructured import UnstructuredReader

reader = UnstructuredReader()
documents = reader.load_data(
    file="员工手册.pdf",
    # unstructured_kwargs 会透传给 Unstructured 的 partition()
    unstructured_kwargs={
        "strategy": "hi_res",
        "languages": ["chi_sim", "eng"],
        "pdf_infer_table_structure": True,
        "include_metadata": True,
    }
)
# 返回的 documents 是标准的 LlamaIndex Document 列表
# 可以直接喂给后续的索引构建
```

#### 15.5.3 两种使用方式对比：原生 Unstructured vs LlamaIndex 集成

这是很多开发者的困惑点——到底用原生 Unstructured 还是用 LlamaIndex 的 Reader 包装？

| 维度 | 原生 Unstructured | LlamaIndex 集成 (UnstructuredReader) |
|------|------------------|--------------------------------------|
| **底层调用** | `partition()` 直接调用，开发者完全控制 | LlamaIndex 封装调用，本质也是 `partition()` |
| **灵活度** | ⭐⭐⭐⭐⭐ 最高——完全控制所有参数和解析流程 | ⭐⭐⭐ 中等——通过 `unstructured_kwargs` 透传参数 |
| **可控性** | ⭐⭐⭐⭐⭐ 最高——拿到 Element 后完全自定义处理逻辑 | ⭐⭐⭐ 较高——LlamaIndex 自动做 Element→Document 转换，中间逻辑不可控 |
| **元数据保留** | ⭐⭐⭐⭐⭐ 全部保留（需手动映射到 Document） | ⭐⭐⭐ 部分保留（Document 结构可能丢弃某些 Element 元数据） |
| **集成便捷性** | ⭐⭐ 需手动转换 Element→Document→Node→Index | ⭐⭐⭐⭐⭐ 最便捷——一行代码出 Document，直接建索引 |
| **后续处理** | 需自己对接所有后续步骤 | ⭐⭐⭐⭐⭐ 零摩擦——`load_data()` 返回值直接可建索引 |
| **适用场景** | 精细控制解析过程、自定义预处理、保留全部元数据 | 快速标准 RAG 原型、中小规模知识库、与 LlamaIndex 管线深度集成 |

**代码对比示例：**

```python
# ═══════════════════════════════════════════════════════════
# 方式 A：原生 Unstructured（最大灵活度）
# ═══════════════════════════════════════════════════════════
from unstructured.partition.auto import partition
from llama_index.core import Document as LlamaDocument
from llama_index.core import VectorStoreIndex

# 第一步：用 Unstructured 解析（完全控制参数）
elements = partition(filename="员工手册.pdf", strategy="hi_res",
                     languages=["chi_sim", "eng"],
                     pdf_infer_table_structure=True)
# 此时 elements 是 Unstructured 的 Element 列表
# 每个 Element 有完整元数据（坐标、层级、字体等）

# 第二步：手动转换（可以在此步骤精确选择保留哪些 metadata）
llama_docs = []
for el in elements:
    doc = LlamaDocument(
        text=el.text,
        metadata={  # 只保留你需要的字段
            "page_number": el.metadata.page_number,
            "element_type": type(el).__name__,
            "source": el.metadata.filename,
            # 坐标/字体等信息选择性保留
        }
    )
    llama_docs.append(doc)

# 第三步：建索引
index = VectorStoreIndex.from_documents(llama_docs)

# ═══════════════════════════════════════════════════════════
# 方式 B：LlamaIndex 集成（最便捷）
# ═══════════════════════════════════════════════════════════
from llama_index.readers.unstructured import UnstructuredReader

# 一步完成：解析 + 转换 → 直接可建索引的 Document 列表
reader = UnstructuredReader()
documents = reader.load_data(
    file="员工手册.pdf",
    unstructured_kwargs={"strategy": "hi_res", "languages": ["chi_sim", "eng"]}
)
index = VectorStoreIndex.from_documents(documents)
```

**选择建议：**

```
你的需求是什么？

  ├─ 需要精细控制解析过程
  │   · 自定义去噪规则（删特定模式的文字）
  │   · 特殊表格处理（合并单元格的还原逻辑）
  │   · 需要 Element 的全部元数据做精细溯源
  │   → 用原生 Unstructured，解析后手动接入 LlamaIndex

  ├─ 快速搭建标准 RAG 原型
  │   · 中小规模知识库（几百到几千份文档）
  │   · 标准问答场景，不需要特殊解析逻辑
  │   → 用 LlamaIndex 集成，一行代码搞定的便利性无可比拟

  ├─ 已经用 LlamaIndex 做索引管理
  │   · 大部分文档用标准解析即可
  │   · 个别特殊文档需要深度定制
  │   → 两者混用：常规文档用集成方式，特殊文档用原生 Unstructured
  │      然后再统一喂给 LlamaIndex 建索引
  │
  └─ 不使用 LlamaIndex（自己搭管线）
      → 用原生 Unstructured，Element 的丰富元数据是自建管线的重要资产
```

---

### 15.6 总结：LlamaIndex 在 RAG 技术栈中的位置

```
                         RAG 全栈技术图谱

  ┌─────────────────────────────────────────────────────────────────┐
  │                        应用层                                    │
  │  FastAPI / Flask / Chainlit / Gradio / Streamlit                │
  ├─────────────────────────────────────────────────────────────────┤
  │                    知识管理层 ← LlamaIndex 核心定位              │
  │                                                                 │
  │  · 索引构建 (VectorStoreIndex / TreeIndex / KnowledgeGraph)    │
  │  · 检索策略 (向量 / 关键词 / 混合 / 递归 / 路由 / 子问题分解)   │
  │  · 查询引擎 (QueryEngine)                                       │
  │  · 对话引擎 (ChatEngine + 多轮上下文管理)                        │
  │  · 评估体系 (检索评估 + 回答评估)                                │
  ├─────────────────────────────────────────────────────────────────┤
  │                    文档提取层 ← Unstructured 核心定位             │
  │                                                                 │
  │  · 格式解析 (PDF/Word/HTML/PPT/图片/邮件...)                    │
  │  · 布局检测 (Visual Model: detectron2/YOLO)                    │
  │  · OCR 识别 (Tesseract / PaddleOCR)                             │
  │  · 表格还原 (Table Structure Inference)                        │
  │  · 元数据提取 (坐标 / 层级 / 字体 / 页码)                        │
  ├─────────────────────────────────────────────────────────────────┤
  │                    存储与计算层                                   │
  │                                                                 │
  │  · 向量数据库 (Milvus / Qdrant / Chroma / FAISS / Weaviate)    │
  │  · Embedding 模型 (text-embedding-3 / bge / m3e / jina)        │
  │  · LLM 推理 (GPT-4o / Claude / DeepSeek / Qwen / 本地模型)     │
  └─────────────────────────────────────────────────────────────────┘
```

**三个核心角色一览：**

| 角色 | 代表工具 | 解决的问题 |
|------|----------|-----------|
| **文档提取** | Unstructured | "如何从五花八门的文件格式中提取出结构化的文字和元数据？" |
| **知识管理** | LlamaIndex | "如何把这些文字组织成可高效检索的知识库，并在查询时精准回答？" |
| **推理生成** | LLM (GPT/Claude/DeepSeek) | "给定参考资料和用户问题，如何生成准确、连贯、有据可查的回答？" |

当你的 RAG 系统从"一个脚本 + 几份文档"发展到"多个数据源 + 复杂检索需求 + 需要对话记忆 + 需要评估质量"时，LlamaIndex 就是帮助你**从手写管线升级到专业框架**的关键一步。


---

