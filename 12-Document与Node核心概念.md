## 十九、LlamaIndex 核心对象概念 —— Document 与 Node

LlamaIndex 的数据模型围绕两个核心对象构建：**Document** 和 **Node**。理解它们之间的区别与关系，是掌握 LlamaIndex 索引和检索机制的基础。

### 19.1 从数据到索引：Document 与 Node 的位置

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                  LlamaIndex 数据流转全景                                     ║
║                                                                              ║
║  各种类型的数据源                                                             ║
║  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                     ║
║  │ PDF  │ │ Word │ │ HTML │ │ 图片 │ │ 代码 │ │ 数据库│                     ║
║  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                     ║
║     │        │        │        │        │        │                           ║
║     └────────┴────────┴────────┴────────┴────────┘                           ║
║                            │                                                 ║
║                 Reader / Loader (加载 + 解析)                                  ║
║                            │                                                 ║
║                            ▼                                                 ║
║  ╔══════════════════════════════════════════════════════════════════════╗   ║
║  ║                      Document 层                                    ║   ║
║  ║                                                                     ║   ║
║  ║  Document[0]: 《考勤管理制度》.pdf                                  ║   ║
║  ║    metadata: {author, created_date, category, version...}           ║   ║
║  ║    text: "第一章 工作时间\n第一条 公司实行标准...\n第二章 请假..."    ║   ║
║  ║                                                                     ║   ║
║  ║  Document[1]: 《费用报销管理办法》.pdf                              ║   ║
║  ║    metadata: {author, created_date...}                              ║   ║
║  ║    text: "第一章 报销范围\n第一条 因公产生的交通..."                  ║   ║
║  ║                                                                     ║   ║
║  ║  Document[2]: 《加班与调休制度》.pdf                                ║   ║
║  ║    metadata: {...}                                                  ║   ║
║  ║    text: "第一条 因工作需要加班的..."                                 ║   ║
║  ╚══════════════════════════════════════════════════════════════════╝   ║
║                            │                                                 ║
║                 NodeParser (切分解析)                                          ║
║                 SentenceSplitter / TokenTextSplitter / SemanticSplitter      ║
║                            │                                                 ║
║                            ▼                                                 ║
║  ╔══════════════════════════════════════════════════════════════════════╗   ║
║  ║                       Node 层                                       ║   ║
║  ║                                                                     ║   ║
║  ║  Node[0]: "第一章 工作时间\n第一条 公司实行标准工时制..."           ║   ║
║  ║    ├─ relationships: {SOURCE: Document[0]}  ← 来自哪个 Document     ║   ║
║  ║    ├─ metadata: 继承自 Document[0]                                  ║   ║
║  ║    └─ node_id: "a1b2c3..."                                         ║   ║
║  ║                                                                     ║   ║
║  ║  Node[1]: "第二章 请假制度\n第三条 事假需提前..."                    ║   ║
║  ║    ├─ relationships: {SOURCE: Document[0], PREVIOUS: Node[0],       ║   ║
║  ║    │                   NEXT: Node[2]}                               ║   ║
║  ║    └─ metadata: 继承自 Document[0] + chunk_index: 1                 ║   ║
║  ║                                                                     ║   ║
║  ║  Node[2]: "第四条 年假天数..."                                       ║   ║
║  ║    ├─ relationships: {SOURCE: Document[0], PREVIOUS: Node[1]}       ║   ║
║  ║    └─ ...                                                          ║   ║
║  ║                                                                     ║   ║
║  ║  Node[3]: "第一章 报销范围\n第一条 因公产生..."                      ║   ║
║  ║    └─ relationships: {SOURCE: Document[1]}  ← 来自不同 Document     ║   ║
║  ║                                                                     ║   ║
║  ╚══════════════════════════════════════════════════════════════════╗   ║
║                            │                                                 ║
║                            ▼                                                 ║
║  ╔══════════════════════════════════════════════════════════════════════╗   ║
║  ║                      Index 层                                       ║   ║
║  ║                                                                     ║   ║
║  ║  VectorStoreIndex / SummaryIndex / TreeIndex / KnowledgeGraphIndex  ║   ║
║  ║                                                                     ║   ║
║  ║  Node 是 Index 的构建原材料 —— 每个 Node 被向量化后存入索引          ║   ║
║  ║  检索时返回的是最相关的 Node（而非 Document）                         ║   ║
║  ╚══════════════════════════════════════════════════════════════════════╝   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  核心流程：
    数据源 → Reader → Document → NodeParser → Node → Index
            (加载)    (组织单元)    (切分)     (检索单元)  (存储+检索)
```

---

### 19.2 Document 与 Node 的对比

| 维度 | Document | Node |
|------|----------|------|
| **核心层级** | 文档层——代表一篇完整的原始文档 | 片段层——代表文档中的一个语义片段 |
| **核心职责** | 承载原始文档的完整内容和元数据；作为 metadata 的"父容器" | 作为索引的构建原材料；作为检索和生成的最小单元 |
| **内容** | 完整文档的全文 + 文档级 metadata（作者、日期、版本、分类等） | 文档的一个片段（chunk 文本） + 继承的 metadata + chunk 级 metadata |
| **大小** | 可能非常大（数万 token） | 通常数百 token（一个 chunk） |
| **可否直接入库** | 一般不直接入库（太大，语义稀释） | 直接入库（每 Node 生成一个向量） |
| **关系** | 可包含多个 Node（1 对多） | 可属于 1 个 Document；可与兄弟 Node 形成链（PREVIOUS/NEXT） |
| **典型使用场景** | 存储文档全局信息、按类别/日期过滤、高级路由策略 | 向量检索、LLM 上下文组装、多跳推理、递归检索 |
| **类比** | 一本书 | 书中的一章/一节 |

---

### 19.3 Document 详解

#### 19.3.1 Document 的职责

Document 在 LlamaIndex 中承担两个核心角色：

**角色一：元数据容器。** Document 是"文档级信息"的天然载体——作者、创建日期、分类标签、版本号等。这些信息在检索时可用于过滤（"只查 2024 年之后的政策"）、路由（"财务类文档走财务索引"）、溯源。

**角色二：Node 的父级。** 一个 Document 经过 NodeParser 切分后产生多个 Node。每个 Node 自动继承其父 Document 的 metadata。这就意味着你只需在 Document 级别设置一次元数据，所有下游 Node 都会携带——不需要逐个 chunk 手动设置。

#### 19.3.2 Document 到 Node 的解析过程

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 1. 创建 Document —— 只关心文档级别的事
doc = Document(
    text="第一章 工作时间\n第一条 公司实行标准工时制...\n"
         "第二章 请假制度\n第三条 事假需提前1个工作日申请...\n"
         "第四条 年假天数：入职1-5年员工享有5天年假...",
    metadata={
        "title": "考勤管理制度",
        "author": "人力资源部",
        "version": "V3.0",
        "created_date": "2024-03-15",
        "category": "人事政策",
        "department": "HR",
    }
)

# 2. NodeParser 将 Document 解析为多个 Node
#    metadata 会自动继承到每个 Node 上
parser = SentenceSplitter(chunk_size=300, chunk_overlap=50)
nodes = parser.get_nodes_from_documents([doc])

# 每个 Node 自动携带了 Document 的 metadata
for node in nodes:
    print(node.metadata["category"])  # "人事政策" —— 继承自 Document
    print(node.metadata["author"])    # "人力资源部" —— 继承自 Document
```

#### 19.3.3 Document 元数据的实战价值:o:

```
  Document metadata 的应用场景：

  场景 1: 按类别过滤
    用户问 "HR 有什么新政策？"
    → 检索时设置 filter: {"category": "人事政策", "created_date": ">2024-01-01"}
    → 排除了财务、技术等其他类别的文档，只检索 HR 相关的

  场景 2: 按日期限制
    用户问 "最新的报销标准是什么？"
    → 检索时按 created_date 排序，只取最新的 N 个 Node

  场景 3: 路由分发
    用户问的内容涉及 "费用报销"
    → QueryRouter 判断 query 属于 "财务" 类别
    → 只检索 category="财务制度" 的 Document 下的 Node

  场景 4: 溯源展示
    LLM 回答后，前端展示引用来源
    → 从 Node 的 metadata 中取出 title + version + created_date
    → 展示 "来源：《考勤管理制度》V3.0（2024-03-15）"
```

---

### 19.4 Node 详解

#### 19.4.1 Node 是索引的基石

在 LlamaIndex 中，**Index 不是建在 Document 上的，而是建在 Node 上的。**

```
为什么是 Node 而不是 Document？

  Document: "第一章 工作时间...第二章 请假制度...第三章..."
     → 全文 5000 token，包含多个话题（工时、请假、年假、婚假...）
     → 如果直接对整个 Document 做 Embedding：
        "工作时间"的语义 + "请假制度"的语义 + "年假"的语义...
        = 一个被稀释的、"什么都有"的向量
        = 检索 "事假申请" 时，相似度被无关内容大幅稀释

  Node: "事假需提前1个工作日申请，病假应在当日8:30前通知部门主管"
     → 仅 50 token，单一话题
     → 向量精准编码"请假"相关的语义
     → 检索 "事假申请" 时，相似度极高

  这就是为什么必须先切分成 Node，再对 Node 建索引。
  Node 是"最小语义单元"，Document 是"最大组织单元"。
```

#### 19.4.2 Node 是灵活的关键

Node 的设计赋予了 LlamaIndex 远超简单向量检索的能力：:o::o::o::o::o::o:

```
  Node 的灵活性体现在三个层面：

  层面 1: 可组合
    · 检索到 Node 后，可以通过 PREVIOUS/NEXT 关系获取上下文
    · SentenceWindowNodeParser：检索到目标 Node → 自动展开前后各 2 个 Node

  层面 2: 可追溯
    · 通过 SOURCE 关系追溯到父 Document
    · 通过 PARENT 关系追溯到父标题 Node
    · 检索结果 → Node → Document → 元数据 → 溯源

  层面 3: 可多层级
    · Node 本身还可以有子 Node（如 "第二章" Node 下有 "第三条""第四条" 等子 Node）
    · 形成树状结构：Document → 章节 Node → 段落 Node → 句子 Node
    · 支持递归检索 (RecursiveRetriever) 和分层索引 (Hierarchical Index)
```

---

### 19.5 Node 之间的关系 (Relationships):o::o::o:

这是 LlamaIndex 区别于 LangChain 纯 Document 模型的核心特性。Node 之间可以建立多种类型的关系，形成一个**有向关系图**：

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Node 关系全景图                                        ║
║                                                                              ║
║                           Document                                           ║
║                   《考勤管理制度》V3.0                                         ║
║                              │                                               ║
║                    SOURCE (来源关系)                                           ║
║                ┌─────────────┼─────────────┐                                 ║
║                │             │             │                                 ║
║                ▼             ▼             ▼                                 ║
║  ┌──────────────────────────────────────────────────────┐                    ║
║  │ Node[0]           Node[1]           Node[2]          │                    ║
║  │ "第一章 工作时间"   "第二章 请假制度"  "第四章 年假"    │                    ║
║  │ (章节级 Node)      (章节级 Node)      (章节级 Node)   │                    ║
║  │                    │                                 │                    ║
║  │                    │ PARENT/CHILD                    │                    ║
║  │                    │ (父子层级关系)                     │                    ║
║  │                    ▼                                 │                    ║
║  │          ┌──────────────────┐                        │                    ║
║  │          │ Node[1-1]         │                        │                    ║
║  │          │ "第三条 事假需提前 │                        │                    ║
║  │          │  1个工作日申请"    │                        │                    ║
║  │          │ (段落级 Node)     │                        │                    ║
║  │          └────────┬─────────┘                        │                    ║
║  │                   │                                   │                    ║
║  │     PREVIOUS ◄────┼────► NEXT                        │                    ║
║  │     (前后顺序关系)   │      (前后顺序关系)                │                    ║
║  │                   │                                   │                    ║
║  │          ┌────────▼─────────┐                        │                    ║
║  │          │ Node[1-2]         │                        │                    ║
║  │          │ "第四条 病假应在   │                        │                    ║
║  │          │  8:30前通知主管"   │                        │                    ║
║  │          └──────────────────┘                        │                    ║
║  └──────────────────────────────────────────────────────┘                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

#### 19.5.1 五种关系类型详解

| 关系类型 | 方向 | 含义 | 值示例 | 使用场景 |
|----------|------|------|--------|----------|
| **SOURCE** | Node → Document | 表示这个 Node 来源于哪个 Document。是所有 Node 都必须有的关系。 | `SOURCE: Document[0]` | 溯源：LLM 回答后展示"该信息来自《员工手册》"；metadata 继承：从 SOURCE Document 获取全局元数据 |
| **PREVIOUS** | Node → Node | 表示同一 Document 内，这个 Node 的前一个兄弟 Node | `PREVIOUS: Node[1-1]` | 上下文展开：检索到 Node[1-2] 后，通过 PREVIOUS 获取前文 → 补充 LLM 上下文；SentenceWindow 自动展开 |
| **NEXT** | Node → Node | 表示同一 Document 内，这个 Node 的后一个兄弟 Node | `NEXT: Node[1-3]` | 与 PREVIOUS 配对使用。检索到中间 Node 时，前后各取 N 个 Node 组成完整上下文窗口 |
| **PARENT** | Node → Node:o: | 表示这个 Node 的父级 Node（通常是章节标题） | `PARENT: Node[1]` | 层级理解：检索到"第三条..."时，知道它的父标题是"第二章 请假制度"；递归检索：从小 Node 追溯到大 Node 获取更完整上下文；MetadataReplacement：用小 Node 检索、用父 Node 做上下文 |
| **CHILD** | Node → Node | 表示这个 Node 的子级 Node。PARENT 的反向关系。 | `CHILD: [Node[1-1], Node[1-2]]` | 层级索引：从章节 Node 下钻到具体的段落 Node |

#### 19.5.2 各关系的具体使用场景

**SOURCE（来源关系）—— 溯源的基础设施**

```
 场景：用户查询 "事假申请规定"
   检索到 Node[1-1] "事假需提前1个工作日申请"
   通过 SOURCE 关系找到 Document[0] 《考勤管理制度》
   从 Document[0].metadata 中获取 {title, version, created_date, author}
   LLM 回答时标注："根据《考勤管理制度》V3.0（人力资源部，2024-03-15）"

   没有 SOURCE 关系 → 你知道"这段文字说了什么"，但不知道"它来自哪里"
```

**PREVIOUS / NEXT（前后关系）—— 上下文窗口的自动展开**

```
 场景：检索到 Node[1-2] "第四条 病假应在当日8:30前通知部门主管"
   这个 Node 单独看是完整的，但缺少了它所在段落的上下文

   通过 PREVIOUS 取 Node[1-1] "第三条 事假需提前1个工作日申请..."
   通过 NEXT     取 Node[1-3] "第五条 婚假为3天..."

   将 Node[1-1] + Node[1-2] + Node[1-3] 拼接 → LLM 看到的是一个完整的
   "请假制度"章节，包含事假、病假、婚假的完整规定

   LLM 能理解：事假(提前1天)、病假(8:30前通知)、婚假(3天)——三条并列关系
   如果没有 PREVIOUS/NEXT，LLM 只看到病假，不知道事假和婚假的规定
```

**PARENT / CHILD（父子关系）—— 层级检索与递归展开**

```
 场景：双层索引 + 递归检索

   索引结构：
     Level 1 (粗): 章节 Node（如 "第二章 请假制度"）
     Level 2 (细): 段落 Node（如 "第三条...", "第四条...", "第五条..."）

   检索流程：
     Query "请假有哪些类型？" 
       → Level 1 检索 → 命中 Node[1] "第二章 请假制度"（相似度 0.88）
       → 通过 CHILD 关系获取所有子 Node
       → Level 2 检索（在子 Node 内部精确搜索）
       → 找到 Node[1-1], Node[1-2], Node[1-3], Node[1-4]
       → 综合生成回答：事假、病假、婚假、年假四种

   如果没有 PARENT/CHILD → 只能做平铺检索，无法利用文档的层级结构
```

---

### 19.6 实战代码：构建带关系的 Document 与 Node

```python
"""
═══════════════════════════════════════════════════════════════════════════════
    LlamaIndex Document 与 Node 核心概念 —— 完整实战代码
═══════════════════════════════════════════════════════════════════════════════

依赖：pip install llama-index-core llama-index-embeddings-openai

目标：演示 Document → NodeParser → Node → 关系图谱 → Index 的完整流程，
      重点展示 metadata 继承和 Node 之间关系的作用。
"""
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,        # 双层切分
    SentenceWindowNodeParser,      # 带上下文窗口的切分
)
from llama_index.core.schema import NodeRelationship
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 全局配置
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.3)


# ═══════════════════════════════════════════════════════════════
# 第一部分：创建带元数据的 Document
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("  第一部分：创建 Document")
print("=" * 70)

# 三个 Document，每个代表一本文档
doc_attendance = Document(
    text=(
        "第一章 工作时间\n"
        "第一条 公司实行标准工时制，每日工作不超过8小时。\n"
        "第二条 上班时间为上午9:00至下午18:00。\n\n"
        "第二章 请假制度\n"
        "第三条 事假需提前1个工作日申请，经部门主管审批。\n"
        "第四条 病假应在当日8:30前通知部门主管，"
        "连续病假超过2天须提供医院证明。\n"
        "第五条 年假天数：入职1-5年5天，5-10年10天，10年以上15天。\n"
        "第六条 婚假为3天，产假按国家规定执行，陪产假为7天。\n\n"
        "第三章 考勤管理\n"
        "第七条 迟到30分钟内不扣工资，超过1小时按旷工半日处理。"
    ),
    metadata={
        "title": "考勤管理制度",
        "author": "人力资源部",
        "version": "V3.0",
        "created_date": "2024-03-15",
        "category": "人事政策",
        "department": "HR",
        "pages": "12",
    }
)

doc_reimbursement = Document(
    text=(
        "第一章 报销范围\n"
        "第一条 因公产生的交通费、住宿费、餐饮费可按标准报销。\n"
        "第二条 交通费：市内交通实报实销，跨市交通需提前申请。\n\n"
        "第二章 报销标准\n"
        "第三条 住宿标准：一线城市不超过500元/天，二线城市不超过350元/天。\n"
        "第四条 餐饮补贴：出差期间每日80元。\n\n"
        "第三章 报销流程\n"
        "第五条 报销需在费用发生后30日内提交。\n"
        "第六条 单次超过5000元的报销需总经理审批。"
    ),
    metadata={
        "title": "费用报销管理办法",
        "author": "财务部",
        "version": "V2.1",
        "created_date": "2024-01-10",
        "category": "财务制度",
        "department": "Finance",
        "pages": "8",
    }
)

print(f"  创建了 {2} 个 Document")
print(f"  Document[0]: {doc_attendance.metadata['title']}")
print(f"    metadata: {doc_attendance.metadata}")
print(f"    text 长度: {len(doc_attendance.text)} 字符")
print(f"  Document[1]: {doc_reimbursement.metadata['title']}")
print(f"    metadata: {doc_reimbursement.metadata}")


# ═══════════════════════════════════════════════════════════════
# 第二部分：Document → Node 切分（metadata 自动继承）
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  第二部分：Document → Node 切分（metadata 继承）")
print("=" * 70)

# 使用 SentenceSplitter 按句子边界切分
parser = SentenceSplitter(
    chunk_size=200,       # 每个 chunk 最多 200 字符
    chunk_overlap=40,     # 相邻 chunk 重叠 40 字符
    paragraph_separator="\n\n",  # 段落分隔符
)

# get_nodes_from_documents 会自动：
#   1. 切分 text → 多个 chunk
#   2. 为每个 chunk 创建 Node
#   3. 自动继承父 Document 的 metadata
#   4. 自动建立 SOURCE 关系（Node → Document）
#   5. 自动建立 PREVIOUS / NEXT 关系（Node → Node）
nodes = parser.get_nodes_from_documents([doc_attendance])

print(f"  切分结果：1 个 Document → {len(nodes)} 个 Node\n")

# 展示每个 Node 的内容和继承的 metadata
for i, node in enumerate(nodes):
    node_text = node.text[:80].replace("\n", "\\n")
    print(f"  ┌── Node[{i}] ─────────────────────────────")
    print(f"  │ node_id: {node.node_id[:16]}...")
    print(f"  │ text:    {node_text}...")

    # metadata 继承自 Document
    print(f"  │ metadata (继承自 Document):")
    print(f"  │   title:    {node.metadata.get('title')}")
    print(f"  │   category: {node.metadata.get('category')}")
    print(f"  │   author:   {node.metadata.get('author')}")

    # Node 之间的关系
    print(f"  │ relationships:")
    rels = node.relationships

    # SOURCE —— 所有 Node 都有
    source = rels.get(NodeRelationship.SOURCE)
    if source:
        print(f"  │   SOURCE → Document node_id: {source.node_id[:12]}...")

    # PREVIOUS —— 第一个 Node 没有
    prev = rels.get(NodeRelationship.PREVIOUS)
    if prev:
        print(f"  │   PREVIOUS → Node: {prev.node_id[:12]}...")

    # NEXT —— 最后一个 Node 没有
    nxt = rels.get(NodeRelationship.NEXT)
    if nxt:
        print(f"  │   NEXT → Node: {nxt.node_id[:12]}...")

    print(f"  └{'─' * 50}")


# ═══════════════════════════════════════════════════════════════
# 第三部分：双层切分（HierarchicalNodeParser）
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  第三部分：双层切分（章节 Node + 段落 Node）")
print("=" * 70)

# HierarchicalNodeParser 先切大块（章节级），再切小块（段落级）
hierarchical_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 200],  # 第一层 512 字符，第二层 200 字符
    chunk_overlap=30,
)

hierarchical_nodes = hierarchical_parser.get_nodes_from_documents(
    [doc_attendance]
)

print(f"  双层切分结果: {len(hierarchical_nodes)} 个 Node")

# 分层展示
level1_nodes = [n for n in hierarchical_nodes 
                if len(n.text) > 200]     # 第一层（大块）
level2_nodes = [n for n in hierarchical_nodes 
                if len(n.text) <= 200]    # 第二层（小块）

print(f"    第1层（章节级大 Node）: {len(level1_nodes)} 个")
for n in level1_nodes:
    print(f"      [{n.text[:60]}...]")

print(f"    第2层（段落级小 Node）: {len(level2_nodes)} 个")
for n in level2_nodes[:4]:  # 只展示前 4 个
    # 检查 PARENT 关系
    parent = n.relationships.get(NodeRelationship.PARENT)
    parent_info = f"→ PARENT: {parent.node_id[:8]}..." if parent else ""
    print(f"      [{n.text[:60]}...] {parent_info}")


# ═══════════════════════════════════════════════════════════════
# 第四部分：SentenceWindowNodeParser（自动上下文窗口）
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  第四部分：SentenceWindowNodeParser（自动上下文展开）")
print("=" * 70)

# 这个 parser 会在检索时自动利用 PREVIOUS/NEXT 关系展开上下文
window_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,              # 前后各取 3 个 Node
    window_metadata_key="window",  # 上下文窗口存储在 metadata 的 window 字段
    original_text_metadata_key="original_text",
)

window_nodes = window_parser.get_nodes_from_documents(
    [doc_attendance, doc_reimbursement]
)

print(f"  窗口切分结果: {len(window_nodes)} 个 Node")
print(f"  参数: window_size=3（检索到目标后自动展开前后各3个句子）")
print(f"")

# 展示一个 Node 的完整关系链
sample_node = window_nodes[3] if len(window_nodes) > 3 else window_nodes[0]
print(f"  示例 Node[{3}]: {sample_node.text[:100]}...")
print(f"    relationships:")
for rel_type, rel_node in sample_node.relationships.items():
    if rel_node:
        print(f"      {rel_type.name}: {rel_node.node_id[:12]}...")
print(f"    window (上下文窗口字段): "
      f"{'已设置' if 'window' in sample_node.metadata else '未设置'}")


# ═══════════════════════════════════════════════════════════════
# 第五部分：建索引 + 查询验证
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  第五部分：建索引 + 查询验证（metadata 过滤）")
print("=" * 70)

# 用所有 Document 的 Node 建一个统一索引
all_documents = [doc_attendance, doc_reimbursement]
all_nodes = parser.get_nodes_from_documents(all_documents)

index = VectorStoreIndex(all_nodes)

# ── 查询 1: 基础查询 ─────────────────────────────────────
print("\n  [查询 1] 基础查询：事假申请规定")
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("事假需要提前多久申请？")
print(f"  回答: {response}")
print(f"  引用来源:")
for node in response.source_nodes:
    print(f"    [{node.metadata.get('title')}] "
          f"相似度:{node.score:.3f} | {node.text[:60]}...")

# ── 查询 2: 带 metadata 过滤的查询 ────────────────────────
print("\n  [查询 2] 按 category 过滤：只查财务制度")
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# 只检索 category="财务制度" 的 Document 下的 Node
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="财务制度")
    ]
)

filtered_engine = index.as_query_engine(
    similarity_top_k=3,
    filters=filters,
)
response2 = filtered_engine.query("报销标准是什么？")
print(f"  回答: {response2}")
print(f"  引用来源:")
for node in response2.source_nodes:
    print(f"    [{node.metadata.get('title')}] "
          f"category={node.metadata.get('category')} | {node.text[:60]}...")

# ── 验证：确认过滤起了作用 ─────────────────────────────
print("\n  [验证] 检查返回结果是否全部属于 '财务制度':")
all_finance = all(
    n.metadata.get("category") == "财务制度"
    for n in response2.source_nodes
)
print(f"    全部属于财务制度: {all_finance}")


# ═══════════════════════════════════════════════════════════════
# 第六部分：总结 —— Document 与 Node 在实战中的分工
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  总结：Document 与 Node 的分工")
print("=" * 70)

print("""
  Document 的职责：
    ✓ 承载原始文档的完整内容和全局元数据
    ✓ 作为 metadata 的"父容器"——设置一次，所有子 Node 自动继承
    ✓ 支持高级检索策略（按类别/日期/作者过滤、路由分发）
    ✓ 1 个 Document = 1 本手册 / 1 篇论文 / 1 条数据库记录

  Node 的职责：
    ✓ 作为 Index 的构建原材料（Index 建在 Node 上，不是 Document 上）
    ✓ 通过 SOURCE 关系追溯到父 Document（溯源）
    ✓ 通过 PREVIOUS/NEXT 关系实现上下文自动扩展
    ✓ 通过 PARENT/CHILD 关系实现层级检索和递归检索
    ✓ 1 个 Node = 1 个语义片段 = 1 个检索单元

  关系图谱的作用：
    SOURCE    → "这段信息来自哪份文档？"（溯源）
    PREVIOUS  → "前面还有什么相关内容？"（上下文前向扩展）
    NEXT      → "后面还有什么相关内容？"（上下文后向扩展）
    PARENT    → "这段属于哪个章节？"（层级定位）
    CHILD     → "这个章节下有哪些具体内容？"（层级下钻）
""")


---

## 二十、大厂面试题深度讲解

### 20.1 Document 与 Node 核心概念类

#### Q1：LlamaIndex 为什么需要 Document 和 Node 两层数据结构？直接用 Node 不行吗？（字节跳动 / 腾讯 高频）

**面试官考察点：** 这道题考察对"为什么要设计两层抽象"的理解，而不是背概念。回答要触及设计哲学。

**回答思路（从三个问题出发，逐步引出两层结构的必要性）：**

**问题一：如果只有 Document，没有 Node——"粒度太粗"**

```
Document 《考勤管理制度》(5000 token, 10个话题)

直接对 Document 做 Embedding → 一个向量编码了 10 个不同话题的语义
→ 用户搜"事假申请"时，这个向量的相似度是：
  "事假"(5%) + "年假"(5%) + "婚假"(5%) + "工作时间"(40%) + "考勤"(20%) + ...
  = 高度稀释的语义信号

→ 用户搜"事假申请"可能搜不到这篇文章，因为向量被其他话题"淹没"了。
  Document 太大 → 语义不聚焦 → 检索精度差 → 必须切分为更细的单元
```

**问题二：如果只有 Node，没有 Document——"丢失了全局视角"**

```
Node[1]: "事假需提前1个工作日申请，经部门主管审批。"
Node[2]: "病假应在当日8:30前通知部门主管。"
Node[3]: "年假天数：入职1-5年5天。"

没有 Document 时，这三个 Node 只是三个"孤立的文本片段"。
你不知道：
  - 它们都来自《考勤管理制度》——失去了"同一份文档"的全局关联
  - 它们的 metadata 需要各自维护——如果作者从"人力资源部"改成"HR部门"，10 个 Node 要改 10 次
  - 它们属于"人事政策"类别——无法在检索时按 category 统一过滤

→ 没有 Document → metadata 管理成本爆炸 + 全局过滤不可行 + 溯源信息丢失
```

**问题三：两层结构解决了什么——"关注点分离"**

```
Document 层 → 负责"全局"：文档来源、元数据、版本、分类、全文
  → 1 份文档的全局信息只存 1 次
  → 修改 metadata 只需要改 Document

Node 层 → 负责"局部"：语义片段、向量匹配、检索返回
  → 每个 Node 编码一个集中的语义信号
  → Node 自动从 Document 继承 metadata（不需要手动复制）
  → Node 之间通过 Relationships 形成关系图

两层结构的本质：把"全局管理"和"局部检索"解耦。
```

**面试官追问："这和 LangChain 的 Document 模型有什么本质区别？"**

"LangChain 的 Document 是 `{page_content, metadata}`——它是一个扁平的袋子。切分后的 chunk 也是 Document 类型，chunk 之间没有任何关系。

LlamaIndex 的 Document 是"父容器"——负责元数据的集中管理。切分后的 Node 是 Document 的"子单元"——继承了 metadata、建立了 SOURCE/PREVIOUS/NEXT/PARENT/CHILD 关系网。这个关系网是 SentenceWindowNodeParser、AutoMergingRetriever、递归检索等所有高级功能的数据基础。

简单说：LangChain 的 Document 之间是孤岛，LlamaIndex 的 Node 之间是网络。"

---

#### Q2：metadata 从 Document 到 Node 是如何继承的？为什么这个继承机制很重要？（阿里巴巴 / 百度）

**面试官考察点：** 考察对 metadata 管理机制的细节理解。metadata 继承是生产系统可维护性的关键。

**回答思路（先说机制，再说价值，最后说注意事项）：**

**metadata 继承的机制：**

```
NodeParser.get_nodes_from_documents([doc]) 的内部逻辑：

1. 切分 doc.text → ["chunk1文本", "chunk2文本", ...]

2. 为每个 chunk 创建 Node：
   for chunk_text in chunks:
       node = Node(
           text=chunk_text,
           metadata=doc.metadata.copy(),  # ← 关键：浅拷贝 Document 的 metadata
       )

3. 建立 Relationship：
   node.relationships[SOURCE] = doc.doc_id
   node.relationships[PREVIOUS] = previous_node.node_id
   node.relationships[NEXT] = next_node.node_id

4. 可选：追加 Node 级别的 metadata
   node.metadata["chunk_index"] = i
   node.metadata["chunk_count"] = len(chunks)
```

**继承机制为什么重要——三个实战场景：**

**场景一——一处修改，全局生效：**

```
没有继承机制 → 10 个 Node 各自维护 metadata
  → Document 的 author 从 "人力资源部" 改为 "HR 管理中心"
  → 需要遍历 10 个 Node，各自修改 metadata["author"]
  → 极易遗漏，导致同一份文档的不同 chunk 有不同的 author

有继承机制 → Document.metadata["author"] 更新
  → 重新解析：node.metadata["author"] 自动更新
  → 或通过 SOURCE 关系动态读取 Document 的最新 metadata
  → 一处修改，全局一致
```

**场景二——按文档级属性做全局过滤：**

```python
# 用户问："HR 部门的 2024 年以后的新政策？"
# → 需要同时用 category 和 created_date 过滤

filters = MetadataFilters(filters=[
    MetadataFilter(key="category", value="人事政策"),
    MetadataFilter(key="created_date", value="2024-01-01", operator=">="),
])

# 如果没有 metadata 继承：
#   → 每个 Node 都需要各自设置 category 和 created_date
#   → 100 份 Document → 1000 个 Node → 1000 次手动设置 metadata
#   → 几乎不可能在生产环境维护

# 有 metadata 继承：
#   → Document 级别设置一次 → 所有子 Node 自动携带
#   → 过滤时每个 Node 都有 category 和 created_date → 直接可过滤
```

**场景三——溯源展示：**

```
LLM 回答后需要展示"该信息来自《考勤管理制度》V3.0（人力资源部，2024-03-15）"
→ 这些信息全部来自 Node 继承的 Document metadata
→ title + version + author + created_date → 前端展示的溯源卡片
```

**面试官追问："metadata 继承是用浅拷贝（shallow copy）还是深拷贝（deep copy）？这有什么影响？"**

"LlamaIndex 用的是 `doc.metadata.copy()`——浅拷贝。这意味着：
- 基础类型（str、int、date）的值被复制了，修改 Document 的 metadata 不会自动更新已生成的 Node → 这是合理的，因为已经入库的 Node 应该是快照
- 如果 metadata 中有嵌套对象（list、dict），浅拷贝只拷贝引用，Document 修改嵌套对象的值会影响已生成的 Node → 这是个潜在坑，所以要避免在 metadata 中放可变对象

最佳实践：只放基础类型（str/int/float/bool）在 metadata 中。如果需要嵌套结构，放在专门的字段中（如 `_raw`），并明确文档说明这是可变的。"

---

### 20.2 Node 关系类

#### Q3：LlamaIndex 中 Node 的五种 Relationships 分别是什么？各自支持什么高级功能？（字节跳动 / 腾讯 高频）

**面试官考察点：** 这是 LlamaIndex 的核心差异化特性。要讲清楚每种关系的"为什么存在"而非"是什么"。

**回答框架（按从基础到高级的顺序讲）：**

**关系一：SOURCE（Node → Document）—— 最基础，所有 Node 都有**

```
作用：溯源
  Node[5] 被检索命中 → 通过 SOURCE 找到 Document[0]《考勤管理制度》
  → 从 Document[0].metadata 获取 title/version/author
  → 展示："来源：《考勤管理制度》V3.0"

没有 SOURCE：你知道"这段文字说了什么"，不知道"它来自哪份文档"
```

**关系二+三：PREVIOUS / NEXT（Node ↔ Node）—— 上下文扩展**:o:

```
作用：SentenceWindowNodeParser 自动上下文展开

检索命中 Node[5] "病假应在当日8:30前通知部门主管"
→ 自动展开：
  PREVIOUS → Node[4] "事假需提前1个工作日申请"
  NEXT     → Node[6] "婚假为3天"

LLM 收到的不是孤立的 Node[5]，而是 Node[4]+Node[5]+Node[6] 的完整上下文
→ 能理解"事假、病假、婚假"的并列关系和各自规则

关键价值：解决"检索粒度"和"生成粒度"的矛盾
  - 检索用小粒度 Node（200 token，匹配精准）
  - 生成时通过 PREVIOUS/NEXT 自动扩展为 600 token 上下文
  - 不需要手工管理 overlap！
```

**关系四+五：PARENT / CHILD（Node ↔ Node）—— 层级检索**:o:

```
作用一：AutoMergingRetriever 自动合并
  多个子 Node（如 Node[4], Node[5], Node[6]）都来自同一个父 Node
  → 命中比例 > 50% → 自动合并为父 Node
  → LLM 看到的是完整章节而非分散碎片

作用二：MetadataReplacementPostProcessor
  检索用子 Node（高精度）→ 返回时替换为父 Node（完整上下文）
  → 子 Node 的 metadata 被父 Node 的 metadata 替换
  → 用户看到的溯源信息是"该信息来自第二章 请假制度"而非"该信息来自第372段"

作用三：递归检索（RecursiveRetriever）
  Query 在第1层（章节级 Node）检索
  → 命中 "第二章 请假制度"
  → 通过 CHILD 关系下钻到第2层（段落级 Node）
  → 在第2层做精确检索
  → 找到最相关的具体条款
```

**五种关系的能力矩阵：**

| 关系 | 支持的 LlamaIndex 功能 | 解决的问题 |
|------|----------------------|-----------|
| **SOURCE** | 溯源、metadata 过滤 | "这段信息来自哪？属于什么类别？" |
| **PREVIOUS/NEXT** | SentenceWindowNodeParser | "检索到这段，但前后文是什么？" |
| **PARENT/CHILD** | AutoMergingRetriever、MetadataReplacement、RecursiveRetriever | "这段属于哪个章节？如何获取完整上下文？" |

**面试官追问："这些关系是存储在 Node 对象的哪个字段里？检索时会随 Node 一起返回吗？"**

"关系存储在 `node.relationships` 字典中，是一个 `Dict[NodeRelationship, RelatedNodeInfo]` 类型。检索返回 `NodeWithScore` 时会携带 relationships。例如 `SentenceWindowNodeParser` 就是在检索结果返回后，遍历每个命中 Node 的 `relationships[PREVIOUS]` 和 `relationships[NEXT]`，递归获取前后的 Node 文本，拼接到最终返回的上下文中。这一操作发生在检索完成之后、LLM 生成之前。"

---

#### Q4：PREVIOUS/NEXT 和 PARENT/CHILD 都是 Node 间的关系，它们的本质区别是什么？什么时候用哪种？（阿里巴巴 / 美团）

**面试官考察点：** 考察对两种关系维度的理解——横向 vs 纵向。

**回答思路——关键的区分点在"关系维度"：**

**维度一：一个是横向的"时间线"，一个是纵向的"层级树"**

```
PREVIOUS/NEXT = 横向的"链表"
  
  Node[0] ⇄ Node[1] ⇄ Node[2] ⇄ Node[3] ⇄ Node[4]
  
  关系维度：顺序关系。描述了"同一层级下，文本的先后顺序"。
  类比：一本书翻页的"前一页/后一页"关系。

PARENT/CHILD = 纵向的"树"
  
           Parent Node（章节标题）
              │
    ┌─────────┼─────────┐
    │         │         │
  Child[0]  Child[1]  Child[2]（具体条款）
  
  关系维度：层级关系。描述了"从概括到具体"的包含关系。
  类比：书的"章/节/条"目录层级关系。
```

**维度二：使用场景完全不同**

```
PREVIOUS/NEXT 的使用场景：
  → 你想让 LLM 看到"一段文字的上下文"
  → 不改变检索的层级，只是横向扩展阅读范围
  → 典型工具：SentenceWindowNodeParser
  → 问自己："LLM 读这一段时，需要看它前后说了什么吗？"

PARENT/CHILD 的使用场景：
  → 你想从"模糊的章节标题"定位到"精确的条款内容"
  → 改变检索的层级——从上到下或从下到上
  → 典型工具：AutoMergingRetriever、RecursiveRetriever
  → 问自己："检索到了章节标题，需要下钻到具体条款吗？"
        或 "检索到了太多碎片，能合并为完整章节吗？"
```

**维度三：一个具体的比较案例**

```
场景：知识库有 1000 个 Node，用户问"请假有哪些类型？"

方案 A（只用 PREVIOUS/NEXT，不用 PARENT/CHILD）：
  ① 检索 Node 层面 → 命中 Node[200] "第二章 请假制度 第三条 事假..."
  ② 通过 PREVIOUS 向前展开 2 个 Node → 多拿到一些上下文
  ③ 通过 NEXT 向后展开 2 个 Node → 多拿到一些上下文
  ④ LLM 看到了约 5 个相邻 Node，可能覆盖了 3 种请假类型
  → 问题：如果请假类型分布在 10 个相邻 Node 之间，窗口不够大就会漏

方案 B（使用 PARENT/CHILD 层级结构）：
  ① Level 1（章节 Node）检索 → "第二章 请假制度" = 父 Node
  ② 通过 CHILD 关系直接获取该章下所有子 Node
  ③ LLM 看到了"第二章"的全部内容 → 所有请假类型 100% 覆盖
  → 优势：不需要猜测"窗口要拉多大"，层级关系本身定义了边界
```

**选择决策：**

```
用 PREVIOUS/NEXT 当：
  ✓ 文档没有明确的层级结构（纯文本）
  ✓ 需要的是"局部连贯性"而非"全局完整性"
  ✓ 不确定需要多少上下文，希望灵活调整窗口大小

用 PARENT/CHILD 当：
  ✓ 文档有明确的层级结构（Markdown 标题、法律条款、教科书）
  ✓ 需要的是"章节级别的完整性"
  ✓ 希望检索后自动合并为"完整的逻辑单元"

两者不互斥——可以同时使用：
  SentenceWindowNodeParser（PREVIOUS/NEXT 横向扩展）
  + MetadataReplacementPostProcessor（PARENT/CHILD 纵向替换）
  = 先用层级定位到正确的章节，再用窗口扩展保证局部连贯
```

---

### 20.3 索引与检索机制类

#### Q5：为什么 LlamaIndex 的 Index 建在 Node 上而不是 Document 上？如果直接对 Document 做 Embedding 会怎样？（腾讯 / 百度）

**面试官考察点：** 是否理解"为什么必须先切分再索引"的底层原因。这是 RAG 的基石认知。

**回答思路（用数值对比讲清楚语义稀释效应）：**

**直接对 Document 做 Embedding——语义稀释效应：**

```
假设一个 Document 包含 5 个不同的话题：
  Document.text = 
    "第一章 工作时间（500字符）...
     第二章 请假制度（600字符）...
     第三章 考勤管理（400字符）...
     第四章 薪酬福利（800字符）...
     第五章 培训发展（300字符）..."

总长度：2600 字符 ≈ 1200 token

Embedding 后生成的向量：
  = 平均池化(第一章的语义 + 第二章的语义 + ... + 第五章的语义)
  = 一个"什么都有、什么都不精准"的向量

检索时：
  用户问精确问题 "事假提前几天申请？"
  
  Document 向量 vs Query 向量 → 相似度 0.52
  → 因为 Document 向量中，"事假"的语义权重只有约 12%（1/5 个章节 + 大部分内容在讲工作时间/薪酬/培训）
  
  Node 向量（来自 "第三条 事假需提前1个工作日申请"） vs Query 向量 → 相似度 0.91
  → Node 向量中，"事假"是核心语义，权重占 80%+
  
  差距：0.91 vs 0.52 → 如果 Top-K=5，Document 级别的检索可能根本排不进前 5
```

**用向量空间的视角理解：**

```
文档 → Embedding：
  
  文档 A 的向量：位于向量空间中"人事政策"、"工作时间"、"薪酬"、"培训"、"考勤"的几何中心
  → 它不和任何一个具体话题接近，它是所有话题的"平均值"
  
  文档 B 的向量：位于"财务报销"、"差旅标准"、"审批流程"的几何中心
  
  用户 query "事假申请" 的向量：位于"请假制度"附近
  
  距离对比：
    query ↔ Node[事假段落] = 很近（语义匹配）→ 相似度 0.91
    query ↔ Document A = 中等距离 → 相似度 0.52
    query ↔ Document B = 很远 → 相似度 0.31
    
  如果 Top-K=3，排序是：Node[事假] > Document A > Document B
  → Node 排第一 ✓
  
  如果 Top-K=3 且只用 Document（没有 Node）：Document A > Document B > 其他不相关文档
  → 虽然 A 排第一，但相似度只有 0.52，LLM 收到的是一份 2600 字符的全文
  → 其中只有 600 字符（23%）是相关的，其余是噪声

  如果检索返回多个 Document（Top-K=5），5 个 Document 加起来可能 10000+ token
  → 大量 token 浪费在"看起来有点相关其实不相关"的内容上
```

**面试官追问："那如果我的文档很短（< 500 token），还需要切分吗？"**

"不需要。切分的目的是解决"文档太大导致语义稀释"。如果文档本身就是一个短 FAQ、一条聊天记录、一条法律条款——它自身就是一个完美的语义单元，切分反而可能破坏它的完整性。LlamaIndex 的 NodeParser 也不会对短文本做无意义的切分——`SentenceSplitter` 会在文本长度 < chunk_size 时保持原样。所以正确的说法是：**对语义完整的短文档不做切分，对语义混杂的长文档必须切分。判断标准不是文档数量，而是单个文档的话题集中度。**"

---

### 20.4 实战与调试类

#### Q6：HierarchicalNodeParser 和 SentenceWindowNodeParser 有什么区别？分别在什么场景下使用？（美团 / 拼多多）

**面试官考察点：** 是否理解这两种 Parser 的定位差异——一个做层级、一个做窗口。

**回答思路（先各自讲清原理，再对比，最后给选择建议）：**

**HierarchicalNodeParser——"组织知识的树形结构"：**

```
工作原理：
  Input: Document
  → 第一轮切分：chunk_size=512 → 产生"章节级 Node"（Level 1）
  → 第二轮切分：对每个 Level 1 Node 内部用 chunk_size=200 再切 → 产生"段落级 Node"（Level 2）
  
  结果：
    Level 1 Node[0] = "第二章 请假制度\n第三条 事假需提前..." (512t)
      ├─ PARENT/CHILD 关系
      ├─ Level 2 Node[0-0] = "第三条 事假需提前1个工作日申请" (200t)
      ├─ Level 2 Node[0-1] = "第四条 病假应在8:30前通知主管" (200t)
      └─ Level 2 Node[0-2] = "第五条 年假天数：入职1-5年5天" (200t)
  
  检索策略：
    先在 Level 2（小 Node）检索 → 命中 Node[0-1]
    → 通过 PARENT 关系获取 Level 1 Node[0]
    → 将 Level 1 Node[0]（完整章节）作为上下文注入 LLM
  
  核心思想：用层级结构保证章节级完整性
```

**SentenceWindowNodeParser——"给每个片段加个窗"：**

```
工作原理：
  Input: Document
  → 按句子边界切分为最小粒度的 Node（每 1-2 句一个 Node）
  
  结果：
    Node[0] = "第一章 工作时间" (单句)
    Node[1] = "第一条 公司实行标准工时制，每日工作不超过8小时。" (单句)
    Node[2] = "第二条 上班时间为上午9:00至下午18:00。" (单句)
    Node[3] = "第二章 请假制度" (单句)
    Node[4] = "第三条 事假需提前1个工作日申请，经部门主管审批。" (单句)
    Node[5] = "第四条 病假应在当日8:30前通知部门主管。" (单句)
    ...
    （所有 Node 之间有 PREVIOUS/NEXT 关系）
  
  检索策略：
    检索命中 Node[4] "事假需提前..."
    → 通过 PREVIOUS 取 Node[3], Node[2], Node[1]
    → 通过 NEXT 取 Node[5], Node[6], Node[7]
    → 将 Node[1]~Node[7] 拼接为 7 个句子的窗口 → 注入 LLM
  
  核心思想：用滑动窗口保证上下文连贯性
```

**核心区别对比：**

| 维度 | HierarchicalNodeParser | SentenceWindowNodeParser |
|------|------------------------|--------------------------|
| **切分层数** | 2 层（章节级 + 段落级） | 1 层（句子级） |
| **关系类型** | 使用 PARENT/CHILD | 使用 PREVIOUS/NEXT |
| **上下文获取方式** | 纵向：子 Node → 父 Node | 横向：命中 Node → 前后 N 个 Node |
| **上下文边界** | 由文档的层级结构决定（章节边界） | 由窗口大小参数决定（固定 N 个句子） |
| **适用文档** | 有明确层级结构（Markdown/法律/教科书） | 无明确结构（散文/聊天记录/纯文本） |
| **上下文完整度** | 保证章节级完整 | 取决于窗口大小+文档特征 |
| **灵活性** | 低（层级结构固定） | 高（窗口大小可调） |

**选择建议：**

```
用 HierarchicalNodeParser 当：
  ✓ 文档有 # ## ### 标题层级
  ✓ 法律条文有"第X章 第X条"结构
  ✓ 教科书/手册的章节结构清晰
  → 好处：PARENT/CHILD 关系天然定义了"完整语义单元"的边界
           不需要猜测"窗口该拉多大"

用 SentenceWindowNodeParser 当：
  ✓ 文档没有明确层级（散文、聊天记录、非结构化 PDF）
  ✓ 需要灵活控制上下文窗口大小
  ✓ 追求最高检索精度（句子级粒度）
  → 好处：检索时用单句（最高精度），生成时展开为窗口（保证上下文）
```

---

#### Q7：用 LlamaIndex 的 metadata filter 做检索过滤时，底层是怎么工作的？有什么坑？（字节跳动 / 华为）:o::o::O::O:o::o::o:

**面试官考察点：** metadata 过滤是生产环境的刚需。考察是否理解过滤的底层机制和性能影响。

**回答思路（先讲原理，再讲坑）：**

**metadata filter 的底层工作流：**

```
用户设置 filters：
  MetadataFilters(filters=[
      MetadataFilter(key="category", value="人事政策"),
      MetadataFilter(key="created_date", value="2024-01-01", operator=">="),
  ])

实际执行（取决于向量数据库）：

方案 A：前置过滤（Pre-filtering）
  ① 先用 metadata 条件过滤所有 Node → 候选集从 10000 缩到 500
  ② 在 500 个候选 Node 中做向量检索 → Top-5
  优点：检索范围缩小，精度更高
  缺点：如果过滤后候选太少（< Top-K），可能漏掉相关内容

方案 B：后置过滤（Post-filtering）
  ① 先做向量检索 → 从 10000 中取 Top-100
  ② 在 Top-100 中按 metadata 过滤 → 得到符合条件的 Top-5
  优点：不会因为过滤条件太严而漏召回
  缺点：如果过滤条件过滤掉了 Top-100 中的大部分内容，实际返回可能 < 5

方案 C：混合过滤（推荐）
  ① 向量检索 + metadata 过滤同时进行（支持向量索引+标量索引的库）
  → Milvus/Qdrant/Weaviate 原生支持
  → 速度快 + 精度高
```

**四个常见坑：**

**坑一——metadata 字段不是索引字段，过滤会退化为全表扫描：**:o:

```
问题：metadata 中的 "department" 字段没有建标量索引
→ 向量数据库做过滤时需要遍历所有 Node 的 metadata
→ 100 万 Node → 全表扫描 → 延迟从 10ms 飙升到 500ms+

解决：确保常用过滤字段在向量数据库中建了标量索引
  Qdrant: payload 中的 indexed 字段
  Milvus: schema 中的 scalar index
  Weaviate: 自动为所有属性建倒排索引
```

**坑二——metadata 类型不一致导致过滤失败：**

```
问题：Document 的 metadata["created_date"] = "2024-03-15"（字符串）
      过滤条件 operator=">=" 要求数值或日期类型
      → 字符串比较 "2024-03-15" >= "2024-01-01" 虽然碰巧能工作，
        但 "2024-11-01" >= "2024-01-01" 也是 True（字符串的字典序不一定等于日期序）
      → 更糟的是跨年时："2025-01-01" >= "2024-12-31" = False（字符串比较）

解决：存储日期时用 ISO 8601 格式 "2024-03-15"
      或转为 Unix timestamp（整数），过滤时用数值比较
```

**坑三——过滤后结果为空时的兜底缺失：**

```
问题：用户问 "2025 年的新政策有哪些？"
      filters: created_date >= 2025-01-01
      但知识库中还没有 2025 年的文档 → 过滤后候选集为空
      → 返回空结果 → 用户看到 "未找到相关信息"
      → 但可能知识库中有 2024 年底的"即将生效的 2025 年政策"

解决：过滤条件分级
  第一级：严格过滤（category=人事政策 AND created_date>=2025）
  第二级（兜底）：放宽过滤（category=人事政策，按 created_date 倒序）
  在 Prompt 中明确告诉 LLM："以下是最接近的信息（日期不完全匹配），请据此作答"
```

**坑四——metadata 字段太多，手动设置成本高且易出错：**

```
问题：团队有 3 个人在维护 Document metadata
  → A 用 "create_date"，B 用 "created_at"，C 用 "date"
  → 过滤时不知道该用哪个字段名

解决：约定 metadata schema（类似数据库 schema）
  {
    "source": str,      # 文档来源（必填）
    "title": str,       # 文档标题（必填）
    "category": str,    # 分类标签（必填，从枚举中选）
    "created_date": str, # ISO 8601 格式（必填）
    "version": str,     # 版本号（选填）
    "department": str,  # 部门（选填）
  }
  → 在代码中用 Pydantic/ dataclass 校验 metadata 结构
  → CI 中检查是否有非标准字段
```

---

### 20.5 面试高频知识点速查

#### 一句话答案系列

| 问题 | 一句话答案 |
|------|-----------|
| Document 和 Node 的关系？ | Document 是"父容器"（管理全局元数据），Node 是"子单元"（索引和检索的最小粒度），1 个 Document → N 个 Node |
| 为什么 Index 建在 Node 上？ | Node 粒度小、话题聚焦，向量语义信号强；Document 太大，语义被稀释，检索精度差 |
| metadata 如何从 Document 到 Node？ | NodeParser 创建 Node 时 `node.metadata = doc.metadata.copy()`（浅拷贝），自动继承 |
| 五种 Relationships 分别解决什么？ | SOURCE=溯源、PREVIOUS/NEXT=上下文扩展、PARENT/CHILD=层级检索 |
| PREVIOUS/NEXT 的最佳应用？ | SentenceWindowNodeParser——检索命中后自动展开前后 N 个句子的上下文窗口 |
| PARENT/CHILD 的最佳应用？ | AutoMergingRetriever——子节点命中比例 > 阈值时自动合并为父节点 |
| HierarchicalNodeParser 怎么用？ | 文档有 `#`/`##` 层级结构时，先切章节级大块再切段落级小块，用 PARENT/CHILD 做层级检索 |
| metadata filter 的坑？ | 字段没建标量索引 → 全表扫描；类型不一致 → 过滤条件失效；结果为空 → 缺兜底策略 |

#### 必知数据结构速查

| 概念 | LlamaIndex 实现 | LangChain 实现 |
|------|----------------|---------------|
| **文档层** | `Document(text, metadata)` — 父容器 | `Document(page_content, metadata)` — 扁平结构 |
| **检索单元** | `Node(text, metadata, relationships)` — 有向关系图 | `Document` — 切分后的 chunk 也是 Document，无关系 |
| **元数据继承** | 自动：`Node.metadata = doc.metadata.copy()` | 手动：需在切分逻辑中自己实现 |
| **节点关系** | `SOURCE/PREVIOUS/NEXT/PARENT/CHILD` 五种关系 | 无内置关系，需自行维护 |
| **关系使用** | SentenceWindow/AutoMerging/RecursiveRetriever | 无对应功能 |
| **层级切分** | `HierarchicalNodeParser` 原生支持 | 需手动实现 |

---



