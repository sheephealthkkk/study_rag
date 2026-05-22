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

#### 19.3.3 Document 元数据的实战价值

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

Node 的设计赋予了 LlamaIndex 远超简单向量检索的能力：

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

### 19.5 Node 之间的关系 (Relationships)

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
| **PARENT** | Node → Node | 表示这个 Node 的父级 Node（通常是章节标题） | `PARENT: Node[1]` | 层级理解：检索到"第三条..."时，知道它的父标题是"第二章 请假制度"；递归检索：从小 Node 追溯到大 Node 获取更完整上下文；MetadataReplacement：用小 Node 检索、用父 Node 做上下文 |
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



---

