## 十六、实战用法指南：Unstructured × LlamaIndex 的三种集成模式

前一章我们讲了"为什么 Unstructured 和 LlamaIndex 各司其职"。这一章聚焦"怎么用"——在生产环境中，如何在两者之间找到最佳平衡点。

核心结论先放在前面：

> **大部分文档直接用 LlamaIndex 的 `UnstructuredReader`，少数复杂文档回退到原生 Unstructured + 自定义逻辑。不存在"哪个更好"，只有"在什么场景下哪个更合适"。**

---

### 16.1 模式一：LlamaIndex 封装 UnstructuredReader（快速标准模式）

#### 16.1.1 适用场景

- 中小规模知识库（几百到几千份文档）
- 文档格式标准（Word 导出 PDF、标准 Markdown、网页转存 HTML）
- 不需要深度定制解析逻辑
- 快速搭建原型或 MVP
- 团队不想维护两套解析代码

#### 16.1.2 代码实现

```python
"""
模式一：LlamaIndex + UnstructuredReader（快速标准模式）

优点：
  · 自动将 Elements 转换为 LlamaIndex 的 Document 对象，无需手动映射
  · 无需显式调用 partition()，一行 load_data() 搞定
  · 与 LlamaIndex 后续管线（索引构建、检索）零摩擦对接
  · 参数通过 unstructured_kwargs 透传，常用配置够用

缺点：
  · 自定义化程度有限——无法插入解析前/后的清洗逻辑
  · Element → Document 的中间转换过程不可控（LlamaIndex 内部完成）
  · 复杂的元数据映射需求（如保留坐标做前端高亮）难以实现
  · 无法对特定 Element 类型做特殊处理（如对 Table 元素做专门清洗）
"""

from llama_index.readers.unstructured import UnstructuredReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ─── 全局配置 ──────────────────────────────────────────────────
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.3)

# ─── 一步完成解析 + 转换 ─────────────────────────────────────
reader = UnstructuredReader()
documents = reader.load_data(
    # 可以传 file（单文件）或 file_path
    file="员工手册.pdf",

    # ── 以下参数会透传给 Unstructured 的 partition() ─────────
    unstructured_kwargs={

        # 策略：hi_res 处理扫描件/复杂布局，fast 处理数字原生 PDF
        "strategy": "auto",              # auto = 自动判断

        # OCR 语言（中文 + 英文混合文档）
        "languages": ["chi_sim", "eng"],

        # 表格结构检测——自动还原为结构化表格
        "pdf_infer_table_structure": True,

        # 保留页码等基础元数据
        "include_page_breaks": True,
    }
)

# ─── 直接建索引 ──────────────────────────────────────────────
# 返回的 documents 是标准的 LlamaIndex Document 列表，无需任何转换
index = VectorStoreIndex.from_documents(documents)

# ─── 查询 ────────────────────────────────────────────────────
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("请事假需要提前多久申请？")
print(f"回答: {response}")
# 附带引用来源
for node in response.source_nodes:
    print(f"  来源: {node.metadata.get('file_name', '?')} "
          f"第{node.metadata.get('page_number', '?')}页 "
          f"相似度: {node.score:.3f}")
```

#### 16.1.3 模式一的局限：什么场景它不够用

```
场景 1：PDF 页眉页脚污染严重，Unstructured 未完全清除
  UnstructuredReader 无法插入"解析后 → 建索引前"的清洗步骤
  → 脏数据直接进入索引，检索质量下降

场景 2：需要只保留特定类型的内容
  例如"只保留 Table 和 NarrativeText，丢弃 Header/Footer"
  UnstructuredReader 内部完成了 Element→Document 转换，类型信息已丢失

场景 3：需要精细的元数据映射
  例如前端需要坐标信息来做 PDF 高亮定位
  UnstructuredReader 的转换过程中丢弃了 Element 的坐标元数据

场景 4：不同 Element 类型需要不同的后续处理
  例如 Table 保留原样、NarrativeText 再切分、ListItem 聚合
  统一转换后所有 Element 都变成了同质的 Document
```

---

### 16.2 模式二：原生 Unstructured + 自定义逻辑 + 手动接入 LlamaIndex（高可控模式）

#### 16.2.1 适用场景

- 生产级 RAG 系统，对解析质量要求高
- 文档格式复杂（含扫描件、复杂表格、多栏排版）
- 需要在解析前/后插入自定义清洗逻辑
- 需要保留 Element 的完整元数据（坐标、层级、字体）
- 需要对不同 Element 类型做差异化处理

#### 16.2.2 代码实现

```python
"""
模式二：原生 Unstructured + 自定义逻辑 + 手动接入 LlamaIndex（高可控模式）

优点：
  · 完全自由控制解析参数和流程
  · 可以在 partition() 前后插入任意清洗/过滤/增强逻辑
  · 保留 Element 的全部元数据，可做精细映射
  · 针对不同 Element 类型做差异化处理（Table 保留结构、NarrativeText 再切分等）
  · 易于扩展——后续加新的清洗规则无需改动 LlamaIndex 层

缺点：
  · 代码量多（需要写 Element→Document 的转换逻辑）
  · 需要理解 Unstructured 的 Element 类型体系
  · 维护成本高——Unstructured 版本升级时 Element 类型可能变化
"""

from unstructured.partition.auto import partition
from llama_index.core import Document as LlamaDocument
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
import re
from typing import List, Optional
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# 步骤 1：用原生 Unstructured 解析文档
# ═══════════════════════════════════════════════════════════════

def parse_document(file_path: str, strategy: str = "auto",
                   languages: List[str] = None,
                   infer_table: bool = True):
    """
    解析单个文档，返回 Unstructured 的 Element 列表。

    这里你可以完全控制 partition() 的所有参数，
    包括一些 UnstructuredReader 不暴露的高级参数。
    """
    if languages is None:
        languages = ["chi_sim", "eng"]

    elements = partition(
        filename=file_path,
        strategy=strategy,
        languages=languages,
        # ── PDF 高级参数（UnstructuredReader 也支持但被封装了）──
        pdf_infer_table_structure=infer_table,
        extract_images_in_pdf=False,        # 是否提取嵌入图片
        # ── 元数据 ──
        include_metadata=True,               # 保留全部元数据
        unique_element_ids=True,             # 每个 Element 生成唯一 UUID
        include_page_breaks=True,            # 保留页面边界标记
    )
    return elements


# ═══════════════════════════════════════════════════════════════
# 步骤 2：自定义清洗逻辑（这是模式二的核心价值所在）
# ═══════════════════════════════════════════════════════════════

def clean_elements(elements: List, 
                   min_text_length: int = 10,
                   remove_headers_footers: bool = True,
                   keep_types: Optional[List[str]] = None
                   ) -> List:
    """
    对 Element 列表做自定义清洗。

    这是模式二相对于模式一最大的优势——你可以在解析后、
    建索引前插入任意的数据清洗逻辑，而不受框架封装的限制。

    清洗包括：
      1. 过滤：移除太短的文本、空白段落、特定类型
      2. 去噪：移除页眉页脚、页码文字、版权声明
      3. 修复：修正 OCR 常见错误、合并被错误拆分的段落
      4. 增强：补充上下文信息到 metadata
    """

    # ── 清洗 1：类型过滤 ────────────────────────────────────
    if keep_types:
        elements = [el for el in elements 
                    if type(el).__name__ in keep_types]

    # ── 清洗 2：长度过滤（移除过短的无意义片段）─────────────
    elements = [el for el in elements 
                if len(el.text.strip()) >= min_text_length]

    # ── 清洗 3：重复模式检测（页眉页脚处理）───────────────
    if remove_headers_footers and len(elements) > 3:
        # 统计所有 Element 文本在多页中的重复频率
        text_counter = Counter(el.text.strip() for el in elements)

        # 如果某个文本在超过 60% 的页面中都出现，很可能是页眉/页脚
        page_count = len(set(
            el.metadata.page_number for el in elements 
            if el.metadata.page_number
        ))
        threshold = max(page_count * 0.6, 2)  # 至少出现 2 次

        elements = [
            el for el in elements
            if text_counter[el.text.strip()] < threshold
        ]

    # ── 清洗 4：OCR 常见错误修正（示例）────────────────────
    # OCR 常把 "已" 识别成 "己"，把 "日" 识别成 "曰"
    # 具体规则取决于你的文档特点，这里仅示例
    corrections = {
        "己申请": "已申请",
        "请假曰期": "请假日期",
        "员エ": "员工",
    }
    for el in elements:
        for wrong, correct in corrections.items():
            el.text = el.text.replace(wrong, correct)

    return elements


# ═══════════════════════════════════════════════════════════════
# 步骤 3：差异化的 Element → Document 转换
# ═══════════════════════════════════════════════════════════════

def element_to_document(el, source_file: str) -> LlamaDocument:
    """
    将单个 Element 转换为 LlamaIndex 的 Document。

    这是模式二的关键：你可以在此精确控制哪些 metadata 保留、
    哪些丢弃，以及 metadata 的字段名映射关系。
    """

    el_type = type(el).__name__

    # ── 元数据映射（从 Element metadata 中提取需要的字段）──
    meta = {
        # 来源信息
        "source": source_file,
        "file_name": el.metadata.filename or source_file,

        # 定位信息（模式一做不到的！）
        "page_number": el.metadata.page_number,
        # 坐标——前端 PDF 高亮需要
        "coordinates": {
            "x": el.metadata.coordinates.points[0][0] 
                if el.metadata.coordinates else None,
            "y": el.metadata.coordinates.points[0][1] 
                if el.metadata.coordinates else None,
            "width": el.metadata.coordinates.width 
                if el.metadata.coordinates else None,
            "height": el.metadata.coordinates.height 
                if el.metadata.coordinates else None,
        },

        # 层级信息（模式一做不到的！）
        "element_type": el_type,
        "parent_id": el.metadata.parent_id if el.metadata.parent_id else None,

        # 格式信息（模式一做不到的！）
        "emphasized_text": el.metadata.emphasized_text_contents 
                          if el.metadata.emphasized_text_contents else [],

        # 语言
        "languages": el.metadata.languages if el.metadata.languages else [],
    }

    # ── 差异化处理：不同 Element 类型用不同策略 ──────────────
    text = el.text

    if el_type == "Table":
        # 表格：保留结构化格式，不额外处理
        # Markdown 格式的表格 LLM 可以直接理解
        text = el.text  # Unstructured 已将其转为 Markdown 表格

    elif el_type == "ListItem":
        # 列表项：保留原始缩进和编号
        pass

    elif el_type == "Title":
        # 标题：可以作为后续 chunking 的锚点
        # 标记 level，后续可用 metadata.category_depth 判断层级
        pass

    elif el_type == "NarrativeText":
        # 正文：如果过长，后续会由 SentenceSplitter 再切分
        pass

    return LlamaDocument(text=text, metadata=meta)


# ═══════════════════════════════════════════════════════════════
# 步骤 4：完整流水线
# ═══════════════════════════════════════════════════════════════

def build_rag_pipeline_controlled(file_path: str):
    """
    模式二的完整流水线：解析 → 清洗 → 转换 → 切分 → 索引
    """

    # 1. 原生 Unstructured 解析（完全控制参数）
    print(f"[1/5] 解析文档: {file_path}")
    elements = parse_document(file_path, strategy="auto")

    # 2. 自定义清洗（插入解析后的处理逻辑）
    print(f"[2/5] 清洗前: {len(elements)} 个 Element")
    elements = clean_elements(
        elements,
        min_text_length=10,
        remove_headers_footers=True,
        # 只保留正文内容类型，丢弃 Header/Footer/Image
        keep_types=["NarrativeText", "Title", "Table", "ListItem",
                     "UncategorizedText"],
    )
    print(f"[2/5] 清洗后: {len(elements)} 个 Element")

    # 3. Element → LlamaIndex Document（精确的元数据映射）
    print(f"[3/5] 转换为 Document...")
    documents = [element_to_document(el, file_path) for el in elements]

    # 4. 分块（LlamaIndex 的 SentenceSplitter 做语义级切分）
    print(f"[4/5] 切分为 Node...")
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separator=" ",
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"[4/5] {len(documents)} 个 Document → {len(nodes)} 个 Node")

    # 5. 建索引
    print(f"[5/5] 构建索引...")
    index = VectorStoreIndex(nodes)
    print(f"[5/5] 索引就绪，共 {len(nodes)} 个 Node")

    return index


# ═══════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    index = build_rag_pipeline_controlled("星辰科技员工手册.pdf")
    engine = index.as_query_engine(similarity_top_k=3)

    response = engine.query("请事假需要提前多久申请？")
    print(f"\n回答: {response}")
    print(f"\n引用来源:")
    for node in response.source_nodes:
        coords = node.metadata.get("coordinates", {})
        print(f"  [{node.metadata.get('element_type', '?')}] "
              f"第{node.metadata.get('page_number', '?')}页 "
              f"相似度={node.score:.3f}")
        if coords.get("x") is not None:
            print(f"    坐标: ({coords['x']:.0f}, {coords['y']:.0f})")
```

#### 16.2.3 模式二的可控性体现在哪里

```
对比模式一和模式二的差异点：

  模式一 (UnstructuredReader)：
    📄 文档 → [UnstructuredReader.load_data()]
                                    │
                    内部做 Elements → Document（透明、不可控）
                                    │
                                    ▼
                            List[LlamaDocument] → 建索引
  
  模式二 (原生 Unstructured + 自定义)：
    📄 文档 → [partition()] 
                    │
                    ▼
              List[Elements]  ← 此时你可以：
                    │             · 查看每个 Element 的类型和内容
                    │             · 插入自定义清洗逻辑
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     过滤噪音    修正OCR错误   差异化处理
        │           │           │
        └───────────┼───────────┘
                    ▼
            [element_to_document()] ← 精确控制 metadata 映射
                    │
                    ▼
            List[LlamaDocument] → [SentenceSplitter] → 建索引
```

**模式二赋予你的控制权：**

| 控制点 | 模式一 | 模式二 |
|--------|:---:|:---:|
| 选择保留哪些 Element 类型 | ❌ 无法控制 | ✅ `keep_types=["Table","NarrativeText"]` |
| 插入自定义清洗逻辑 | ❌ | ✅ `clean_elements()` 完全自定义 |
| 精确控制 metadata 映射 | ❌ 被框架封装 | ✅ `element_to_document()` 逐字段映射 |
| 保留坐标信息（前端高亮） | ❌ 元数据丢失 | ✅ 手动提取后保留 |
| 差异化处理不同 Element 类型 | ❌ | ✅ Table保留结构/Text再切分/Title做锚点 |
| 修改 Element 文本内容 | ❌ | ✅ OCR纠错/格式修正/文本清洗 |

---

### 16.3 模式三：混合模式（最佳实践）

#### 16.3.1 核心策略

绝大多数生产级 RAG 系统最终都会走向混合模式：

```
  │  95% 的文档                          5% 的复杂文档
  │  · 格式标准的 PDF/Word              · 含扫描件的 PDF
  │  · 结构清晰的 Markdown              · 多栏排版 + 复杂表格
  │  · 常规 HTML 网页                  · 需要特殊清洗逻辑的文档
  │        │                                   │
  │        ▼                                   ▼
  │  模式一：UnstructuredReader          模式二：原生 Unstructured
  │  (快速标准模式)                      + 自定义清洗 + 手动接入
  │        │                                   │
  │        └───────────────┬───────────────────┘
  │                        │
  │                        ▼
  │             统一的 LlamaIndex Document 列表
  │                        │
  │                        ▼
  │              VectorStoreIndex.from_documents()
  │                        │
  │                        ▼
  │              统一的查询引擎 (QueryEngine)
```

#### 16.3.2 代码实现

```python
"""
模式三：混合模式（生产环境最佳实践）

核心思想：
  · 大部分常规文档 → 用 UnstructuredReader（省代码、快）
  · 少数复杂文档 → 回退到原生 Unstructured + 自定义逻辑
  · 所有文档最终汇聚到 LlamaIndex 的统一管线下建索引

实现方式：
  用一个工厂函数，根据文档的"复杂度标记"自动选择模式一或模式二。
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DocConfig:
    """单个文档的配置"""
    path: str
    complexity: str = "auto"      # "simple" / "complex" / "auto"
    strategy: str = "auto"        # "fast" / "hi_res" / "auto"
    languages: List[str] = None
    custom_clean: bool = False    # 是否需要自定义清洗
    keep_types: Optional[List[str]] = None  # 只保留的 Element 类型

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["chi_sim", "eng"]


def load_documents_hybrid(doc_configs: List[DocConfig]) -> List[LlamaDocument]:
    """
    混合模式的文档加载工厂。

    根据每份文档的配置，自动选择：
      - complexity="simple" → UnstructuredReader（快速）
      - complexity="complex" → 原生 Unstructured + 自定义逻辑（高可控）
      - complexity="auto"   → 根据文件扩展名自动判断
    """
    all_documents = []

    for cfg in doc_configs:
        # ── 自动判断复杂度 ──────────────────────────────────
        if cfg.complexity == "auto":
            # 简单的启发式规则（可扩展为用 AI 判断）
            ext = Path(cfg.path).suffix.lower()
            if ext == ".pdf":
                cfg.complexity = "complex"  # PDF 默认走复杂模式（安全）
            elif ext in [".md", ".txt"]:
                cfg.complexity = "simple"   # Markdown/TXT 走简单模式
            else:
                cfg.complexity = "simple"   # 其余默认简单

        print(f"  加载: {cfg.path} → 模式: {cfg.complexity}")

        # ── 简单模式 → UnstructuredReader ──────────────────
        if cfg.complexity == "simple":
            from llama_index.readers.unstructured import UnstructuredReader
            reader = UnstructuredReader()
            docs = reader.load_data(
                file=cfg.path,
                unstructured_kwargs={
                    "strategy": cfg.strategy,
                    "languages": cfg.languages,
                    "pdf_infer_table_structure": True,
                }
            )
            all_documents.extend(docs)

        # ── 复杂模式 → 原生 Unstructured + 自定义逻辑 ──────
        elif cfg.complexity == "complex":
            # 复用模式二的函数
            elements = parse_document(
                cfg.path, strategy=cfg.strategy, languages=cfg.languages
            )

            # 如果配置了自定义清洗
            if cfg.custom_clean:
                elements = clean_elements(
                    elements, keep_types=cfg.keep_types
                )

            # 精确转换
            docs = [element_to_document(el, cfg.path) for el in elements]
            all_documents.extend(docs)

    return all_documents


# ═══════════════════════════════════════════════════════════════
# 混合模式使用示例
# ═══════════════════════════════════════════════════════════════

doc_configs = [
    # 常规文档：走快速模式
    DocConfig(path="./docs/员工手册.md", complexity="simple"),

    # 复杂 PDF：走原生 + 自定义清洗
    DocConfig(
        path="./docs/财务年报.pdf",
        complexity="complex",
        strategy="hi_res",              # 含表格的财务报告用高精度
        custom_clean=True,              # 开启自定义清洗
        keep_types=["NarrativeText", "Table", "Title"],  # 只保留这些
    ),

    # 扫描件：必须走 hi_res + OCR
    DocConfig(
        path="./docs/旧版合同扫描件.pdf",
        complexity="complex",
        strategy="hi_res",
        languages=["chi_sim"],          # 纯中文扫描件
        custom_clean=True,
    ),

    # 标准 Word 文档：走快速模式
    DocConfig(path="./docs/培训资料.docx", complexity="simple"),
]

# 统一加载
all_docs = load_documents_hybrid(doc_configs)

# 统一建索引 → 统一查询
from llama_index.core.node_parser import SentenceSplitter
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = node_parser.get_nodes_from_documents(all_docs)
index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine()
```

#### 16.3.3 复杂度判断的自动化

在生产环境中，"手工为每份文档标 complexity" 是不现实的。以下是几种自动判断策略：

```
策略 A：基于文件扩展名（最简单，代码中已实现）
  .md / .txt / .csv  → simple
  .pdf               → complex（安全起见）
  .docx / .pptx      → simple（Unstructured 对 Office 格式支持好）
  
策略 B：基于文件大小 + 页数
  文件 < 1MB 且 页数 < 50  → simple
  文件 > 1MB 或  页数 > 50  → complex

策略 C：基于采样分析（最可靠，但增加开销）
  对 PDF 的前 3 页用 fast 模式采样：
    如果提取结果为空 → 扫描件 → complex
    如果提取结果中 Table 元素 > 总元素 30% → 表格密集型 → complex
    其他 → simple

策略 D：基于业务标记
  在文件管理系统中标记"重要文档" → complex
  标记"普通文档" → simple
  在文件入库时就做好分类
```

---

### 16.4 工具选型全景决策框架

不仅仅是 Unstructured 和 LlamaIndex 的选型，RAG 生态中有多种文档解析工具。以下是一个基于场景的决策框架：

```
═══════════════════════════════════════════════════════════════════════════
                    文档解析工具选型决策框架
═══════════════════════════════════════════════════════════════════════════

  你的文档类型是什么？

  ┌─ 纯文本 / Markdown
  │   → 直接用 Python 的 open().read()
  │   → 或 LlamaIndex 的 SimpleDirectoryReader
  │   理由：无需任何解析，直接读取即可
  │
  ├─ Word / PPT / Excel (Office 格式)
  │   → LlamaIndex 的 SimpleDirectoryReader(file_extractor={".docx":"default"})
  │   理由：Office 格式结构规范，内置解析器够用，不需 Unstructured
  │   如果需要保留复杂格式（嵌套表格等）→ 用 Unstructured
  │
  ├─ 数字原生 PDF（Word/LaTeX 导出）
  │   → 优先 PyMuPDF (fitz)，速度最快、中文支持好
  │   → 如果含复杂表格 → Unstructured fast 模式
  │   → 如果需要结构化输出 + 元数据 → Unstructured
  │   理由：数字原生 PDF 有文本层，不需要 OCR
  │
  ├─ 扫描件 PDF / 图片型 PDF
  │   → Unstructured hi_res + PaddleOCR（中文为主）
  │   → Unstructured hi_res + Tesseract（英文为主）
  │   → 商业 OCR API（百度OCR/腾讯OCR）→ 准确率最高但有成本
  │   理由：必须 OCR，Unstructured 集成了 OCR 引擎
  │
  ├─ 学术论文 / 含数学公式的 PDF
  │   → Marker（PDF → Markdown，对公式有专门处理）
  │   → Nougat（Meta 开源，学术 PDF 转 Markdown + LaTeX）
  │   理由：通用工具对数学公式的提取效果差
  │
  ├─ 表格密集的 PDF（财务报表、数据报告）
  │   → Unstructured hi_res + pdf_infer_table_structure=True
  │   → 或 Camelot + pdfplumber（纯表格提取）
  │   → 或 LlamaParse（LlamaIndex 官方云服务，表格效果好）
  │   理由：表格还原是专门的难题
  │
  ├─ HTML 网页
  │   → BeautifulSoup 清洗 + LlamaIndex SimpleWebPageReader
  │   → 动态渲染页面 → Playwright/Selenium 先渲染再提取
  │   理由：HTML 的结构信息（标签）可以辅助切分
  │
  └─ 代码仓库
      → LlamaIndex GithubRepositoryReader
      → 按文件/函数/类切分为自然单元，而非按字符数硬切
      理由：代码的结构单元（函数/类）才是自然的检索单元
```

---

### 16.5 总结：从选择到落地

```
                      Unstructured × LlamaIndex 选型总结

  ┌─────────────────────────────────────────────────────────────────────┐
  │  模式选择                                                            │
  │                                                                     │
  │  模式一：快速标准                                                    │
  │    UnstructuredReader 一行代码                                       │
  │    适合：原型 / MVP / 标准文档 / 小规模                               │
  │    成本：低                                                          │
  │                                                                     │
  │  模式二：高可控                                                      │
  │    原生 Unstructured + 自定义清洗 + 手动接入 LlamaIndex              │
  │    适合：生产环境 / 复杂文档 / 需要精细控制                           │
  │    成本：中高                                                        │
  │                                                                     │
  │  模式三：混合（最佳实践）                                             │
  │    95% 文档用模式一 + 5% 复杂文档用模式二                            │
  │    适合：绝大多数生产级 RAG 系统                                     │
  │    成本：中（绝大多数文档不额外投入，只对少数复杂文档投入）            │
  └─────────────────────────────────────────────────────────────────────┘

  工具选型原则：
    1. 先简单后复杂：先用最简单的工具验证可行性，不够再升级
    2. 根据文档类型选工具，而非"一把梭"
    3. 生产环境中混合使用是常态，不要追求单一工具覆盖所有场景
    4. 保留回退路径：任何封装都可能不够用，确保你能退回到底层库
```

**核心经验：**

> 1. 文档解析不是"选一个最好的工具"，而是"为每类文档选最合适的工具"。
> 2. 框架封装（模式一）解决 80% 的问题，原生库（模式二）解决剩下的 20%。
> 3. 永远保留能退回到 `partition()` 的能力——当封装不够用时，你不会被困住。
> 4. 元数据是 RAG 质量的倍增器——能保留就保留，它们在未来（前端高亮、精确溯源、按日期过滤）会用得上。


---

## 十七、大厂面试题深度讲解

### 17.1 三种集成模式对比类

#### Q1：Unstructured 和 LlamaIndex 的三种集成模式分别是什么？各自的适用场景和 trade-off 是什么？（字节跳动 / 阿里巴巴 高频）

**面试官考察点：** 这道题考察对"简单 vs 可控"这一工程核心矛盾的认知。大厂面试官想看的不是一个标准答案，而是你如何根据场景做 trade-off 判断。

**回答框架（先总览三种模式，再讲决策逻辑）：**

**三种模式一句话定义：**

| 模式 | 一句话 | 关键词 |
|------|--------|--------|
| **模式一：UnstructuredReader 封装** | 一行代码搞定解析+转换，但中间过程不可控 | 快、简单、黑盒 |
| **模式二：原生+自定义+手动接入** | 完全控制每一步，但代码量多、维护成本高 | 慢、精细、白盒 |
| **模式三：混合模式** | 95% 文档用模式一，5% 复杂文档用模式二 | 务实、ROI 最高 |

**三者的核心差异——"权力与责任的转移"：**

```
模式一（UnstructuredReader）：
  Unstructured 持有权力 → 自动做 Element→Document 转换
  你付出的代价 → 丧失对转换过程的控制权
  你的责任 → 接受框架的默认行为

模式二（原生 + 自定义）：
  你持有权力 → 你决定每个 Element 如何清洗、如何映射
  你付出的代价 → 需要写更多代码、理解 Element 类型体系
  你的责任 → 自己维护解析质量

模式三（混合）：
  权力按文档分配 → 简单文档交给框架，复杂文档亲自掌控
  代价 → 需要维护两套逻辑 + 一个路由层
  收益 → 用最小的投入获得最大的控制力
```

**场景决策矩阵：**

| 场景条件 | 推荐模式 | 理由 |
|----------|:---:|------|
| 原型 / MVP，预算 3 天 | 模式一 | 速度优先，能跑通就行 |
| 文档全是标准格式（Word 导出 PDF、纯 MD） | 模式一 | 标准格式不需要定制逻辑 |
| 含扫描件 / 复杂表格 / 多栏排版 | 模式二 | 需要 hi_res + 自定义清洗 |
| 需要前端 PDF 高亮（需要坐标元数据） | 模式二 | 模式一会丢弃坐标信息 |
| 生产系统，文档 > 1000 份，类型混杂 | 模式三 | 95/5 分配，ROI 最高 |
| 团队只有后端工程师，无 NLP 背景 | 模式一 | 降低维护成本 |
| RAG 是核心产品，解析质量 = 产品竞争力 | 模式三 → 逐步过渡到模式二 | 解析是核心壁垒，值得投入 |

**面试官追问（高压）："你个人在实际项目中更倾向哪个？"**

"我在三个项目上都验证过这个结论——**模式三是最务实的起点，模式二是最终目的地**。原因：
1. 刚接手一个文档库时，你不知道哪些文档会有问题。先用模式一把所有文档跑一遍，观察哪些文档解析质量差——这就是你的'复杂文档清单'。
2. 对这个清单中的文档（通常占比 5-15%），切换到模式二做精细处理。
3. 随着系统持续运行，复杂文档的处理逻辑逐渐沉淀，模式二的代码越来越多，但都是被真实 bad case 驱动的——每一行清洗逻辑都有对应的线上问题做背书。

这比一上来就全部用模式二聪明得多——你不会为 80% 根本不需要清洗的文档写无用的清洗代码。"

---

#### Q2：模式一中，UnstructuredReader 内部做了什么？哪些信息在封装过程中丢失了？（百度 / 腾讯）

**面试官考察点：** 是否真正理解框架封装的信息损失。能答出"什么丢失了"说明你不是只会用 API，而是思考过原理。

**回答思路（先讲转换流程，再讲丢失了啥）：**

**UnstructuredReader 内部的转换流程：**

```
UnstructuredReader.load_data(file="员工手册.pdf")
    │
    ▼
1. 调用 Unstructured 的 partition()
   → 返回 List[Element]（每个 Element 有：type, text, metadata）

2. 遍历每个 Element，调用 LlamaIndex 内部的 Element→Document 转换器
   → 将 Element.type 映射为简单的字符串标签
   → 从 Element.metadata 中提取部分字段
   → 丢弃大量"非标准"字段
   → 生成 LlamaIndex Document

3. 返回 List[Document]
```

**封装过程中丢失的五个关键信息：**

| 丢失信息 | 在 Element 中（模式二可保留） | 在 Document 中（模式一） | 丢失后果 |
|----------|------------------------------|------------------------|----------|
| **元素类型标签** | `type(el).__name__` → "Title" / "Table" / "ListItem" | 丢失（所有 element 合并为同质 Document） | 无法按类型做差异化处理 |
| **坐标信息** | `metadata.coordinates` → {x, y, width, height} | 丢失 | 前端 PDF 高亮无法实现 |
| **层级关系** | `metadata.parent_id` / `category_depth` | 丢失 | 无法按章节层级过滤检索范围 |
| **字体信息** | `metadata.emphasized_text` → 哪些文字是加粗的 | 丢失 | 无法利用加粗信息做检索权重提升 |
| **元素间关系** | 相邻 Element 的 PREVIOUS/NEXT 关系 | 丢失 | 无法做上下文自动扩展 |

**关键信息丢失的实战案例：**

```
场景：财务报表 PDF 中的一页

  Element #1 (Title):       "三、2024年度财务数据"
  Element #2 (NarrativeText): "本年度公司实现营业收入..."
  Element #3 (Table):        "| 项目 | Q1 | Q2 | Q3 | Q4 |\n| 营收 | 1.2亿 | ... |"
  Element #4 (NarrativeText): "注：以上数据未经审计。"

模式一处理后：
  → 4 个同质的 Document，type 信息全部丢失
  → 检索时无法区分哪个是标题、哪个是表格、哪个是脚注
  → Table Document 可能被 SentenceSplitter 拦腰截断

模式二处理后：
  → 4 个 Document，每个带着 element_type metadata
  → 可以写规则：Table 类型的 Document 不参与切分（保持完整）
  → 检索结果可以标注类型：答案来自于 Table 元素
```

**面试官追问："这些丢失的信息真的有用吗？不用它们，RAG 不是也能跑？"**

"能跑和跑得好是两回事。我举两个具体例子：
1. **坐标信息=溯源体验。** 你回答'年假是 5 天'，用户追问'你凭什么这么说？'。有坐标信息→前端 PDF 查看器直接红框高亮原文位置，用户秒信任。没有坐标→只能告诉用户'大概是员工手册第 12 页'，用户自己翻。在 B 端 SaaS 产品中，这个体验差异直接影响付费意愿。
2. **元素类型=检索精度。** 标题和正文的检索权重应该不同。匹配到一个 Title 元素意味着整个章节都相关，但模式一下你无法知道它是一个 Title。你可以用这些丢失的信息做检索增强，这是模式一的隐性损失。"

---

### 17.2 自定义清洗与处理类

#### Q3：模式二中，你会设计哪些自定义清洗逻辑？页眉页脚如何有效去除？（美团 / 腾讯 工程细节题）

**面试官考察点：** 这是工程落地题——是否有真实处理过"脏数据"的经验。能说出清洗策略说明真的上线过 RAG。

**回答思路（四层清洗策略，从简单到复杂）：**

**第一层：类型过滤（最基础，也最有效）**

```python
# 只保留对 RAG 有意义的 Element 类型
KEEP_TYPES = [
    "Title",           # 标题（作为 chunk 的语义标签）
    "NarrativeText",   # 正文（RAG 的核心信息来源）
    "Table",           # 表格（保留 Markdown 格式）
    "ListItem",        # 列表项（保留层级）
    "UncategorizedText", # 未分类文字（兜底，防遗漏）
]

# 主动丢弃的：
DROP_TYPES = [
    "Header",    # 页眉（每页重复，纯噪声）
    "Footer",    # 页脚（"第X页"、"版权所有"等重复内容）
    "Image",     # 图片（文本 RAG 无法利用，多模态 RAG 单独处理）
]
```

**第二层：长度过滤**

```python
# 太短的文本大概率是噪声（OCR 碎片、单个字符、纯数字页码）
MIN_TEXT_LENGTH = 10  # 丢弃 < 10 个字符的片段

# 太长的文本可能是解析错误（整页没有做布局分析，把一页当一段）
# 这种需要标记，后续人工审核或重新解析
MAX_TEXT_LENGTH = 5000
```

**第三层：页眉页脚去除（这是高频追问）——三种策略：**

**策略 A——统计重复检测（最常用，也最稳健）：**

```python
def detect_headers_footers_by_repetition(elements, threshold=0.6):
    """
    原理：页眉页脚的最大特征是"每页都有，内容几乎相同"。
    统计所有 Element 的文本在多页中的出现频率，
    如果某文本出现在 > 60% 的页面中 → 判定为页眉/页脚。
    """
    from collections import Counter
    
    # 统计文本 → 出现的页数
    text_pages = {}
    for el in elements:
        text = el.text.strip()
        page = el.metadata.page_number
        if page is None:
            continue
        if text not in text_pages:
            text_pages[text] = set()
        text_pages[text].add(page)
    
    total_pages = len(set(p for pages in text_pages.values() for p in pages))
    threshold_count = max(total_pages * threshold, 2)
    
    # 标记高频文本
    repeated_texts = {
        text for text, pages in text_pages.items()
        if len(pages) >= threshold_count
    }
    
    return repeated_texts
```

**策略 B——位置启发式（精确但需要坐标信息）：**

```python
def detect_by_position(el):
    """
    页眉：页面顶部 10% 区域内的 Element
    页脚：页面底部 10% 区域内的 Element
    """
    page_height = el.metadata.coordinates.system_height  # 需要获取
    y_top = el.metadata.coordinates.points[0][1]
    
    if y_top < page_height * 0.10:
        return "header"          # 在页面顶部 10%
    elif y_top > page_height * 0.90:
        return "footer"          # 在页面底部 10%
    return "body"
```

**策略 C——内容模式匹配（兜底）：**

```python
import re

HEADER_FOOTER_PATTERNS = [
    r'^第\s*\d+\s*页$',           # "第3页"
    r'^Page\s+\d+$',              # "Page 3"
    r'^\d+\s*/\s*\d+$',           # "3 / 45"
    r'^版权所有.*$',               # "版权所有 © 2025"
    r'^Confidential$',            # 水印
    r'^内部资料.*$',               # "内部资料 请勿外传"
]

def match_header_footer_pattern(text):
    for pattern in HEADER_FOOTER_PATTERNS:
        if re.match(pattern, text.strip()):
            return True
    return False
```

**最佳实践：三者组合使用。** 策略 A（统计检测）做主要手段，策略 B（位置）做辅助确认，策略 C（模式匹配）做兜底。顺序是：先用策略 B 快速过滤明显的位置特征，再用策略 A 捕获"不靠边但重复"的内容（如侧边栏的导航文字），最后用策略 C 捕获特定模式（如不同格式的页码）。

**第四层：OCR 错误修正（中文场景特殊需求）**

```python
# OCR 常见错误映射表（随 bad case 持续积累）
OCR_CORRECTIONS = {
    "己申请": "已申请",     # "已" → "己"
    "请假曰期": "请假日期", # "日" → "曰"
    "员エ": "员工",         # "工" → "エ"
    "缴纟内": "缴纳",       # "纳" 被拆分识别
    "—般": "一般",          # "一" → "—"
}

# 对于关键业务词汇做严格校验
# 例如："缴纳比例" 出现的词，检查是否被 OCR 错误为 "缴纟内比例"
```

**面试官追问："页眉页脚检测的阈值 60% 是怎么定的？低了会漏，高了会误删怎么办？"**

"这是一个在项目调试中根据实际文档调整的经验值。实践中我分两步处理：
1. 先用自动检测标记候选，但**不直接删除**——降低误删风险。标记为 `metadata.is_suspected_noise=True`。
2. 人工抽查 50-100 个被标记的 Element，确认准确率。如果准确率 > 95%，自动删除；如果 80-95%，只对高置信度（出现频率 > 80%）的自动删除；如果 < 80%，全部交给人工审核。

宁可'漏删'（页眉页脚混入索引，检索时权重自然低），不要'误删'（把重要的章节标题当页眉删掉）。因为漏删最多增加一些检索噪声，误删意味着这段知识彻底在 RAG 中消失了。"

---

#### Q4：模式二中，如何设计 Element → Document 的 metadata 映射？哪些 metadata 该保留、哪些该丢弃？（阿里巴巴 / 华为）

**面试官考察点：** 元数据设计是 RAG 工程化的标志。考察是否理解"什么信息有价值、什么信息是噪声"。

**回答思路（分层决策哪些该保留，按成本-收益排序）：**

**元数据保留的三层决策框架：**

```
第一层（P0，必须保留——零成本、高收益）：
  这些是最基础的元数据，不需要额外计算，Unstructured 直接提供
  ├─ source / filename    → 溯源的基础
  ├─ page_number          → "见员工手册第3页"
  └─ element_type         → 差异化处理的基础

第二层（P1，强烈建议保留——中成本、高收益）：
  这些需要额外映射工作，但收益明显
  ├─ coordinates          → 前端 PDF 高亮的唯一依据
  ├─ parent_id / section  → 检索时扩展上下文、按章节过滤
  └─ emphasized_text      → 加粗内容在检索评分中加权

第三层（P2，按需保留——高成本、场景依赖）：
  这些需要额外存储和处理
  ├─ font_size            → 推断标题层级
  ├─ languages            → 多语言知识库过滤
  └─ category_depth       → 按内容层级过滤（查"章"级 vs "条"级）
```

**metadata 设计的具体实现（代码示例的核心设计）：**

```python
def element_to_document(el, source_file: str) -> Document:
    """精确控制 metadata 映射"""
    
    el_type = type(el).__name__
    
    meta = {
        # ── P0：必须保留 ──
        "source": source_file,
        "page_number": el.metadata.page_number,
        "element_type": el_type,
        
        # ── P1：强烈建议 ──
        "coordinates": _extract_coordinates(el),   # 前端高亮
        "parent_id": el.metadata.parent_id,         # 层级关系
        
        # ── P2：按需保留 ──
        "emphasized_text": el.metadata.emphasized_text_contents or [],
        "languages": el.metadata.languages or [],
    }
    
    # ── 关键设计：可扩展性 ──
    # 保留一个 raw 字段，存放完整的原始 metadata
    # 这样未来需要新字段时，不需要重新解析文档
    meta["_raw_metadata"] = {
        "category_depth": el.metadata.category_depth,
        "filetype": el.metadata.filetype,
        "last_modified": el.metadata.last_modified,
    }
    
    return Document(text=el.text, metadata=meta)
```

**哪些 metadata 该主动丢弃——三项原则：**

1. **对检索无帮助且占用存储的字段：** 如 Element 的 UUID（唯一标识，但没有检索语义）、内嵌图片的 base64 编码（太大了，存对象存储即可，不放在 metadata 里）。

2. **不可靠的字段：** 如 PDF 文档属性中的 author 和 title（很多 PDF 这些字段为空或填充错误）。如果要使用，必须有兜底逻辑——author 为空时用文件名推断。

3. **不适合进向量索引的字段：** 如长文本的完整路径、详细坐标点数组等。这些适合放在传统数据库（PostgreSQL/ES）中按需关联查询，而非塞进每个 Node 的 metadata 里。

**面试官追问："_raw_metadata 这种设计会不会维度爆炸？"**

"_raw_metadata 不会被检索也不进入 Embedding。它只在两个场景被用到——调试 bad case 时回看原始数据，以及未来需要新增 metadata 字段时直接从 _raw 取而无需重新解析。它本质是一个降级缓存。存储成本上，metadata 本身不参与向量化，每个 Node 的 _raw 字段也就几百字节，10 万个 Node 才几十 MB，完全可以接受。"

---

### 17.3 混合模式与生产化类

#### Q5：模式三（混合模式）中，如何自动判断一份文档该走"简单模式"还是"复杂模式"？（字节跳动 / 拼多多）

**面试官考察点：** 自动化决策是生产系统的标志。考察能否设计一个实用的文档复杂度判定算法。

**回答思路（分级策略，从简单到复杂）：**

**策略一：基于文件扩展名（最简单，覆盖 70% 场景）**

```python
COMPLEXITY_RULES_BY_EXT = {
    ".md": "simple",      # Markdown 结构清晰
    ".txt": "simple",      # 纯文本，无需解析
    ".csv": "simple",      # CSV 结构固定
    ".docx": "simple",     # Office 格式结构规范，Unstructured 支持好
    ".pptx": "simple",
    ".html": "simple",
    ".pdf": "complex",     # PDF 默认走复杂模式（安全第一）
    ".png": "complex",     # 图片 → 必须 OCR
    ".jpg": "complex",
}
```

**策略二：PDF 内部分类——判断是数字原生还是扫描件（最关键的判断）**

```python
def classify_pdf(file_path: str) -> str:
    """
    对 PDF 做内容采样，判断是数字原生还是扫描件。
    这是整个复杂度判断中最关键的一步。
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(file_path)
    
    # 采样前3页（控制开销）
    sample_pages = min(3, len(doc))
    total_text_length = 0
    total_image_count = 0
    
    for i in range(sample_pages):
        page = doc[i]
        
        # 检测文字层
        text = page.get_text()
        total_text_length += len(text.strip())
        
        # 检测图片数量
        images = page.get_images()
        total_image_count += len(images)
    
    avg_text_per_page = total_text_length / sample_pages
    
    # ── 判定规则 ──
    if avg_text_per_page > 200:
        # 每页平均 > 200 字符 → 有丰富文本层 → 数字原生 PDF
        return "simple"    # 可以用 fast 模式
    elif avg_text_per_page < 10:
        # 几乎提取不到文字 → 扫描件
        return "complex"   # 必须 hi_res + OCR
    else:
        # 中间地带 → 可能有文本但质量差 → 走复杂模式（安全侧）
        return "complex"
    
    doc.close()
```

**策略三：基于内容的表格密度检测**

```python
def detect_table_density(elements) -> str:
    """
    先用 fast 模式跑一遍，统计 Table Element 占比。
    表格密集 = 复杂文档（需要更精细的解析）。
    """
    table_count = sum(1 for el in elements if type(el).__name__ == "Table")
    total_count = len(elements)
    
    if total_count == 0:
        return "simple"
    
    table_ratio = table_count / total_count
    
    if table_ratio > 0.30:
        return "complex"   # 超过30%的内容是表格 → 财务报表/数据报告类
    return "simple"
```

**策略四：综合评分模型（生产环境推荐）**

```python
def compute_complexity_score(file_path: str) -> float:
    """
    综合多个特征打分，返回 0-1 的复杂度分数。
    > 0.5 → 走复杂模式
    ≤ 0.5 → 走简单模式
    """
    score = 0.0
    
    ext = Path(file_path).suffix.lower()
    
    # 特征1：文件格式（权重 0.3）
    if ext == ".pdf":
        score += 0.3
    elif ext in [".png", ".jpg", ".tiff"]:
        score += 0.5                    # 图片一定复杂
    
    # 特征2：PDF 内容类型（权重 0.4）
    if ext == ".pdf":
        is_scanned = check_if_scanned(file_path)
        if is_scanned:
            score += 0.4
    
    # 特征3：文件大小（权重 0.1）
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 50:
        score += 0.1                    # 大文件更可能是扫描件
    
    # 特征4：页数（权重 0.2）
    if ext == ".pdf":
        page_count = get_page_count(file_path)
        if page_count > 100:
            score += 0.2                # 多页文档更可能有复杂布局
    
    return score
```

**面试官追问："如果分类错了会怎样？比如把扫描件误判为数字原生 PDF？"**

"这是生产中会遇到的情况。两个层面的防护：
1. **自动降级兜底：** fast 模式提取结果如果文字覆盖率 < 5% → 自动判断为误判 → 重新用 hi_res 模式处理。成本多一次解析，但不会丢数据。
2. **质量评分标识：** 每个文档解析后计算质量分（文字覆盖率、非法字符比例）。低分文档标记为'需人工处理'或'下次用更高质量策略重新解析'。这意味着误判不会导致数据丢失，最多是延迟——等人工或用更强工具再处理一次。"

---

#### Q6：在大规模文档库场景下（10 万+ 文档），文档解析管线应该怎么设计？如何平衡成本和时效性？（腾讯 / 百度 系统设计题）

**面试官考察点：** 大规模场景下的工程架构能力。考察对"把解析做成独立服务"的理解。

**回答思路（四层架构 + 成本控制）：**

**架构设计——离线解析、在线服务分离：**

```
┌──────────────────────────────────────────────────────────────┐
│                     离线解析管线（异步）                       │
│                                                              │
│  [文档上传] → [S3/MinIO 对象存储]                            │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────┐                                       │
│  │   消息队列 (Kafka) │  ← 文档元数据事件                      │
│  │   {doc_id, path,  │                                       │
│  │    complexity_tag} │                                       │
│  └──────┬───────────┘                                       │
│         │                                                    │
│    ┌────┴──────────────┐                                    │
│    ▼                   ▼                                     │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │Simple Worker │  │Complex Worker│  ← 按复杂度分流           │
│  │ (CPU, 模式一) │  │ (GPU, 模式二) │                          │
│  │              │  │              │                           │
│  │ 处理 95% 文档 │  │ 处理 5% 文档  │                           │
│  │ fast 模式    │  │ hi_res 模式  │                           │
│  │ 速度：秒级   │  │ 速度：分钟级  │                           │
│  └──────┬──────┘  └──────┬──────┘                           │
│         │                │                                    │
│         └────────┬───────┘                                    │
│                  ▼                                            │
│         ┌──────────────┐                                     │
│         │  解析结果存储  │                                     │
│         │  · JSON 原文  │ → 对象存储 (S3)                     │
│         │  · Chunk 文本 │ → 向量数据库 (Milvus)               │
│         │  · 元数据     │ → PostgreSQL / ES                   │
│         └──────────────┘                                     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                     在线 RAG 服务（同步）                      │
│                                                              │
│  用户 Query → LlamaIndex QueryEngine                         │
│    → 向量检索 (已建好的索引)                                   │
│    → LLM 生成                                                 │
│    → 返回回答 + 溯源                                           │
│                                                              │
│  不包含任何文档解析操作！                                       │
└──────────────────────────────────────────────────────────────┘
```

**成本控制——按文档复杂度分级使用资源：**

```
10 万份文档的解析成本估算：

假设：
  - 85% 是简单文档（数字原生 PDF/Word/Markdown）
  - 10% 是中等复杂度（含表格的 PDF）
  - 5% 是高复杂度（扫描件/图文混排）

资源配置：
  ┌────────────┬──────────┬────────────┬──────────┐
  │ 复杂度      │ 占比      │ Worker 类型 │ 单份耗时   │
  ├────────────┼──────────┼────────────┼──────────┤
  │ 简单        │ 85,000   │ CPU (8核)  │ ~5s      │
  │ 中等        │ 10,000   │ CPU (8核)  │ ~30s     │
  │ 复杂        │ 5,000    │ GPU (T4)   │ ~60s     │
  └────────────┴──────────┴────────────┴──────────┘

并行度：10 个 CPU Worker + 2 个 GPU Worker
  
  简单文档：85,000 × 5s / 10 Workers ≈ 11.8 小时
  中等文档：10,000 × 30s / 10 Workers ≈ 8.3 小时
  复杂文档：5,000 × 60s / 2 Workers ≈ 41.7 小时
  
  总计：~60 小时（并行处理，约 2.5 天）

成本（自建）：
  CPU 服务器：~$200/月
  GPU 云实例：~$300 (按需，处理完可释放)
  总计：约 $500
```

**时效性保证——增量更新 vs 全量重跑：**

```
全量重跑（首次入库 / 大版本升级）：
  → 10 万份，2.5 天，约 $500
  → 策略：分批处理，优先高优先级文档

增量更新（日常运行）：
  → 每天新增/更新 100 份
  → 1 个 CPU Worker + 1 个 GPU Worker 即可
  → 每天 ~1 小时处理完
  → 解析结果缓存（文档未变不重解析）
  → 成本：几乎可以忽略

关键设计：文档指纹（MD5 / SHA256）
  → 文档内容没变 → 跳过解析，复用缓存结果
  → 只在文档实际变化时才触发重新解析
```

**面试官追问："如果一份 500 页的 PDF 解析到第 450 页失败了，要重跑整个文件吗？"**

"不需要。两个机制：
1. **断点续传：** 每页解析完就写入对象存储，记录 `{doc_id}_{page_number} → parsed_result`。失败了只需要从失败页继续，不需要重跑前面的页。
2. **渐进式可用：** 不需要等整个 PDF 解析完才可检索。解析完的前 100 页可以先建索引上线，后面的页面逐步追加。这意味着用户至少能检索到已解析的部分，而不是面对一个完全不可用的知识库。"

---

### 17.4 工具选型与决策框架类

#### Q7：面对一个实际的 RAG 项目，你会如何为不同文档选择最合适的解析工具？画一个决策树。（所有大厂通用 综合决策题）

**面试官考察点：** 多维度的工具选型能力——不是记住每个工具的名字，而是知道什么场景下该选什么。

**回答思路（分层决策树）：**

```
═══════════════════════════════════════════════════════════════════
              文档解析工具全景决策树
═══════════════════════════════════════════════════════════════════

第一层：文档格式是什么？

① 纯文本 (.txt)
  → Python 内置 open().read()
  理由：零依赖，最快

② Markdown (.md)
  → LlamaIndex SimpleDirectoryReader + MarkdownReader
  理由：保留 Markdown 结构（标题层级/列表/代码块）用于后续语义切分

③ Office 格式 (.docx / .pptx / .xlsx)
  ├─ 标准格式，无特殊嵌套
  │   → LlamaIndex SimpleDirectoryReader (file_extractor={"docx":"default"})
  │   理由：内置解析器够用，不引入 Unstructured 依赖
  │
  └─ 含复杂嵌套表格 / 特殊排版
      → Unstructured (fast 模式)
      理由：表格还原能力比内置解析器强

④ PDF (.pdf) —— 最复杂的判断
  │
  ├─ 第一步：判断 PDF 类型
  │   ├─ 数字原生 PDF（Word/LaTeX 导出，有文本层）
  │   │   → 继续第二步
  │   │
  │   └─ 扫描件 PDF（文本层为空）
  │       → Unstructured hi_res + PaddleOCR（中文）/ Tesseract（英文）
  │       理由：需要 OCR，Unstructured 集成了 OCR 引擎
  │
  ├─ 第二步（数字原生）：判断内容特征
  │   ├─ 简单文本为主，无表格
  │   │   → PyMuPDF (fitz)
  │   │   理由：最快，C 层面实现
  │   │
  │   ├─ 含表格
  │   │   → Unstructured (fast + pdf_infer_table_structure=True)
  │   │   理由：表格结构还原能力强
  │   │
  │   ├─ 含数学公式（学术论文）
  │   │   → Marker / Nougat
  │   │   理由：专用公式处理，通用工具做不到
  │   │
  │   └─ 含图表 + 图片（多模态需求）
  │       → Unstructured (hi_res + extract_images) + VLM Caption
  │       理由：需要同时提取文字和图片描述
  │
  └─ 第三步：判断规模与预算
      ├─ < 100 份 → 用最合适的工具，不计较成本
      ├─ 100-1000 份 → Unstructured auto 模式（自动判断）
      └─ > 1000 份 → 混合策略
          数字原生用 PyMuPDF（快+便宜）
          扫描件用 Unstructured（准+贵）
          按复杂度路由

⑤ 网页 (.html)
  → BeautifulSoup 清洗 + LlamaIndex SimpleWebPageReader
  理由：HTML 标签可作为语义切分的辅助信息

⑥ 代码仓库
  → LlamaIndex GithubRepositoryReader
  理由：按函数/类边界切分，而非按字符数硬切
```

**关键话术：** "这个决策树不是为了让你记住每个节点，而是展示一个方法论——**先判断文档的物理特征（格式→类型→内容），再匹配工具的强项**。没有人用一个工具覆盖所有场景，混合使用是常态。"

---

#### Q8：你在什么情况下会选择 LlamaParse（LlamaIndex 云服务），而不是自己搭建 Unstructured 解析管线？（美团 / 拼多多 工程决策题）

**面试官考察点：** 自建 vs 云服务的 trade-off 判断力。

**回答思路：**

**LlamaParse 的优势（什么时候选它）：**

| 优势 | 具体表现 |
|------|----------|
| **零运维** | 不需要管理 GPU 实例、不需要处理模型版本升级 |
| **表格处理强** | LlamaParse 对复杂表格的还原效果是开源的标杆之一 |
| **深度集成** | 与 LlamaIndex 无缝衔接，一行代码 `LlamaParse().load_data()` |
| **持续优化** | 解析模型持续更新，你不需要自己跟进最新模型 |
| **快速验证** | MVP 阶段 1 天内跑通，不需要搭建解析基础设施 |

**自建 Unstructured 管线的优势（什么时候自建）：**

| 优势 | 具体表现 |
|------|----------|
| **数据不出内网** | 金融/医疗/政务等强合规场景 |
| **成本可控** | 日均 > 10000 页时，自建成本远低于 API 调用费 |
| **完全定制** | 自定义清洗逻辑、自定义 OCR 引擎、自定义表格还原规则 |
| **无 vendor lock-in** | 不被特定云服务绑定，技术栈自主可控 |

**决策框架：**

```
你的项目特征是什么？

├─ 数据敏感 / 合规严格（金融、医疗、政务）
│   → 必须自建，没有商量余地

├─ MVP 验证阶段
│   → LlamaParse，快速出效果给老板看
│   → 验证通过后再评估是否切换自建

├─ 日均解析量 < 1000 页
│   → LlamaParse，人力成本 > API 成本
│   → 把精力花在检索和生成的优化上，不要花在解析管线搭建上

├─ 日均解析量 > 10000 页
│   → 自建 Unstructured 管线
│   → API 费用会快速增长，自建的经济优势越来越明显

├─ RAG 是核心产品、差异化竞争力
│   → 自建（解析质量是你的护城河）
│   → 竞争对手也能用 LlamaParse，解析层面拉不开差距

└─ RAG 是支撑性功能、内部工具
    → LlamaParse 或 Dify/RAGFlow 这类全托管平台
    → 不要在不产生差异化的地方投入
```

**面试官追问（高分陷阱）："如果先用了 LlamaParse，后面要迁移到自建，迁移成本大吗？"**

"这是架构设计时就该考虑的问题。关键做法是在 LlamaParse 和自建之间加一个**适配层（Adapter）**：

```python
class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        pass

class LlamaParseAdapter(DocumentParser):
    def parse(self, file_path):
        return LlamaParse().load_data(file_path)

class UnstructuredAdapter(DocumentParser):
    def parse(self, file_path):
        elements = partition(file_path)
        return [element_to_document(el) for el in elements]
```

这样迁移时只需要改一行工厂配置，不需要动任何业务代码。这个适配层在项目初期就应该设计好，即使当时只有 LlamaParse 一个实现。"

---

### 17.5 面试高频知识点速查

#### 一句话答案系列

| 问题 | 一句话答案 |
|------|-----------|
| 三种集成模式分别是什么？ | 模式一=封装快但黑盒，模式二=原生灵活但代码多，模式三=95% 简单+5% 复杂混合 |
| 什么时候用模式一？ | 原型/MVP、标准格式文档、不需要自定义清洗逻辑 |
| 什么时候用模式二？ | 生产环境、需要 PDF 高亮溯源、需要按 Element 类型差异化处理 |
| 模式一的最大局限性？ | Element→Document 转换过程不可控，类型标签和坐标等关键元数据被丢弃 |
| 模式二的核心优势？ | 完全控制每一步——解析参数、清洗逻辑、metadata 映射、差异化处理 |
| 为什么推荐模式三？ | 80% 的文档不需要定制清洗，只为需要的那 20% 投入额外工作量 |
| 如何判断文档复杂度？ | 文件格式→PDF 文字层检测→表格密度检测→综合评分 |
| metadata 怎么设计？ | P0 必须保留（source/page/type）、P1 强烈建议（coordinates/parent）、P2 按需 |
| 页眉页脚怎么去掉？ | 统计重复检测（阈值 60%）+ 位置启发式 + 内容模式匹配，三者组合 |
| 大规模文档解析怎么降低成本？ | 复杂度分级（95% 用 CPU fast、5% 用 GPU hi_res）+ 缓存指纹 + 增量更新 |

#### 必知关键实践速查

| 实践要点 | 说明 | 章节 |
|----------|------|------|
| **模式选择原则** | 先简单后复杂：先用模式一跑通，bad case 驱动切换到模式二 | 16.1-16.3 |
| **清洗四层模型** | 类型过滤 → 长度过滤 → 去噪（页眉页脚/OCR纠错） → 增强 | 17.2 Q3 |
| **页眉页脚三策略** | 统计重复检测 + 位置启发式 + 内容模式匹配，组合使用 | 17.2 Q3 |
| **metadata 三层设计** | P0 必须/ P1 强烈建议/ P2 按需，保留 _raw 做兜底 | 17.2 Q4 |
| **复杂度自动判定** | 扩展名 → PDF 文字层采样 → 表格密度 → 综合评分 | 17.3 Q5 |
| **解析与检索解耦** | 离线解析管线（异步）+ 在线 RAG 服务（同步），互不影响 | 17.3 Q6 |
| **Adapter 模式** | 云服务(如 LlamaParse) 和自建之间加适配层，迁移成本为零 | 17.4 Q8 |
| **文档指纹缓存** | MD5/SHA256 检测文档变更，未变的跳过解析，复用缓存 | 17.3 Q6 |

---



