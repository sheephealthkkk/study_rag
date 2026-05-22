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

