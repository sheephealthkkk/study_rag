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

#### 16.2.3 模式二的可控性体现在哪里:o::o::o::o:

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

### 16.3 模式三：混合模式（最佳实践）:o:

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

### 17.2 自定义清洗与处理类:o::o:

#### Q3：模式二中，你会设计哪些自定义清洗逻辑？页眉页脚如何有效去除？（美团 / 腾讯 工程细节题）

**面试官考察点：** 这是工程落地题——是否有真实处理过"脏数据"的经验。能说出清洗策略说明真的上线过 RAG。

**回答思路（四层清洗策略，从简单到复杂）：**

**第一层：类型过滤（最基础，也最有效）**:o:

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

**第二层：长度过滤**:o:

```python
# 太短的文本大概率是噪声（OCR 碎片、单个字符、纯数字页码）
MIN_TEXT_LENGTH = 10  # 丢弃 < 10 个字符的片段

# 太长的文本可能是解析错误（整页没有做布局分析，把一页当一段）
# 这种需要标记，后续人工审核或重新解析
MAX_TEXT_LENGTH = 5000
```

**第三层：页眉页脚去除（这是高频追问）——三种策略：**

**策略 A——统计重复检测（最常用，也最稳健）：**:o:

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

**策略 B——位置启发式（精确但需要坐标信息）：**:o:

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

**策略 C——内容模式匹配（兜底）：**:o:

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

**回答思路（分级策略，从简单到复杂）：**:o:

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

**架构设计——离线解析、在线服务分离：**:o:

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

**时效性保证——增量更新 vs 全量重跑：**:o::o::o::o::o:

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

### 

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



