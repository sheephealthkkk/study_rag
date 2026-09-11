## 十四、Partition 深入解析 —— 从参数到实战

本章对应代码文件 `partition_demo.py`。我们将围绕 `partition()` 这个 Unstructured 最核心的函数，讲解其通用参数、实战用法，以及如何利用它解决第十三章中提到的各种文档解析难题。

---

### 14.1 快速上手：最简单的 partition 调用

```python
from unstructured.partition.auto import partition

# 一行代码，解析任意格式文档
elements = partition(filename="星辰科技员工手册.md")

# elements 是一个 Element 列表，每个 Element 有类型、文本、元数据
for el in elements:
    print(f"[{type(el).__name__}] {el.text[:80]}...")
```

**输出示例：**

```
[Title] 星辰科技有限公司 — 员工手册...
[NarrativeText] 星辰科技有限公司成立于2018年，总部位于北京市...
[ListItem] · 客户第一：持续为客户创造价值...
[Title] 第一章 公司简介...
```

仅仅一行 `partition(filename=...)`，Unstructured 就自动完成了：识别文件格式 → 选择解析策略 → 检测布局 → 提取文本 → 分类为结构化元素的全部过程。

---

### 14.2 partition :o:通用参数详解

`partition()` 是 Unstructured 的顶层入口，它的参数可以归为 **六大类**：:o::o::o::o:输入源参数,核心处理策略, 语言参数, 表格与图片参数, 页面与分块控制参数, 元数据控制参数

#### 14.2.1 输入源参数 —— 你要处理什么

| 参数 | 类型 | 说明 | 使用场景 |
|------|------|------|----------|
| `filename` | `str` | 本地文件路径 | 最常见的用法 |
| `file` | `file-like object` | 二进制文件对象 | `open(file, "rb")` 返回的对象，适用于流式场景 |
| `url` | `str` | 直接传入 URL | 处理网上的文档（Unstructured 会先下载再解析） |
| `text` | `str` | 纯文本字符串 | 已有文本内容，不需要文件读取 |

```python
# 方式一：文件路径（最常用）
elements = partition(filename="报告.pdf")

# 方式二：文件对象（流式上传场景）
with open("报告.pdf", "rb") as f:
    elements = partition(file=f, content_type="application/pdf")

# 方式三：URL（需安装 requests）
elements = partition(url="https://example.com/报告.pdf")

# 方式四：纯文本（已在内存中的内容）
elements = partition(text="# 标题\n\n正文内容...", content_type="text/markdown")
```

#### 14.2.2 核心处理策略 —— 怎么处理

| 参数 | 可选值 | 含义 |
|------|--------|------|
| `strategy` | `"auto"` | 自动判断（数字原生 → fast，扫描件 → hi_res） |
| | `"fast"` | 快速模式：直接从文本层提取，不做 OCR/布局检测 |
| | `"hi_res"` | 高精度模式：渲染为图片 → 布局检测 → OCR/文本提取 |
| | `"ocr_only"` | 纯 OCR 模式：整页做 OCR，不做布局检测 |

```python
# 战略选择的核心逻辑：
#
# 你的文档是数字原生的（Word 导出、LaTeX 编译）？
#   → strategy="fast"      速度最快（秒级），质量够好
#
# 你的文档是扫描件 / 有复杂表格 / 多栏排版？
#   → strategy="hi_res"    质量最高，耗时长（分钟级）
#
# 不确定文档类型？
#   → strategy="auto"      自动判断，省心
```

**`strategy` 的选择如何应对第十三章提到的难题：**

| 难题 | 需要的 strategy | 原因 |
|------|:---:|------|
| 布局解析（多栏/侧边栏） | `hi_res` | 只有 hi_res 会做视觉布局检测，理解多栏结构 |
| 表格结构还原 | `hi_res` | fast 模式看不到"像素级别的表格线" |
| 扫描件/图片 PDF | `hi_res` | 扫描件没有文本层，fast 模式提取为空 |
| 元数据提取（坐标） | `hi_res` | 坐标信息来自布局检测阶段，fast 模式没有 |

#### 14.2.3 语言参数 —— 告诉 OCR 要识别什么语言

| 参数 | 类型 | 说明 |
|------|------|------|
| `languages` | `List[str]` | OCR 识别语言列表 |

```python
# 中文文档
partition(filename="合同.pdf", strategy="hi_res", languages=["chi_sim"])

# 中英混合文档
partition(filename="论文.pdf", strategy="hi_res", languages=["chi_sim", "eng"])

# 中日韩文档
partition(filename="多语言手册.pdf", strategy="hi_res",
          languages=["chi_sim", "jpn", "kor", "eng"])
```

**语言代码速查表：**

| 代码 | 语言 |
|------|------|
| `chi_sim` | 简体中文 |
| `chi_tra` | 繁体中文 |
| `eng` | 英语 |
| `jpn` | 日语 |
| `kor` | 韩语 |
| `fra` | 法语 |
| `deu` | 德语 |
| `spa` | 西班牙语 |

**每个难题对应的语言设置：** 中英文混排文档（如技术论文、外企合同）必须同时设置 `["chi_sim", "eng"]`，否则 OCR 会把中文识别为乱码、或把英文单词识别为无意义的中文字符序列。

#### 14.2.4 表格与图片参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pdf_infer_table_structure` | `bool` | `False` | 是否检测并还原 PDF 中的表格 |
| `extract_images_in_pdf` | `bool` | `False` | 是否提取 PDF 中嵌入的图片 |
| `image_output_dir_path` | `str` | `None` | 提取图片的保存目录 |
| `extract_image_block_types` | `List[str]` | `["Image", "Table"]` | 哪些类型的图片块要提取 |
| `skip_infer_table_types` | `List[str]` | `["pdf"]` 外的 | 跳过哪些格式的表格检测 |

```python
# 财务报告/员工手册场景 —— 表格是关键信息
elements = partition(
    filename="财务年报.pdf",
    strategy="hi_res",
    languages=["chi_sim", "eng"],
    pdf_infer_table_structure=True,   # 还原表格结构！
    # 不开启此参数，表格会被提取为散落的单元格文字
    # 开启后，表格被提取为结构化文本（HTML/Markdown 格式）
)

# 同时提取嵌入图片
elements = partition(
    filename="产品手册.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True,               # 提取图片
    image_output_dir_path="./extracted_images", # 图片存放目录
)
```

**`pdf_infer_table_structure=True` 的效果对比：**

```
未开启时（散落文字）：
  "项目 公司 个人 养老 16% 8% 医疗 9.8% 2% 失业 0.5% 0.5%"
  ↑ 数字与列头分离，LLM 无法理解行列关系

开启后（结构化输出）：
  | 项目 | 公司缴纳 | 个人缴纳 |
  |------|---------|---------|
  | 养老 | 16%     | 8%      |
  | 医疗 | 9.8%    | 2%      |
  | 失业 | 0.5%    | 0.5%    |
  ↑ Markdown 表格，LLM 原生理解
```

#### 14.2.5 页面与分块控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include_page_breaks` | `bool` | `False` | 是否在页面边界插入 `PageBreak` 元素 |
| `starting_page_number` | `int` | `1` | 起始页码（多文件拼接时有用） |
| `max_partition` | `int` | `None` | 单个元素最大字符数 |
| `chunking_strategy` | `str` | `None` | 分块策略：`"basic"` / `"by_title"` / `"by_page"` |
| `max_characters` | `int` | `None` | 与 chunking 配合，每块最大字符数 |
| `overlap` | `int` | `0` | 与 chunking 配合，块之间重叠字符数 |

```python
# 场景：一次性完成"解析 + 分块"，简化 RAG 管线
elements = partition(
    filename="员工手册.pdf",
    strategy="auto",
    languages=["chi_sim"],

    # 分块策略：在标题边界处分块
    chunking_strategy="by_title",
    # 每个 chunk 最大 800 字符
    max_characters=800,
    # 块之间重叠 80 字符
    overlap=80,
)
# 返回的 elements 已经是分好块的 Chunk 列表，可以直接送入 Embedding
# 无需再调用 split_text()！

# 场景：保留页面信息（需要精确溯源到第几页时）
elements = partition(
    filename="合同.pdf",
    include_page_breaks=True,     # 明确标记页面边界
)
```

**`chunking_strategy` 的三种策略：**

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| `"basic"` | 简单的字符数切分，与第十二章手写的 split_text 类似 | 纯文本，无结构信息 |
| `"by_title"` | 在标题（Title 元素）的位置切开，保持标题 + 正文的语义完整性 | 有明确章节结构的文档 |
| `"by_page"` | 在页面边界（PageBreak）处切开 | 需要精确页面溯源的场景 |

#### 14.2.6 元数据控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include_metadata` | `bool` | `True` | 是否在 Element 中附加坐标/页码等元数据 |
| `metadata_filename` | `str` | `None` | 覆盖元数据中的文件名（自定义命名） |
| `unique_element_ids` | `bool` | `False` | 是否为每个 Element 生成唯一 UUID |

```python
# 精确溯源场景（需要知道每个 chunk 来自第几页、什么位置）
elements = partition(
    filename="法律文书.pdf",
    strategy="hi_res",
    include_metadata=True,         # 保留全部元数据
    unique_element_ids=True,       # 每个元素有唯一 ID
)

# 查看元数据
for el in elements[:3]:
    meta = el.metadata
    print(f"类型: {type(el).__name__}")
    print(f"页码: {meta.page_number}")
    print(f"坐标: ({meta.coordinates.x}, {meta.coordinates.y})")
    print(f"尺寸: {meta.coordinates.width} x {meta.coordinates.height}")
    print()
```

---

### 14.3 PDF 实战：三种策略的对比

`partition_demo.py` 中的 `demo_pdf_strategies()` 函数演示了对同一个 PDF 使用三种不同策略的差异。

#### 策略 1：fast —— 直接从文本层提取

```python
elements = partition_pdf(
    filename="员工手册.pdf",
    strategy="fast",
)
```

**处理路径：**

```
  PDF 文件 → [PDF 解析器] → 读取文本层 → 输出 Elements
                              ↑
                         不渲染为图片
                         不做布局检测
                         不做 OCR
```

**fast 模式能处理的难题：**

| 难题 | 能否处理 | 说明 |
|------|:---:|------|
| 数字原生 PDF 的文本提取 | ✅ | 这是 fast 模式的设计目标 |
| 扫描件 | ❌ | 扫描件没有文本层，fast 模式读出的全是空 |
| 表格结构还原 | ❌ | 没有布局检测，表格就是散乱文字 |
| 元数据（坐标） | ❌ | 没有渲染，无法获取像素级坐标 |
| 多栏布局 | ❌ | 读取顺序可能错乱 |

#### 策略 2：hi_res —— 渲染+检测+识别

```python
elements = partition_pdf(
    filename="员工手册.pdf",
    strategy="hi_res",
    languages=["chi_sim", "eng"],
    pdf_infer_table_structure=True,   # 表格结构检测
    include_metadata=True,            # 保留坐标等元数据
)
```

**处理路径：**

```
  PDF 文件
    │
    ▼
  [Step 1] 渲染 (pdf2image + poppler)
    每页 → 高分辨率 PNG 图片（如 300 DPI）
    │
    ▼
  [Step 2] 布局检测 (detectron2 / YOLO 视觉模型)
    识别每页的区域类型：Title / Text / Table / Image / List
    输出：每个区域的 (x, y, width, height) 坐标 + 类型标签
    │
    ▼
  [Step 3] 文本提取
    文本区域 → Tesseract/PaddleOCR（如果是扫描件）
           或 PyPDFium（如果是数字原生，从坐标对应的文本层提取）
    表格区域 → 表格结构识别 → Markdown/HTML 格式
    │
    ▼
  [Step 4] 组装 Elements
    按照阅读顺序排列（根据 y 坐标从上到下，同行的按 x 坐标从左到右）
    剔除页眉页脚（坐标在页面顶部/底部、且多页重复的元素）
    输出带有完整元数据的 Element 列表
```

**hi_res 模式能处理的难题：**

| 难题 | 能否处理 | 具体机制 |
|------|:---:|------|
| 扫描件 | ✅ | OCR 引擎读取图片中的文字 |
| 表格结构还原 | ✅ | `pdf_infer_table_structure=True`，视觉模型+结构化还原 |
| 元数据（坐标） | ✅ | 每一步都有精确的像素坐标 |
| 多栏布局 | ✅ | 布局检测模型识别栏边界，正确排序 |
| 页眉页脚污染 | ✅ | 坐标位置 + 多页重复检测 → 自动去除 |
| 图片中的文字 | ✅ | OCR 覆盖所有像素区域 |

#### 策略 3：auto —— 自动选择最优策略

```python
elements = partition_pdf(
    filename="员工手册.pdf",
    strategy="auto",               # ← Unstructured 自己判断
    languages=["chi_sim", "eng"],
    pdf_infer_table_structure=True,
)
```

**auto 的判断逻辑：**

```
  auto 策略的内部决策树：

  文档有文本层？
    ├─ 是 → 用 fast 模式（速度快）
    │      （大多数 Word/LaTeX 导出的 PDF 都在此列）
    │
    └─ 否（文本层为空）→ 用 hi_res 模式（需要 OCR）
           （扫描件、某些 CAD 导出的 PDF）

  实际项目中推荐使用 auto：
    - 数字原生 PDF 自动享受 fast 的高速度
    - 扫描件自动切换到 hi_res 的高精度
    - 混合文档库不用逐个判断文档类型
```

---

### 14.4 图片 OCR 实战

对于图片文件（`.png` / `.jpg` / `.tiff` 等），Unstructured 使用 `partition_image`：

```python
from unstructured.partition.image import partition_image

elements = partition_image(
    filename="合同截图.png",
    strategy="hi_res",                # 图片只能用 hi_res 或 ocr_only
    languages=["chi_sim", "eng"],      # OCR 语言
    include_metadata=True,
)
```

**图片 OCR 的两种模式：**

| 模式 | 参数 | 行为 |
|------|------|------|
| 整页 OCR | `ocr_mode="entire_page"` | 对整张图片做一次 OCR |
| 逐块 OCR | `ocr_mode="individual_blocks"` | 先用布局检测分割区域，对每个区域单独做 OCR |

逐块 OCR 更精确，因为不同区域的文字方向、字体大小、语言可能不同。但整页 OCR 更快（一次调用 vs 多次调用）。

**图片 OCR 处理的关键难题与对应字段：**

```
难题：中英文混排
  解决方案：languages=["chi_sim", "eng"]
  Unstructured 会把语言列表传给 Tesseract，Tesseract 会同时加载
  中英文识别模型。但注意：语言越多，OCR 越慢。

难题：低分辨率/模糊图片
  解决方案：hi_res 模式默认渲染为 300 DPI，比原图更清晰。
          如果原图实在太模糊，可能需要预处理（去噪/二值化），
          Unstructured 本身不提供图像预处理。

难题：手写文字
  解决方案：Tesseract 有手写识别模型（eng_hnd 等），但准确率不如印刷体。
          生产环境可能需要结合专业手写 OCR 服务。
```

---

### 14.5 元数据深入分析

`partition_demo.py` 的 `demo_metadata_inspection()` 函数展示了每个 Element 携带的完整元数据。

#### 元数据字段全景

```
每个 Element 的 metadata 包含以下字段（实际可用字段取决于 strategy）：

┌─────────────────────────────────────────────────────────────────┐
│  来源信息                                                        │
│  ├─ filename: "员工手册.pdf"          # 原始文件名                │
│  ├─ filetype: "application/pdf"       # MIME 类型                 │
│  └─ file_directory: "/data/docs/"     # 文件所在目录              │
├─────────────────────────────────────────────────────────────────┤
│  定位信息                                                        │
│  ├─ page_number: 3                    # 页码（从 1 开始）         │
│  ├─ coordinates: {                    # 精确位置                  │
│  │     "x": 72.0,                     #   左上角 x (单位: 点)     │
│  │     "y": 340.5,                    #   左上角 y (单位: 点)     │
│  │     "width": 468.0,                #   区域宽度                │
│  │     "height": 18.5,                #   区域高度                │
│  │   }                                                           │
│  └─ coordinate_system: "PixelSpace"   # 坐标系统                  │
├─────────────────────────────────────────────────────────────────┤
│  层级信息                                                        │
│  ├─ parent_id: "c9d2f..."             # 父元素 UUID               │
│  ├─ category_depth: 2                 # 层级深度                  │
│  └─ is_continuation: False            # 是否为前一个元素的延续    │
├─────────────────────────────────────────────────────────────────┤
│  格式信息                                                        │
│  ├─ emphasized_text_contents:         # 被强调的文字内容          │
│  │     ["1个工作日", "80%"]                                       │
│  ├─ emphasized_text_tags: ["b", "b"]  # 强调标签 <b> / <i>        │
│  ├─ link_texts: ["HR部门"]            # 超链接文字                │
│  └─ link_urls: ["mailto:hr@..."]      # 超链接 URL                │
├─────────────────────────────────────────────────────────────────┤
│  语言信息                                                        │
│  └─ languages: ["chi_sim"]            # 检测到的文本语言          │
└─────────────────────────────────────────────────────────────────┘
```

#### 元数据在 RAG 各环节的价值

在之前章节中我们反复强调"元数据很重要"，这里用具体场景说明为什么：

```
场景 1：精确定位
  用户问："第 5 页的报销标准是多少？"
  检索时用 page_number 过滤 → 只检索第 5 页的 chunk
  回答时标注来源 → "根据员工手册第 5 页，一线城市住宿标准为 500 元/天"

场景 2：前端高亮
  检索到 chunk 后，用 coordinates 在 PDF 查看器中高亮对应的文字区域
  用户体验：点击引用 → PDF 自动滚动到对应位置 → 高亮对应的文字块

场景 3：层级扩展
  检索到一个 ListItem "· 工作日加班按 1.5 倍计算"
  通过 parent_id 找到它的父标题 "9.2 加班费计算"
  将父标题补入 chunk 的上下文 → 检索时语义更完整

场景 4：按日期/来源过滤
  metadata.filetype → "只检索 Word 格式的部门制度，不要 PPT 里的简报"
  metadata.last_modified → "只检索 2025 年 1 月之后更新的政策"

场景 5：强调内容加权
  metadata.emphasized_text_contents = ["1个工作日"]
  在检索打分时，包含"强调内容"的 chunk 相关性 +0.1 的权重
```

---

### 14.6 完整管线：从 partition 到 RAG-ready

结合第十二章的手写 RAG 管线，下面展示 Unstructured + 手写管线的完整流程：

```
═══════════════════════════════════════════════════════════════════════
         Unstructured partition → 手写 RAG 管线的完整流程
═══════════════════════════════════════════════════════════════════════

  Step 1: partition() 解析文档
  ┌─────────────────────────────────────────────────────────────┐
  │ elements = partition(                                       │
  │     filename="员工手册.pdf",                                │
  │     strategy="auto",                                       │
  │     languages=["chi_sim", "eng"],                           │
  │     pdf_infer_table_structure=True,                         │
  │     include_metadata=True,                                  │
  │ )                                                           │
  │                                                             │
  │ 输出: [Title, NarrativeText, Table, ListItem, ...]          │
  │ 每个 Element 都带类型标签 + 完整元数据                       │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  Step 2: 转换为统一 Document（适配手写管线）
  ┌─────────────────────────────────────────────────────────────┐
  │ documents = []                                              │
  │ for el in elements:                                         │
  │     doc = Document(                                         │
  │         page_content=el.text,                                │
  │         metadata={                                          │
  │             "source": el.metadata.filename,                  │
  │             "page_number": el.metadata.page_number,          │
  │             "element_type": type(el).__name__,               │
  │             "coordinates": el.metadata.coordinates,          │
  │         }                                                   │
  │     )                                                       │
  │     documents.append(doc)                                   │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  Step 3: (可选) Text Splitter —— 对过长的 Element 再切分
  ┌─────────────────────────────────────────────────────────────┐
  │ # Table 和 Title 类型的 Element 通常不需要再切分             │
  │ # NarrativeText 如果很长，按 chunk_size=800 切分             │
  │ chunks = []                                                 │
  │ for doc in documents:                                       │
  │     if len(doc.page_content) > 800:                         │
  │         chunks.extend(split_text(doc, chunk_size=800))       │
  │     else:                                                   │
  │         chunks.append(doc)                                  │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
  Step 4: Embedding + FAISS 索引 + 检索 + LLM 生成
  ┌─────────────────────────────────────────────────────────────┐
  │ # 与第十二章的管线完全一致                                    │
  │ store = build_vector_store(chunks)                           │
  │ answer = rag_pipeline("请事假需要提前多久？", store)           │
  └─────────────────────────────────────────────────────────────┘
```

**关键差异：** 相比于第十二章用纯文本 `split_text()` 直接从 Markdown 文件切分，引入 Unstructured 之后：
- Table 元素保持了表格结构（不会被切散）
- Title 元素可以作为 chunk 的标题补充到 metadata
- 坐标/页码元数据支持前端高亮和精确溯源
- 页眉页脚等噪声被自动清理

---

### 14.7 总结：partition 的价值与局限性

**核心价值：**

| 能力 | 说明 |
|------|------|
| **格式无关** | 同样的 API 调用处理 PDF/Word/HTML/图片…… 不需要为每种格式写一套代码 |
| **结构理解** | 不只是提取文字，还知道每段文字是什么类型（标题/正文/表格/列表） |
| **元数据丰富** | 坐标、页码、层级、字体、链接 —— 信息的密度远超纯文本 |
| **噪声过滤** | 自动去除页眉页脚、空白段落、乱码字符 |
| **与 RAG 框架深度集成** | LangChain 和 LlamaIndex 都原生支持 Unstructured |

**局限性（需要知道）：**

| 局限 | 影响 | 缓解方式 |
|------|------|----------|
| **hi_res 速度慢** | 处理大批量 PDF 时耗时较长 | 对数字原生 PDF 用 `strategy="fast"`，仅对扫描件用 `hi_res` |
| **OCR 准确率受限于底层引擎** | Tesseract 对中文字符的识别准确率不如 PaddleOCR | 中文场景配置 PaddleOCR 作为 OCR 代理 |
| **复杂表格仍可能出错** | 合并单元格、跨页表格的还原偶尔失败 | 关键数据表格建议人工抽查 |
| **需要额外系统依赖** | hi_res 需要 poppler + Tesseract/detectron2 | 部署时在 Dockerfile 中预装这些依赖 |
| **Windows 兼容性** | 部分原生依赖（detectron2）在 Windows 上不稳定 | 开发用 `fast` 模式验证逻辑，生产部署在 Linux 上 |


---

