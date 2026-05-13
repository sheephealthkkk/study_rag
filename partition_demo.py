"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Unstructured Partition 实战演示 —— PDF & 图片处理                   ║
║                                                                              ║
║  依赖：pip install "unstructured[pdf,image]"                                  ║
║        pip install pytesseract          (OCR)                                ║
║        pip install pdf2image            (PDF→图片渲染，hi_res 需要)           ║
║                                                                              ║
║  系统要求：                                                                    ║
║        Tesseract OCR 引擎需单独安装：                                          ║
║        - Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-chi-sim        ║
║        - Mac:    brew install tesseract                                      ║
║        - Win:    下载安装 https://github.com/UB-Mannheim/tesseract/wiki      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os
import sys
import json
from typing import List

# ─── 修复 Windows 终端 UTF-8 编码 ─────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# ============================================================================
# 第一部分：Partition 通用参数详解
# ============================================================================
"""
partition() 是 Unstructured 的核心入口函数。它的通用参数分为以下几组：

┌─────────────────────────────────────────────────────────────────────────────┐
│                           partition() 通用参数总览                            │
├──────────────────┬──────────────────────────────────────────────────────────┤
│ 📁 输入源参数                                                                 │
│   filename       │ 文件路径 (str)。支持 .pdf / .docx / .html / .png ...      │
│   file           │ 类文件对象 (file-like object)，如 open() 返回的对象         │
│   url            │ 直接传入 URL 下载并解析（需安装 requests）                  │
│   text           │ 直接传入文本字符串进行解析                                  │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ 🔧 核心处理策略                                                               │
│   strategy       │ "auto" / "fast" / "hi_res" / "ocr_only"                  │
│                  │ 决定了文档的处理深度（详见十三章）                           │
│   languages      │ OCR 识别语言列表。中文: ['chi_sim']                        │
│                  │ 多语言: ['chi_sim', 'eng']                                │
│   content_type   │ 强制指定内容类型（如 "image/png"），防止自动检测错误         │
│   encoding       │ 文本文件编码，默认 "utf-8"                                 │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ 📊 表格与图片处理                                                             │
│   pdf_infer_table_structure │ 是否自动识别并还原 PDF 中的表格结构              │
│   extract_images_in_pdf     │ 是否提取 PDF 中嵌入的图片                        │
│   image_output_dir_path     │ 提取图片的保存目录                               │
│   extract_image_block_types │ 哪些类型的图片块要提取                            │
│   skip_infer_table_types    │ 跳过表格检测的格式列表                           │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ 📄 页面与分块控制                                                             │
│   include_page_breaks       │ 是否在页面边界插入 PageBreak 元素               │
│   starting_page_number      │ 起始页码（多文件拼接时有用）                     │
│   max_partition             │ 单个元素的最大字符数（可选）                      │
│   chunking_strategy         │ 分块策略："basic" / "by_title" / "by_page"     │
│   max_characters            │ 与 chunking_strategy 配合，每块最大字符数        │
│   overlap                   │ 与 chunking_strategy 配合，块之间重叠字符数      │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ 🏷️ 元数据控制                                                                │
│   include_metadata          │ 是否输出完整的坐标/页码等元数据                  │
│   metadata_filename         │ 覆盖元数据中的文件名（自定义命名）               │
│   unique_element_ids        │ 是否自动生成唯一 UUID 给每个元素                │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ 🔍 OCR 专用                                                                  │
│   ocr_languages             │ OCR 识别语言（与 languages 语义相同）            │
│   ocr_mode                  │ "entire_page" / "individual_blocks"            │
│                             │ 整页 OCR vs 逐块 OCR（后者更精确）              │
└──────────────────┴──────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 第二部分：PDF 实战 —— 三种策略的对比
# ============================================================================

def demo_pdf_strategies(pdf_path: str):
    """
    演示对同一个 PDF 使用 fast / hi_res / auto 三种策略的效果差异。

    策略选择的核心逻辑：
      fast   → 直接从 PDF 文本层提取文字，不做 OCR，不做布局检测
               → 适用于数字原生 PDF（如 Word 导出、LaTeX 编译的 PDF）
               → 速度快（秒级），但不能处理扫描件
               → 表格还原能力弱

      hi_res → 把每页渲染成高分辨率图片，用视觉模型识别文本区域
               → 适用于扫描件、图片型 PDF、复杂布局（多栏/表格）
               → 速度慢（每页需渲染+检测+OCR），但质量最高
               → 能精准还原表格、标题层级

      auto   → 自动检测：数字PDF → fast，扫描件 → hi_res
               → 混合文档库的首选，让 Unstructured 自己判断
    """
    from unstructured.partition.pdf import partition_pdf
    import time

    print("=" * 70)
    print("  PDF Partition 策略对比演示")
    print(f"  文件: {pdf_path}")
    print("=" * 70)

    # ─── 策略 1: fast ──────────────────────────────────────────────
    print("\n  [策略 1] fast 模式 —— 直接从文本层提取")
    start = time.time()
    elements_fast = partition_pdf(
        filename=pdf_path,
        strategy="fast",
        # fast 模式下不需要 OCR 语言参数
    )
    elapsed_fast = time.time() - start
    print(f"  耗时: {elapsed_fast:.2f}s")
    print(f"  元素数量: {len(elements_fast)}")
    print(f"  元素类型分布:")
    _print_element_type_counts(elements_fast)

    # 展示几个元素示例
    _print_sample_elements(elements_fast, "fast (快速)", max_items=4)

    # ─── 策略 2: hi_res ─────────────────────────────────────────────
    # 注意：hi_res 需要 pdf2image + 系统安装的 poppler
    #       以及检测模型（如 detectron2 / yolo）
    print("\n\n  [策略 2] hi_res 模式 —— 渲染为图片 + 布局检测 + OCR")
    print("  (如果本机未安装 detectron2/poppler/Tesseract，此步可能报错)")
    try:
        start = time.time()
        elements_hires = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            # 语言设置 —— 告诉 OCR 引擎要识别哪些语言
            languages=["chi_sim", "eng"],        # 简体中文 + 英文
            # 表格结构检测 —— 自动识别表格并还原为结构化格式
            pdf_infer_table_structure=True,
            # 提取嵌入图片 —— 如果你想同时得到 PDF 中的图片
            extract_images_in_pdf=False,          # 演示时不提取
            # 元数据——包含坐标信息
            include_metadata=True,
        )
        elapsed_hires = time.time() - start
        print(f"  耗时: {elapsed_hires:.2f}s")
        print(f"  元素数量: {len(elements_hires)}")
        print(f"  元素类型分布:")
        _print_element_type_counts(elements_hires)
        _print_sample_elements(elements_hires, "hi_res (高精度)", max_items=4)
    except Exception as e:
        print(f"  [WARN] hi_res 执行失败: {e}")
        print(f"  可能原因：未安装 poppler / Tesseract / detectron2")
        print(f"  安装指南:")
        print(f"    Windows: choco install poppler tesseract")
        print(f"    Mac:     brew install poppler tesseract")
        print(f"    Linux:   sudo apt install poppler-utils tesseract-ocr")
        elements_hires = []

    # ─── 策略 3: auto ───────────────────────────────────────────────
    print("\n\n  [策略 3] auto 模式 —— 自动判断最佳策略")
    start = time.time()
    elements_auto = partition_pdf(
        filename=pdf_path,
        strategy="auto",
        languages=["chi_sim", "eng"],
        pdf_infer_table_structure=True,
    )
    elapsed_auto = time.time() - start
    print(f"  耗时: {elapsed_auto:.2f}s")
    print(f"  实际采用的策略: (auto 会自动选择 fast 或 hi_res)")
    print(f"  元素数量: {len(elements_auto)}")
    _print_sample_elements(elements_auto, "auto (自动)", max_items=3)

    # ─── 对比总结 ──────────────────────────────────────────────────
    print("\n\n  " + "=" * 60)
    print("  三种策略对比总结")
    print("  " + "=" * 60)
    print(f"  {'策略':<10} {'耗时':<10} {'元素数':<10} {'适用场景'}")
    print(f"  {'-'*50}")
    print(f"  {'fast':<10} {elapsed_fast:.2f}s{'':<5} {len(elements_fast):<10} 数字原生 PDF")
    if elements_hires:
        print(f"  {'hi_res':<10} {elapsed_hires:.2f}s{'':<5} {len(elements_hires):<10} 扫描件/复杂布局")
    print(f"  {'auto':<10} {elapsed_auto:.2f}s{'':<5} {len(elements_auto):<10} 混合文档库")


# ============================================================================
# 第三部分：图片 OCR —— 将图片中的文字提取为结构化元素
# ============================================================================

def demo_image_ocr(image_path: str):
    """
    演示对图片文件使用 partition 做 OCR 提取。

    Unstructured 支持直接传入 .png / .jpg 等图片文件。
    内部会调用 Tesseract（或 PaddleOCR）做文字识别，并用布局检测
    模型区分文字区域和非文字区域（如图表、照片）。

    关键参数：
      - strategy="hi_res" 或 "ocr_only"（图片只能用这两个）
      - languages:  告诉 OCR 引擎图片中的文字是什么语言
      - ocr_mode:   "entire_page" vs "individual_blocks"
    """
    from unstructured.partition.image import partition_image

    if not os.path.exists(image_path):
        print(f"  [WARN] 图片文件不存在: {image_path}")
        print(f"  请准备一张包含文字的图片（截图即可）用于测试。")
        return

    print("=" * 70)
    print("  图片 OCR 提取演示")
    print(f"  文件: {image_path}")
    print("=" * 70)

    # ─── 基础 OCR ──────────────────────────────────────────────────
    print("\n  [模式 A] 基础 OCR —— 整页识别")
    try:
        elements_img = partition_image(
            filename=image_path,
            strategy="hi_res",                   # 图片必须用 hi_res 或 ocr_only
            languages=["chi_sim", "eng"],         # OCR 识别语言
            include_metadata=True,               # 保留坐标等元数据
        )
        print(f"  元素数量: {len(elements_img)}")
        _print_sample_elements(elements_img, "图片OCR", max_items=5)
    except Exception as e:
        print(f"  [WARN] 图片 OCR 执行失败: {e}")

    # ─── 带 chunking 的 OCR（一步到位）─────────────────────────────
    print("\n\n  [模式 B] OCR + 分块 —— 直接在 partition 内分块")
    try:
        elements_with_chunks = partition_image(
            filename=image_path,
            strategy="hi_res",
            languages=["chi_sim", "eng"],
            # chunking 参数 —— partition 内置的分块能力
            chunking_strategy="by_title",         # 按标题边界分块
            max_characters=800,                   # 每个 chunk 最大 800 字符
            overlap=80,                           # 块之间重叠 80 字符
            include_metadata=True,
        )
        print(f"  分块后的元素数量: {len(elements_with_chunks)}")
        _print_sample_elements(
            elements_with_chunks, "图片OCR+分块", max_items=4
        )
    except Exception as e:
        print(f"  [WARN] OCR+分块执行失败: {e}")


# ============================================================================
# 第四部分：元数据深入分析 —— 每个 Element 携带了什么信息
# ============================================================================

def demo_metadata_inspection(pdf_path: str):
    """
    深入查看每个 Element 的元数据，理解 Unstructured 到底"读"到了什么。

    Element 元数据中的关键字段：
      · page_number  → 原始页码
      · coordinates  → 元素在页面上的精确位置 (x, y, width, height)
      · parent_id    → 所属父元素的 ID（标题与正文的层级关系）
      · filetype     → 原始文件类型
      · languages    → 检测到的文本语言
      · emphasized_text_contents → 被强调（加粗/倾斜）的文字内容
    """
    from unstructured.partition.auto import partition

    print("=" * 70)
    print("  元数据深入分析")
    print("=" * 70)

    try:
        # 用 auto 模式，它会自动选 fast（如果 PDF 是数字原生的）
        elements = partition(
            filename=pdf_path,
            strategy="auto",
            languages=["chi_sim", "eng"],
            include_metadata=True,
            unique_element_ids=True,       # 给每个元素生成唯一 UUID
            include_page_breaks=True,      # 保留 PageBreak 元素
        )

        # 只取前几个做详细分析
        for i, el in enumerate(elements[:6]):
            print(f"\n  ┌── Element[{i}] ──────────────────────────────")
            print(f"  │ 类型: {type(el).__name__}")
            print(f"  │ ID:   {el.id}")
            print(f"  │ 文本: {repr(el.text[:100])}...")
            print(f"  │ 元数据:")

            meta = el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else vars(el.metadata)
            # 过滤掉值为 None 的字段，保持输出简洁
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if value is not None and value != [] and value != {}:
                        # 格式化坐标信息
                        if key == "coordinates" and value:
                            c = value
                            print(f"  │   {key}: (x={c.get('x','?')}, y={c.get('y','?')}, "
                                  f"w={c.get('width','?')}, h={c.get('height','?')})")
                        elif key == "emphasized_text_contents" and value:
                            print(f"  │   {key}: {value} (加粗/强调的文字)")
                        elif key == "link_urls" and value:
                            print(f"  │   {key}: {value}")
                        else:
                            print(f"  │   {key}: {value}")
            print(f"  └{'─' * 50}")

    except Exception as e:
        print(f"  [WARN] 元数据分析失败: {e}")


# ============================================================================
# 第五部分：实用场景 —— 保留表格结构
# ============================================================================

def demo_table_extraction(pdf_path: str):
    """
    演示如何从 PDF 中提取表格并保留其结构化信息。

    这在企业 RAG 中极其重要 —— 财务报告、员工手册、产品规格表中
    通常有大量表格数据。如果表格被拆散成散落文字，LLM 就无法正确理解。

    pdf_infer_table_structure=True 会触发：
      1. 检测页面上的表格区域
      2. 还原表格的行列结构
      3. 输出为 HTML/Markdown 格式（而非纯文本）
    """
    from unstructured.partition.auto import partition

    print("=" * 70)
    print("  表格结构提取演示")
    print("=" * 70)

    try:
        elements = partition(
            filename=pdf_path,
            strategy="auto",
            languages=["chi_sim", "eng"],
            pdf_infer_table_structure=True,   # 关键参数！
            include_metadata=True,
        )

        # 筛选出 Table 类型的元素
        tables = [el for el in elements if type(el).__name__ == "Table"]
        print(f"  共检测到 {len(tables)} 个表格")
        print()

        for i, table in enumerate(tables[:3]):  # 只展示前3个
            print(f"  ┌── Table[{i}] ──────────────────────────────")
            print(f"  │ 文本内容 (已转为结构化格式):")
            # 表格文本通常以 | 分隔列，\n 分隔行
            for line in table.text.split("\n")[:10]:
                print(f"  │  {line}")
            print(f"  └{'─' * 50}")

            # 如果表格元数据中包含 HTML 表示
            meta = table.metadata.to_dict() if hasattr(table.metadata, "to_dict") else {}
            if meta.get("text_as_html"):
                print(f"  │ HTML 格式 (可被 LLM 更好理解):")
                print(f"  │ {meta['text_as_html'][:300]}...")
                print()

    except Exception as e:
        print(f"  [WARN] 表格提取失败: {e}")


# ============================================================================
# 辅助函数
# ============================================================================

def _print_element_type_counts(elements) -> None:
    """统计并打印元素类型分布"""
    from collections import Counter
    types = Counter(type(el).__name__ for el in elements)
    for t, count in types.most_common():
        print(f"    {t}: {count}")


def _print_sample_elements(elements, label: str, max_items: int = 4) -> None:
    """打印几个元素样本，便于直观对比"""
    print(f"\n  [{label}] 元素样本:")
    shown = 0
    for el in elements:
        if shown >= max_items:
            break
        text_preview = el.text[:100].replace("\n", "\\n")
        el_type = type(el).__name__
        # 标出页码
        page = el.metadata.page_number if el.metadata and el.metadata.page_number else "?"
        print(f"    [{el_type}] (第{page}页) {text_preview}...")
        shown += 1


# ============================================================================
# 第六部分：主函数 —— 选择运行哪个演示
# ============================================================================

if __name__ == "__main__":
    # 测试文件路径（使用之前创建的员工手册）
    handbook_pdf = "星辰科技员工手册.pdf"     # 如果有 PDF 版本
    handbook_md  = "星辰科技员工手册.md"      # Markdown 版本（肯定存在）
    test_image   = "test_image.png"           # 图片测试（需要自己准备）

    print("=" * 70)
    print("  Unstructured Partition 功能演示")
    print("=" * 70)
    print()
    print("  可用的演示:")
    print("    1. 三种策略对比 (fast / hi_res / auto)")
    print("    2. 图片 OCR 提取")
    print("    3. 元数据深入查看")
    print("    4. 表格结构提取")
    print()
    print("  Unstructured 版本: ", end="")
    try:
        import unstructured
        print(unstructured.__version__)
    except ImportError:
        print("未安装")

    # ─── 根据文件存在情况自动选择演示 ──────────────────────────────
    # 优先用 Markdown 文件做 fast 演示（任何环境都能跑）
    # PDF 演示需要额外的系统依赖

    print("\n" + "=" * 70)
    print(f"  第一步：基础概念验证（用 Markdown 文件测试 partition）")
    print("=" * 70)
    try:
        from unstructured.partition.auto import partition
        elements_md = partition(filename=handbook_md, strategy="auto")
        print(f"  Markdown 文件元素数: {len(elements_md)}")
        _print_element_type_counts(elements_md)
        _print_sample_elements(elements_md, "Markdown", max_items=5)
    except Exception as e:
        print(f"  [WARN] {e}")

    # 如果有 PDF 文件，跑完整的 PDF 演示
    if os.path.exists(handbook_pdf):
        print(f"\n  检测到 PDF 文件，运行完整演示...")
        demo_pdf_strategies(handbook_pdf)
        print("\n")
        demo_metadata_inspection(handbook_pdf)
        print("\n")
        demo_table_extraction(handbook_pdf)
    else:
        print(f"\n  [INFO] 未找到 PDF 文件 '{handbook_pdf}'")
        print(f"  如果要测试 PDF 功能，请将员工手册转为 PDF 放在同目录下。")
        print(f"  或安装 pandoc 后运行: pandoc 星辰科技员工手册.md -o 星辰科技员工手册.pdf")

    # 如果有测试图片，跑 OCR 演示
    if os.path.exists(test_image):
        print("\n")
        demo_image_ocr(test_image)
    else:
        print(f"\n  [INFO] 未找到测试图片 '{test_image}'")
        print(f"  准备一张含中文文字的截图命名为 test_image.png 即可测试。")
