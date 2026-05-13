"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              手写 RAG 完整实现 —— 零框架依赖（纯 Python）                     ║
║                                                                              ║
║  依赖清单：                                                                   ║
║    pip install openai numpy faiss-cpu                                        ║
║                                                                              ║
║  本文档用途：                                                                 ║
║    1. 学习 RAG 的每一个底层环节                                               ║
║    2. 理解数据从原始文档 → 向量 → 检索 → LLM 回答的全链路变化                  ║
║    3. 不依赖 LangChain / LlamaIndex，用最少的代码看清原理                      ║
║                                                                              ║
║  使用前请设置环境变量：                                                        ║
║    export OPENAI_API_KEY="sk-..."            # Linux/Mac                     ║
║    set OPENAI_API_KEY=sk-...                 # Windows CMD                   ║
║    $env:OPENAI_API_KEY="sk-..."              # Windows PowerShell            ║
║                                                                              ║
║  如果您使用硅基流动(SiliconFlow)等中转API：                                     ║
║    需额外设置 OPENAI_BASE_URL 和 OPENAI_API_KEY                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import math
import json
import sys
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field

# ─── 修复 Windows 终端 UTF-8 输出乱码 ─────────────────────────────
# Windows 的 Git Bash / CMD 默认使用 GBK 编码，直接打印中文和 Unicode
# 特殊字符（如 ╔ ═ █）会显示为乱码。这里强制 stdout 使用 UTF-8。
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        # Python < 3.7 没有 reconfigure
        pass

import numpy as np

# ============================================================================
# 第一部分：Embedding 工具层 —— 文字与向量的桥梁
# ============================================================================

# ─── OpenAI 客户端初始化 ─────────────────────────────────────────────
# 说明：这里使用 openai 库，但它本质上是 HTTP 调用，不依赖任何 RAG 框架。
#       如果你的环境变量 OPENAI_API_KEY 未设置，下面的代码会在调用时报错。
#       国内用户如果使用硅基流动(SiliconFlow)、智谱等中转服务，
#       请同时设置 OPENAI_BASE_URL 为对应地址。

from openai import OpenAI

# 初始化客户端。base_url 如果不传则默认连接 OpenAI 官方。
# 国内中转示例：base_url="https://api.siliconflow.cn/v1"
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """延迟创建 OpenAI 客户端，避免导入时就必须设置 API key"""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if not api_key:
            raise RuntimeError(
                "请设置环境变量 OPENAI_API_KEY。\n"
                "  export OPENAI_API_KEY='sk-...'  (Linux/Mac)\n"
                "  $env:OPENAI_API_KEY='sk-...'   (PowerShell)"
            )
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


# ─── 单条文本向量化 ────────────────────────────────────────────────
def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
    dims: Optional[int] = None,
) -> np.ndarray:
    """
    将一段文本转化为向量（浮点数数组）。

    参数：
        text:  要向量化的文本
        model: 使用的 embedding 模型
               "text-embedding-3-small" → 默认 512 维，可扩展至 1536 维
               "text-embedding-3-large" → 默认 1024 维，可扩展至 3072 维
        dims:  输出维度（可选）。不传则使用模型默认维度。
               OpenAI 的 MRL 技术支持在较小维度下保留大部分精度。

    返回：
        np.ndarray，shape = (dims,)，dtype = float32

    原理说明：
        Embedding 模型（Transformer Encoder）将文本分词后，通过多层
        Self-Attention 提取语义特征，最后做 Pooling 得到一个固定长度的
        浮点数数组。这个数组的每个数值本身无人类可读含义，但两个向量在
        空间中的"方向"接近程度反映了两段文字的语义相似度。

    示例：
        >>> vec = embed_text("事假需提前1个工作日申请")
        >>> vec.shape
        (512,)
        >>> vec[:5]  # 前 5 个维度的值（每次调用结果不同——浮点数精度）
        array([ 0.0234, -0.0147,  0.0089, -0.0051,  0.0163], dtype=float32)
    """
    client = _get_client()
    kwargs = {"model": model, "input": text}
    if dims is not None:
        kwargs["dimensions"] = dims

    response = client.embeddings.create(**kwargs)
    # response.data[0].embedding 是一个 Python list[float]
    emb = response.data[0].embedding
    return np.array(emb, dtype=np.float32)


# ─── 批量文本向量化 ─────────────────────────────────────────────────
def embed_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
    dims: Optional[int] = None,
) -> np.ndarray:
    """
    一次性将多条文本向量化。

    为什么要做批处理？
      - API 调用有网络往返开销。把多条文本放在一次请求中，可以大幅减少
        总耗时（20条一起传 vs 20次单条传，可能快 10 倍以上）。
      - 离线索引构建阶段，成千上万个 chunk 如果逐条调用会很慢。

    参数：
        texts: 文本列表，每条文本会被独立向量化
        model: 同上
        dims:  同上

    返回：
        np.ndarray，shape = (len(texts), dims)，dtype = float32
        第 i 行对应 texts[i] 的向量。

    示例：
        >>> chunks = ["文本A", "文本B", "文本C"]
        >>> vecs = embed_batch(chunks, dims=512)
        >>> vecs.shape
        (3, 512)
    """
    client = _get_client()
    kwargs = {"model": model, "input": texts}
    if dims is not None:
        kwargs["dimensions"] = dims

    response = client.embeddings.create(**kwargs)
    # response.data 按输入顺序返回，每个包含 embedding 列表
    vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
    return np.stack(vectors)  # 堆叠成 (N, D) 矩阵


# ============================================================================
# 第二部分：Document Loader —— 把五花八门的文件统一成标准 Document
# ============================================================================

@dataclass
class Document:
    """
    RAG 中的标准文档对象。无论原始文件是 PDF、Markdown 还是纯文本，
    最终都统一成这个结构，后续的切分、向量化、检索都基于它。

    字段说明：
        page_content: 文档的正文内容（已提取为纯文本）
        metadata:     元信息字典，包含来源、页码、日期等
                      metadata 在检索溯源阶段非常关键！
    """
    page_content: str
    metadata: Dict[str, str] = field(default_factory=dict)


# ─── 各种格式的加载函数 ─────────────────────────────────────────────
# 注意：这里的加载函数仅为教学演示。生产环境中的 PDF 解析、DOCX 解析
#       远比这里复杂（详见教程正文中的讲解）。

def _load_txt(file_path: str) -> Document:
    """加载纯文本文件 (.txt)"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        page_content=content,
        metadata={
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": "txt",
        },
    )


def _load_md(file_path: str) -> Document:
    """
    加载 Markdown 文件 (.md)
    Markdown 本质上就是纯文本，直接读取即可。
    后续切分时可以利用 # 标题层级做语义切分（进阶话题）。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        page_content=content,
        metadata={
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": "markdown",
        },
    )


def _load_pdf_text(file_path: str) -> Document:
    """
    加载 PDF 文件 —— 简易版（仅提取文本层）。

    警告：这是教学用简化实现。真实的 PDF 解析远比这复杂：
      1. 扫描件 PDF 没有文本层，需要 OCR（光学字符识别）。
      2. 多栏排版会导致文字提取顺序错乱。
      3. 页眉/页脚会被当成正文混入（下面会详细讲解）。

    这里使用 PyPDF2 做基础提取，如果你没有安装，请：
        pip install PyPDF2

    常见陷阱（详见教程正文）：
      - 页眉/页脚混入：PDF 每一页顶部和底部通常是重复的页眉页脚，
        如 "星辰科技员工手册 V3.0"、"第 X 页"。
        这些文字与正文内容无关，但 PyPDF2 会原样提取出来，
        污染文档语义，降低检索精度。
      - 表格被拆散：PDF 中的表格在提取时往往变成错位的零散文字。
      - 换行符丢失：段落内部的换行可能被当成段落边界。
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("请安装 PyPDF2：pip install PyPDF2")

    pages_text: List[str] = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append(text)

    full_text = "\n---PAGE_BREAK---\n".join(pages_text)
    return Document(
        page_content=full_text,
        metadata={
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": "pdf",
            "pages": str(len(pages_text)),
        },
    )


# ─── 分发器（关键模式！）────────────────────────────────────────
# 支持的文件类型与对应加载函数
_LOADER_REGISTRY: Dict[str, Callable[[str], Document]] = {
    ".txt": _load_txt,
    ".md":  _load_md,
    ".pdf": _load_pdf_text,
}


def load_document(file_path: str) -> Document:
    """
    ┌────────────────────────────────────────────────────────────┐
    │  文档加载的统一入口（分发器模式 / Dispatcher Pattern）      │
    │                                                            │
    │  调用方只需要传入文件路径，不需要关心具体是什么格式。        │
    │  函数内部根据文件后缀名自动分发到对应的加载函数。            │
    │                                                            │
    │  扩展新格式时，只需：                                        │
    │    1. 写一个 _load_xxx 函数                                 │
    │    2. 在 _LOADER_REGISTRY 中注册                           │
    │    3. 调用方代码完全不用改                                  │
    └────────────────────────────────────────────────────────────┘
    """
    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()

    loader = _LOADER_REGISTRY.get(ext_lower)
    if loader is None:
        supported = ", ".join(_LOADER_REGISTRY.keys())
        raise ValueError(
            f"不支持的文件格式 '{ext_lower}'。当前支持: {supported}"
        )
    return loader(file_path)


# ============================================================================
# 关于 PDF 页眉页脚污染的详细说明（代码层面不做处理，这里集中讲解）
# ============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
                PDF 页眉页脚污染的应对方案
═══════════════════════════════════════════════════════════════════════════

问题描述：
  以《星辰科技员工手册》为例，如果它是 PDF 格式，每一页顶部可能印有
  "星辰科技有限公司 — 员工手册 V3.0"，底部印有"第 X 页 / 共 Y 页"。
  PyPDF2 提取文本时会把这些重复文字也提取出来，混入正文。

  后果：
    - 页眉 "星辰科技有限公司 — 员工手册" 出现在每一个 chunk 中
    - 向量化后，这些高频但无意义的文字会拉偏语义
    - 用户搜"星辰科技"时，几乎所有 chunk 都有高相似度，检索失效

应对方案（三阶梯）：

  【方案一：正则过滤（最轻量）】
    识别页眉/页脚的固定模式，在提取后直接删除。
    适用于页眉页脚格式固定的场景。

    示例模式：
      - r'第\\s*\\d+\\s*页'           → 匹配 "第 X 页"
      - r'共\\s*\\d+\\s*页'           → 匹配 "共 Y 页"
      - r'^\\d{4}-\\d{2}-\\d{2}$'    → 匹配独立一行的日期
      注意：需要保证正则不会误删正文中的合法内容。

  【方案二：重复模式检测（中等复杂度）】
    页眉和页脚会在一份 PDF 的几乎每一页出现，正文不会。
    利用这个性质：统计所有行，出现频率接近页数的行很可能是页眉页脚。

    步骤：
      1. 按行分割每页文本
      2. 统计每行文本在多页中重复出现的频率
      3. 重复率 > 90%（即几乎每页都有）的行 → 标记为页眉/页脚
      4. 删除这些行
    这是目前业界最常用的方案，Unstructured.io 库就内置了类似逻辑。

  【方案三：专用工具（最省心）】
    使用专门处理文档解析的库，它们已经内置了页眉页脚检测：
      - Unstructured.io：开源，支持去页眉页脚、表格还原、多栏识别
      - Apache Tika：Java 生态，多格式支持，有 Python 封装 tika-python
      - Marker (GitHub)：专注 PDF → Markdown 转换，页眉页脚处理优秀
"""


# ============================================================================
# 第三部分：文本切分（Text Splitter）
# ============================================================================

def split_text(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    将一篇长文档切分成多个有重叠的短片段（Chunk）。

    ╔══════════════════════════════════════════════════════════════╗
    ║  [!!] 重要概念澄清：chunk_size 是"字符数"还是"token 数"？  ║
    ║                                                            ║
    ║  这个函数使用 字符数 控制切分大小，因为：                    ║
    ║    1. 字符数不依赖 tokenizer（不需要加载模型）               ║
    ║    2. 字符数计算简单、无额外开销                             ║
    ║    3. 对中文而言，1 个汉字 ≈ 1.5~2 个 GPT token             ║
    ║                                                            ║
    ║  但是！Embedding 模型和 LLM 的计费/限制都是按 token 算的。   ║
    ║  如果你用 chunk_size=512（字符），中文约 340 字，            ║
    ║  实际可能占用 ~500-700 token。                              ║
    ║                                                            ║
    ║  解决方案（从简单到精确）：                                   ║
    ║    方案 A: 经验估算                                         ║
    ║      中文: token ≈ 字符数 × 1.5                             ║
    ║      英文: token ≈ 字符数 / 4                               ║
    ║      设置 chunk_size 时预留余量                             ║
    ║                                                            ║
    ║    方案 B: 使用 tiktoken 精确计算                            ║
    ║      import tiktoken                                       ║
    ║      enc = tiktoken.get_encoding("cl100k_base")             ║
    ║      token_count = len(enc.encode(text))                   ║
    ║      按实际 token 数切分，但每次切分都要编码，开销较大        ║
    ║                                                            ║
    ║    方案 C: 方案 A + B 混合                                  ║
    ║      用字符数做粗切分（快），切完后再用 tiktoken 检查         ║
    ║      每个 chunk 的 token 数是否超限，超了就再切              ║
    ╚══════════════════════════════════════════════════════════════╝

    参数：
        document:      待切分的文档对象
        chunk_size:    每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 之间重叠的字符数
        separators:    分隔符优先级列表。切分时优先在靠前的分隔符处切。
                       默认: ["\\n\\n", "\\n", "。", ".", "？", "！", "；", " ", ""]

    返回：
        List[Document]：切分后的文档片段列表，每个片段保留了原始文档的
        metadata 并补充了 chunk_index 信息。

    算法逻辑：
        1. 从 separators 列表中取第一个分隔符
        2. 用该分隔符把文本拆成多段
        3. 尝试把拆开的段合并：如果当前段 + 下一段 > chunk_size，就输出当前段
        4. 输出当前段时，保留 chunk_overlap 个字符作为下一个 chunk 的开头
        5. 如果某个段本身就 > chunk_size（无法在分隔符处切），
           降级到下一个分隔符，重复以上过程
        6. 最后如果连空字符串分隔符都处理不了（理论上无此情况），硬截断
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", "？", "！", "；", " ", ""]

    metadata = dict(document.metadata)
    text = document.page_content

    # ─── 递归切分核心 ──────────────────────────────────────────────
    def _split_recursive(text_to_split: str, seps: List[str]) -> List[str]:
        """
        递归地在分隔符处切分文本。
        如果当前分隔符切不动（存在某段 > chunk_size），降级到下一个分隔符。
        """
        sep = seps[0]
        remaining_seps = seps[1:]

        # 用当前分隔符拆开
        if sep == "":
            # 空字符串分隔符 = 逐字切（最后的兜底手段）
            pieces = list(text_to_split)
        else:
            pieces = text_to_split.split(sep)

        merged: List[str] = []
        current = ""

        for piece in pieces:
            # 尝试把 piece 追加到 current
            candidate = current + (sep if current else "") + piece

            if len(candidate) <= chunk_size:
                # 还没超，继续攒
                current = candidate
            else:
                # 超了，需要输出 current 并开始新 current
                # 先处理 current（如果不为空）
                if current:
                    # 如果 current 本身没超，直接输出
                    if len(current) <= chunk_size:
                        merged.append(current)
                    else:
                        # current 本身就超了 → 降级到下一个分隔符处理
                        if remaining_seps:
                            merged.extend(_split_recursive(current, remaining_seps))
                        else:
                            # 无更多分隔符可用，硬截断
                            for i in range(0, len(current), chunk_size):
                                merged.append(current[i:i + chunk_size])
                current = piece

        # 处理最后攒着的文本
        if current:
            if len(current) <= chunk_size:
                merged.append(current)
            else:
                if remaining_seps:
                    merged.extend(_split_recursive(current, remaining_seps))
                else:
                    for i in range(0, len(current), chunk_size):
                        merged.append(current[i:i + chunk_size])

        return merged

    # ─── 带 overlap 的最终切分 ─────────────────────────────────────
    raw_chunks = _split_recursive(text, list(separators))

    # 合并过短的 chunk（如最后剩余几个字符的片段）
    merged_chunks: List[str] = []
    for i, chunk in enumerate(raw_chunks):
        if (
            merged_chunks
            and len(merged_chunks[-1]) + len(chunk) < chunk_size * 1.2
        ):
            # 和前一个合并不会超太多，合并
            merged_chunks[-1] = merged_chunks[-1] + "\n" + chunk
        else:
            merged_chunks.append(chunk)

    # 如果有 overlap 要求，生成重叠 chunk
    if chunk_overlap > 0 and len(merged_chunks) > 1:
        chunks_with_overlap: List[str] = []
        step = max(1, chunk_size - chunk_overlap)

        # 滑动窗口方式重新切分（保证相邻 chunk 有重叠）
        pos = 0
        while pos < len(text):
            chunk = text[pos:pos + chunk_size]
            if len(chunk) >= 10:  # 丢弃太短的尾片
                chunks_with_overlap.append(chunk)
            pos += step
        final_chunks = chunks_with_overlap
    else:
        final_chunks = merged_chunks

    # ─── 包装成 Document 对象 ─────────────────────────────────────
    result: List[Document] = []
    for idx, chunk_text in enumerate(final_chunks):
        chunk_meta = dict(metadata)
        chunk_meta["chunk_index"] = str(idx)
        chunk_meta["chunk_count"] = str(len(final_chunks))
        result.append(Document(page_content=chunk_text, metadata=chunk_meta))

    return result


# ============================================================================
# 第四部分：余弦相似度 —— 用 Numpy 手写
# ============================================================================

def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """
    L2 归一化：将每个向量的模长（L2范数）缩放到 1。

    数学：v' = v / ||v||₂ = v / sqrt(Σ vᵢ²)

    为什么需要归一化？
      归一化后，任意两个向量的内积（dot product）= 余弦相似度。
      因为：cos_sim(A,B) = (A·B) / (||A|| × ||B||)
      如果 ||A||=||B||=1，则 cos_sim = A·B

    参数：
        vectors: shape (N, D) 或 (D,)

    返回：
        归一化后的向量，shape 与输入相同
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0:
            return vectors
        return vectors / norm
    else:
        # 按行归一化 (N, D) 矩阵
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 避免除以零
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算余弦相似度。

    两种情况：
      1. a 是单个向量 (D,)  , b 是矩阵 (N, D) → 返回 (N,) ，a 与 b 中每个向量的相似度
      2. a 是矩阵 (M, D), b 是矩阵 (N, D) → 返回 (M, N)，两两相似度矩阵

    数学：
      cos_sim(A, B) = (A · B) / (||A|| × ||B||)

    参数：
        a: 向量或矩阵
        b: 向量或矩阵

    返回：
        相似度值，范围 [-1, 1]，值越大表示语义越接近

    示例（单个 Query 与多个 Document）：
        >>> Q_vec = embed_text("事假需要提前多久申请?")   # shape (768,)
        >>> D_mat = embed_batch(["事假需提前1个工作日",      # shape (3, 768)
        ...                      "公司提供五险一金",
        ...                      "加班费按1.5倍计算"])
        >>> scores = cosine_similarity(Q_vec, D_mat)        # shape (3,)
        >>> scores
        array([0.921, 0.153, 0.287])  # 第一个最相关
    """
    # 归一化
    a_norm = normalize_l2(a)
    b_norm = normalize_l2(b)

    if a_norm.ndim == 1 and b_norm.ndim == 2:
        # 单向量 vs 矩阵
        return np.dot(a_norm, b_norm.T)
    elif a_norm.ndim == 2 and b_norm.ndim == 1:
        # 矩阵 vs 单向量
        return np.dot(a_norm, b_norm)
    elif a_norm.ndim == 1 and b_norm.ndim == 1:
        # 单向量 vs 单向量
        return np.array([np.dot(a_norm, b_norm)])
    else:
        # 矩阵 vs 矩阵（少见，但在评估时会用到）
        return np.dot(a_norm, b_norm.T)


# ============================================================================
# 第五部分：FAISS 索引 —— IndexFlatIP + Normalize L2
# ============================================================================

# 注意：faiss 的导入名就是 faiss（不是 faiss-cpu，那个是 pip 包名）
try:
    import faiss
except ImportError:
    raise ImportError("请安装 faiss：pip install faiss-cpu（或 faiss-gpu）")


@dataclass
class VectorStore:
    """
    FAISS 向量存储封装。

    核心原理：
      IndexFlatIP = Flat(暴力搜索) + IP(Inner Product = 内积)
      因为我们在入库前已经对所有向量做了 Normalize L2，
      归一化后内积 = 余弦相似度，所以 IndexFlatIP 就是在做余弦相似度检索！

    为什么不用 IndexFlatL2（欧氏距离）？
      欧氏距离对向量模长敏感。两段"意思相同但长度不同"的文本（比如一段
      300字的回答和一段500字的回答），欧氏距离可能很大，但余弦相似度能
      正确识别它们的语义相近。语义匹配关心的是"方向"而非"长度"。

    存什么：
      - index:    FAISS 索引对象（存储归一化后的向量）
      - id_map:   dict，FAISS内部ID → 我们的 Chunk 信息
    """
    index: faiss.IndexFlatIP    # 暴力内积索引
    id_map: Dict[int, Document] = field(default_factory=dict)

    def add(self, vectors: np.ndarray, documents: List[Document]):
        """
        将向量和对应文档加入索引。

        参数：
            vectors:   已经归一化后的向量矩阵，shape (N, D)
            documents: 对应的 Document 列表，长度必须 = N
        """
        start_id = self.index.ntotal  # 当前索引中的向量总数
        self.index.add(vectors)
        for i, doc in enumerate(documents):
            self.id_map[start_id + i] = doc

    def search(self, query_vec: np.ndarray, k: int = 5):
        """
        检索与查询向量最相似的 Top-K 个文档。

        前提：query_vec 必须已经做过 Normalize L2！

        参数：
            query_vec: 归一化后的查询向量，shape (D,)
            k:         返回多少个最相似的结果

        返回：
            List[Tuple[Document, float]]：按相似度从高到低排列。
            元组的第一项是 Document，第二项是余弦相似度分数（范围-1到1）。
        """
        # faiss 需要 (1, D) 的 2D 输入
        q = query_vec.reshape(1, -1).astype(np.float32)

        # search 返回 (distances, indices)
        # 对于 IndexFlatIP，"distances" 其实是内积值（= 余弦相似度）
        scores, ids = self.index.search(q, k)

        results: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                # faiss 用 -1 表示"无更多结果"
                continue
            doc = self.id_map.get(int(idx))
            if doc is not None:
                results.append((doc, float(score)))
        return results


def build_vector_store(
    chunks: List[Document],
    embed_model: str = "text-embedding-3-small",
    embed_dims: Optional[int] = None,
) -> VectorStore:
    """
    离线索引构建的完整流程（一步到位）：

      每个 chunk 文字 → Embedding 模型 → 原始向量 → Normalize L2 → 归一化向量
                                                                    │
                                                                    ▼
                                                        faiss.IndexFlatIP

    参数：
        chunks:      切分后的文档片段列表
        embed_model: Embedding 模型名
        embed_dims:  向量维度（None = 用模型默认值）

    返回：
        VectorStore 对象（已包含所有 chunk 的索引）
    """
    print(f"  [索引构建] 正在向量化 {len(chunks)} 个 Chunk...")
    start = time.time()

    # Step 1：批量提取文本
    texts = [chunk.page_content for chunk in chunks]

    # Step 2：批量向量化（比逐个调用快 10 倍以上）
    vectors = embed_batch(texts, model=embed_model, dims=embed_dims)
    dim = vectors.shape[1]
    print(f"  [索引构建] 向量化完成，维度={dim}，耗时 {time.time() - start:.1f}s")

    # Step 3：L2 归一化（关键！归一化后内积 = 余弦相似度）
    vectors_norm = normalize_l2(vectors)

    # Step 4：创建 IndexFlatIP 并入库
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product（内积）
    store = VectorStore(index=index)
    store.add(vectors_norm, chunks)

    print(f"  [索引构建] 索引构建完成，共 {index.ntotal} 个向量")
    return store


# ============================================================================
# 第六部分：检索（Retrieval）—— Top-K + 相似度阈值过滤
# ============================================================================

def retrieve(
    query: str,
    store: VectorStore,
    embed_model: str = "text-embedding-3-small",
    embed_dims: Optional[int] = None,
    top_k: int = 5,
    threshold: float = 0.0,
) -> List[Tuple[Document, float]]:
    """
    检索阶段：把用户自然语言问题转化为向量，在 FAISS 中找最相关的 Chunk。

    流程：
      用户问题 → Embedding 模型 → 查询向量 → Normalize L2
                                                │
                                                ▼
                                         FAISS IndexFlatIP.search()
                                         (内积 = 余弦相似度，因为已归一化)

    参数：
        query:        用户问题（自然语言文本）
        store:        已构建好的 VectorStore 对象
        embed_model:  Embedding 模型（必须与构建索引时相同！）
        embed_dims:   向量维度（必须与构建索引时相同！）
        top_k:        返回多少个最相关的 chunk
        threshold:    相似度阈值（低于此分数的 chunk 会被过滤）
                      设为 0.0 表示不过滤

    返回：
        List[Tuple[Document, float]]：按相似度从高到低排列的 (文档, 分数) 列表

    关于 threshold 的选择：
      - 0.0  = 不过滤（教学演示时常用，可以看到所有结果的分数）
      - 0.60 = 较严格，只返回高度相关的内容
      - 0.40 = 适中的阈值，适合大多数场景的起步值
      - 需要根据你的 Embedding 模型和业务场景实际调试

    关于模型一致性：
      检索时的 Embedding 模型必须与构建索引时的模型完全一致！
      不同模型的向量空间不同，用模型 A 入库、模型 B 查询会导致
      检索结果完全随机，相似度分数毫无意义。
    """
    # Step 1：问题向量化
    query_vec = embed_text(query, model=embed_model, dims=embed_dims)

    # Step 2：L2 归一化（必须！对应索引中的归一化向量）
    query_vec = normalize_l2(query_vec)

    # Step 3：FAISS 检索
    raw_results = store.search(query_vec, k=top_k * 2)
    # 多取一些，因为后续可能被 threshold 过滤掉一些

    # Step 4：相似度阈值过滤
    filtered: List[Tuple[Document, float]] = []
    for doc, score in raw_results:
        if score >= threshold:
            filtered.append((doc, score))

    # Step 5：Top-K 截断
    return filtered[:top_k]


# ============================================================================
# 第七部分：Prompt 拼接（Augmentation）
# ============================================================================

def build_prompt(
    query: str,
    retrieved: List[Tuple[Document, float]],
    max_tokens: int = 4000,
) -> str:
    """
    将检索到的文档与用户问题拼接成最终送入 LLM 的 Prompt。

    Token 预算控制策略：
      - System Prompt + 格式标记 ≈ 200 tokens
      - 检索文档取最重要的 3-5 个，每个约 200-500 tokens
      - 余量用于用户问题和缓冲
      - 超长的检索结果会自动截断（此处用字符数粗略估算）

    参数：
        query:      用户原始问题
        retrieved:  retrieve() 返回的 (Document, 分数) 列表
        max_tokens: 总 token 预算上限（注意这里是字符数粗略估算）

    返回：
        完整的 prompt 字符串
    """
    # System Prompt（约 200 tokens 的指令）
    system = (
        "你是星辰科技的智能员工助手。请严格根据以下参考资料回答用户问题。\n\n"
        "必须遵守的规则：\n"
        "1. 仅使用参考资料中的信息作答，不要使用参考资料之外的知识。\n"
        "2. 回答时标注引用的资料编号，如 [1]。\n"
        "3. 如果参考资料中没有相关信息，明确回答：\n"
        '   "参考资料中未找到相关信息，建议您查阅公司最新政策文件或联系HR部门。"\n'
        "4. 回答应简洁、结构化，便于员工快速理解。\n"
        "5. 引用具体的条款和数据时，确保与参考资料完全一致。\n"
    )

    # 组装参考资料
    refs = ""
    for i, (doc, score) in enumerate(retrieved, 1):
        source = doc.metadata.get("file_name", "未知来源")
        refs += f"\n[参考文档 {i}] 来源：{source}（相似度: {score:.3f}）\n"
        refs += f"{doc.page_content}\n"

    # Token 预算粗略估算（中文: 1 字符 ≈ 1.5 token）
    system_tokens = len(system) * 1.5
    query_tokens = len(query) * 1.5
    avail_tokens = max_tokens - system_tokens - query_tokens - 100  # 100 缓冲

    # 如果参考资料太长，截断
    if len(refs) * 1.5 > avail_tokens:
        # 从最相关的开始保留，直到接近 token 预算
        truncated = ""
        used = 0.0
        for i, (doc, score) in enumerate(retrieved, 1):
            block = (
                f"\n[参考文档 {i}] 来源：{doc.metadata.get('file_name', '未知')}"
                f"（相似度: {score:.3f}）\n"
                f"{doc.page_content}\n"
            )
            block_tokens = len(block) * 1.5
            if used + block_tokens > avail_tokens:
                break
            truncated += block
            used += block_tokens
        refs = truncated

    # 最终拼接
    prompt = (
        f"{system}\n"
        f"{'='*60}\n"
        f"## 参考资料\n"
        f"{refs}\n"
        f"{'='*60}\n"
        f"## 用户问题\n"
        f"{query}\n"
    )

    estimated_tokens = len(prompt) * 1.5
    print(f"  [Prompt] 估计 token 数: {estimated_tokens:.0f} / {max_tokens}")
    return prompt


# ============================================================================
# 第八部分：LLM 调用（Generation）
# ============================================================================

def generate_answer(
    prompt: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    调用 LLM 生成最终回答。

    参数：
        prompt: build_prompt() 拼好的完整 prompt
        model:  LLM 模型名，可选 gpt-4o-mini / gpt-4o / deepseek-chat 等

    返回：
        LLM 的回答文本

    注意：如果你使用的是硅基流动等中转接口，model 需要改成对应的模型名，
          如 "deepseek-ai/DeepSeek-V3"、"Qwen/Qwen3-235B" 等
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # 低温度 → 更稳定、更少编造
        max_tokens=1024,   # 回答的最大长度
    )
    return response.choices[0].message.content or ""


# ============================================================================
# 第九部分：完整 Pipeline 端到端串联
# ============================================================================

def rag_pipeline(
    query: str,
    store: VectorStore,
    top_k: int = 3,
    threshold: float = 0.4,
    verbose: bool = True,
) -> str:
    """
    RAG 完整流水线：检索 → 增强 → 生成。

    这是整个系统的顶层入口，串联了所有步骤。

    参数：
        query:     用户问题
        store:     已构建的向量索引
        top_k:     检索数量
        threshold: 相似度阈值
        verbose:   是否打印中间过程信息

    返回：
        LLM 生成的最终回答
    """
    if verbose:
        print("=" * 60)
        print(f"  [Query] {query}")
        print("=" * 60)

    # Step 1+2：检索（向量化 + FAISS 搜索）
    results = retrieve(query, store, top_k=top_k, threshold=threshold)

    if verbose:
        print(f"\n  [Retrieval] 检索到 {len(results)} 个相关文档：")
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("file_name", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    {i}. [{source}] 相似度={score:.4f} | {preview}...")

    if not results:
        return "参考资料中未找到相关信息，建议您查阅公司最新政策文件或联系HR部门。"

    # Step 3：Prompt 拼接（增强）
    prompt = build_prompt(query, results)

    # Step 4：LLM 生成
    if verbose:
        print(f"\n  [LLM] 正在生成回答...")

    answer = generate_answer(prompt)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  [Answer]")
        print(f"{answer}")
        print(f"{'=' * 60}")

    return answer


# ============================================================================
# 第十部分：测试入口 —— 验证每一步的效果
# ============================================================================

if __name__ == "__main__":
    """
    运行方式：
        # 先设置 API Key
        export OPENAI_API_KEY="sk-xxx"

        # 然后运行
        python rag_from_scratch.py

    注意：
        - 如果没有 OpenAI API Key，可以用硅基流动(SiliconFlow)等中转服务
        - 设置 OPENAI_BASE_URL 指向中转地址即可
        - 模型名也需要对应修改，如 "deepseek-ai/DeepSeek-V3"
    """
    import sys

    # ─── 测试 1：Embedding 单条与批处理 ──────────────────────────
    print("\n" + "=" * 70)
    print("    测试 1：Embedding 向量化")
    print("=" * 70)

    # 在 try-except 中做，方便用户看到清晰的错误提示
    try:
        # 单条向量化
        test_text = "事假需提前1个工作日申请"
        vec = embed_text(test_text, dims=512)
        print(f"  单条文本: \"{test_text}\"")
        print(f"  向量维度: {vec.shape}")
        print(f"  向量类型: {vec.dtype}")
        print(f"  前5个值:  [{', '.join(f'{v:.4f}' for v in vec[:5])}]")
        print()

        # 批量向量化（3 条文本一起传）
        test_texts = [
            "事假需提前1个工作日申请",
            "公司为员工缴纳五险一金",
            "加班费按1.5倍工资计算",
        ]
        batch_vecs = embed_batch(test_texts, dims=512)
        print(f"  批量文本: {len(test_texts)} 条")
        print(f"  批量结果形状: {batch_vecs.shape}  (期望: 3×512)")
        print(f"  单条 vs 批量第1条是否一致: {np.allclose(vec, batch_vecs[0], atol=1e-5)}")
        print("  (注意：由于浮点精度，单条和批量的结果可能略有差异，这是正常的)")

        # 用余弦相似度验证语义
        Q = embed_text("员工请假需要提前多久？", dims=512)
        D = embed_batch([
            "事假需提前1个工作日向部门主管申请",
            "五险一金包括养老、医疗、失业、工伤和生育保险",
            "工作日加班按1.5倍工资计算加班费",
        ], dims=512)
        scores = cosine_similarity(Q, D)
        print(f"\n  语义相似度测试 (Query: '员工请假需要提前多久？')：")
        for i, (txt, s) in enumerate(zip([
            "事假需提前1个工作日向部门主管申请",
            "五险一金包括养老、医疗...",
            "工作日加班按1.5倍工资计算",
        ], scores)):
            print(f"    [{i}] 相似度={s:.4f} | {txt[:40]}...")
        print("  (预期: 第0条相似度最高，因为它讲的是请假)")

    except RuntimeError as e:
        print(f"  [WARN] 跳过测试 1：{e}")
        print("  请设置环境变量 OPENAI_API_KEY 后重试。")

    # ─── 测试 2：Document Loader ──────────────────────────────────
    print("\n" + "=" * 70)
    print("    测试 2：Document Loader（文档加载器）")
    print("=" * 70)

    handbook_path = "星辰科技员工手册.md"
    if os.path.exists(handbook_path):
        doc = load_document(handbook_path)
        print(f"  加载文件: {handbook_path}")
        print(f"  文件类型: {doc.metadata['file_type']}")
        print(f"  文档长度: {len(doc.page_content)} 字符")
        print(f"  Metadata: {doc.metadata}")
        print(f"  正文前200字符:\n    {doc.page_content[:200].replace(chr(10), chr(10)+'    ')}")
    else:
        print(f"  [WARN] 未找到测试文件 '{handbook_path}'，请先创建该文件。")

    # ─── 测试 3：Text Splitter ────────────────────────────────────
    print("\n" + "=" * 70)
    print("    测试 3：Text Splitter（文本切分）")
    print("=" * 70)

    if os.path.exists(handbook_path):
        doc = load_document(handbook_path)
        chunks = split_text(doc, chunk_size=512, chunk_overlap=64)
        print(f"  切分参数: chunk_size=512(字符), overlap=64(字符)")
        print(f"  Chunk 数量: {len(chunks)}")
        print(f"\n  前5个 Chunk 预览：")
        for i, chunk in enumerate(chunks[:5]):
            length = len(chunk.page_content)
            preview = chunk.page_content[:100].replace("\n", "\\n")
            print(f"    Chunk[{i}] len={length} | {preview}...")
    else:
        print(f"  [WARN] 跳过（需要测试文件）")

    # ─── 测试 4：余弦相似度 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("    测试 4：余弦相似度（Numpy 手写）")
    print("=" * 70)

    # 用模拟向量演示（不需要 API）
    np.random.seed(42)
    A = np.random.randn(8).astype(np.float32)     # 模拟 query
    B_good = A + 0.1 * np.random.randn(8).astype(np.float32)  # 相似
    B_bad = -A + 0.1 * np.random.randn(8).astype(np.float32)   # 不相似
    B_random = np.random.randn(8).astype(np.float32)           # 无关

    sim_good = cosine_similarity(A, B_good)[0]
    sim_bad = cosine_similarity(A, B_bad)[0]
    sim_random = cosine_similarity(A, B_random)[0]

    print(f"  Query向量 A:   [{', '.join(f'{v:.3f}' for v in A[:4])}...]")
    print(f"  相似向量 B1:   [{', '.join(f'{v:.3f}' for v in B_good[:4])}...]")
    print(f"  不相似向量 B2: [{', '.join(f'{v:.3f}' for v in B_bad[:4])}...]")
    print(f"  无关向量 B3:   [{', '.join(f'{v:.3f}' for v in B_random[:4])}...]")
    print(f"\n  cos(A, B1) = {sim_good:.4f}  ← 应该接近 1.0（B1 由 A+噪声生成）")
    print(f"  cos(A, B2) = {sim_bad:.4f}  ← 应该接近 -1.0（B2 = -A+噪声）")
    print(f"  cos(A, B3) = {sim_random:.4f}  ← 应该接近 0（随机向量）")

    # 同时验证归一化后内积 = 余弦相似度
    A_norm = normalize_l2(A)
    B_norm = normalize_l2(B_good)
    dot_product = np.dot(A_norm, B_norm)
    print(f"\n  验证：归一化后 A·B = {dot_product:.4f}")
    print(f"        余弦相似度    = {sim_good:.4f}")
    print(f"  两者是否相等（误差 < 10^-6）: {abs(dot_product - sim_good) < 1e-6}")

    # ─── 测试 5：完整 RAG Pipeline ────────────────────────────────
    print("\n" + "=" * 70)
    print("    测试 5：完整 RAG Pipeline")
    print("=" * 70)

    if not os.path.exists(handbook_path):
        print("  ⚠ 跳过（需要测试文件）")
        sys.exit(0)

    try:
        # Step 1: 加载文档
        doc = load_document(handbook_path)
        print(f"  [1] 文档加载: OK ({len(doc.page_content)} 字符)")

        # Step 2: 切分
        chunks = split_text(doc, chunk_size=512, chunk_overlap=64)
        print(f"  [2] 文本切分: OK ({len(chunks)} 个 Chunk)")

        # Step 3: 构建索引（这里需要调用 Embedding API）
        print(f"  [3] 构建索引中...")
        store = build_vector_store(chunks, embed_dims=512)
        print(f"  [3] 索引构建: OK ({store.index.ntotal} 个向量)")

        # Step 4: 测试几个问题
        test_queries = [
            "请事假需要提前多久申请？",
            "公司给员工缴纳哪些保险？",
            "加班费怎么算的？",
            "试用期一般多长？",
            "报销的流程是怎样的？",
        ]

        for q in test_queries:
            print(f"\n{'-' * 60}")
            answer = rag_pipeline(q, store, top_k=3, threshold=0.4)
            print(f"  [最终回答]\n{answer}")

    except RuntimeError as e:
        print(f"  [WARN] 需要 API Key 的测试跳过：{e}")
        print()
        print("  === 无 API 环境下的验证 === ")
        print("  以下测试不依赖 API，仅用 numpy 和 faiss 本地验证。")

        # 构建模拟数据
        mock_texts = [
            "事假需提前1个工作日向部门主管申请。",
            "公司为员工缴纳五险一金。",
            "工作日加班按1.5倍工资计算。",
            "试用期为3个月，表现优秀可提前转正。",
            "报销需在30日内提交申请。",
        ]
        # 用随机向量模拟 embedding
        np.random.seed(123)
        mock_vectors = np.random.randn(len(mock_texts), 128).astype(np.float32)
        mock_vectors = normalize_l2(mock_vectors)

        # 构建 FAISS 索引
        index = faiss.IndexFlatIP(128)
        mock_store = VectorStore(index=index)
        mock_docs = [
            Document(page_content=t, metadata={"source": f"doc_{i}", "file_name": f"政策手册第{i}章.md"})
            for i, t in enumerate(mock_texts)
        ]
        mock_store.add(mock_vectors, mock_docs)

        # 模拟查询
        Q_vec = mock_vectors[0] + 0.1 * np.random.randn(128).astype(np.float32)
        Q_vec = normalize_l2(Q_vec)

        results = mock_store.search(Q_vec, k=3)
        print(f"\n  模拟查询（最相关的是第0条）：")
        for i, (d, s) in enumerate(results):
            print(f"    [{i}] 相似度={s:.4f} | {d.page_content}")
        print(f"\n  [OK] 预期：第0条（事假...）相似度最高且接近1.0")
        print(f"  [OK] FAISS IndexFlatIP 本地工作正常")

    print("\n" + "=" * 70)
    print("    所有测试完成")
    print("=" * 70)
