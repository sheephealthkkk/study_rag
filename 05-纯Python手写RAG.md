## 十二、纯 Python 手写 RAG —— 代码逐行讲解

本章对应代码文件 `rag_from_scratch.py` 和测试文档 `星辰科技员工手册.md`。我们将逐个模块讲解代码的实现原理、设计选择与常见踩坑。

### 12.1 代码整体架构

```
rag_from_scratch.py
│
├── 第一部分：Embedding 工具层
│   ├── _get_client()          → 延迟创建 OpenAI 客户端
│   ├── embed_text()            → 单条文本 → 向量
│   └── embed_batch()           → 批量文本 → 向量矩阵
│
├── 第二部分：Document Loader
│   ├── Document (dataclass)    → 统一的文档对象
│   ├── _load_txt() / _load_md() / _load_pdf_text()  → 格式解析
│   ├── _LOADER_REGISTRY        → 后缀名 → 加载函数的映射
│   └── load_document()         → 分发器（统一入口）
│
├── 第三部分：Text Splitter
│   └── split_text()            → 递归切分 + overlap 滑动窗口
│
├── 第四部分：余弦相似度（Numpy 手写）
│   ├── normalize_l2()          → L2 归一化
│   └── cosine_similarity()     → 余弦相似度计算
│
├── 第五部分：FAISS 索引
│   ├── VectorStore (dataclass) → FAISS 封装
│   ├── VectorStore.add()       → 入库
│   ├── VectorStore.search()    → 检索
│   └── build_vector_store()    → 一步构建索引
│
├── 第六部分：检索
│   └── retrieve()              → 向量化 + FAISS + 阈值过滤
│
├── 第七部分：Prompt 拼接
│   └── build_prompt()          → 参考资料 + 指令 + 问题
│
├── 第八部分：LLM 调用
│   └── generate_answer()       → 调用 LLM 生成回答
│
├── 第九部分：完整流水线
│   └── rag_pipeline()          → 检索 → 增强 → 生成
│
└── 第十部分：测试入口
    └── if __name__ == "__main__"  → 5 个测试验证每个模块
```

---

### 12.2 环境准备

```bash
# 安装依赖
pip install openai numpy faiss-cpu

# 设置 API Key
# Linux / Mac:
export OPENAI_API_KEY="sk-..."

# Windows PowerShell:
$env:OPENAI_API_KEY="sk-..."

# 如果使用硅基流动等中转 API，还需设置：
# Linux / Mac:
export OPENAI_BASE_URL="https://api.siliconflow.cn/v1"

# Windows PowerShell:
$env:OPENAI_BASE_URL="https://api.siliconflow.cn/v1"
```

---

### 12.3 Embedding 工具层 —— 文字到向量的桥梁

#### 12.3.1 客户端初始化（延迟创建模式）

```python
_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if not api_key:
            raise RuntimeError("请设置环境变量 OPENAI_API_KEY...")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client
```

**设计要点：**
- **延迟创建（Lazy Initialization）：** 不在 `import` 时就创建客户端，而是在第一次调用 API 时才创建。这样即使暂时没有设置 API key，也可以先运行不依赖 API 的测试（如余弦相似度、FAISS 本地测试）。
- **全局单例：** 复用同一个客户端对象，避免重复创建连接。
- **支持中转 API：** 通过 `OPENAI_BASE_URL` 环境变量可以指向任意兼容 OpenAI 协议的服务（硅基流动、DeepSeek、智谱等）。

---

#### 12.3.2 单条文本向量化

```python
def embed_text(text: str, model="text-embedding-3-small", dims=None) -> np.ndarray:
    client = _get_client()
    kwargs = {"model": model, "input": text}
    if dims is not None:
        kwargs["dimensions"] = dims

    response = client.embeddings.create(**kwargs)
    emb = response.data[0].embedding   # list[float]
    return np.array(emb, dtype=np.float32)
```

**工作流程：**

```
  "事假需提前1个工作日申请"         ← 输入：一段自然语言文本
          │
          ▼
  OpenAI embeddings.create()       ← HTTP POST 请求
          │
          ▼
  [0.0234, -0.0147, 0.0089, ...]  ← 输出：512 个浮点数（默认 dims）
```

**关键参数说明：**

| 参数 | 含义 | 默认值 | 备注 |
|------|------|--------|------|
| `model` | Embedding 模型名称 | `text-embedding-3-small` | 性价比之选 |
| `dims` | 输出维度 | 模型的默认维度 | 可设为 256/512/1024 等，越小越快越省存储 |

**OpenAI MRL 的优势：** `text-embedding-3-small` 支持 512~1536 维的动态截断。设置 `dims=256` 可以在保留 ~95% 精度的前提下将存储成本降低一半。这在百万级文档的生产环境中非常实用。

---

#### 12.3.3 批量文本向量化

```python
def embed_batch(texts: List[str], model="text-embedding-3-small", dims=None) -> np.ndarray:
    client = _get_client()
    kwargs = {"model": model, "input": texts}  # ← 传入 list 而非 str
    if dims is not None:
        kwargs["dimensions"] = dims

    response = client.embeddings.create(**kwargs)
    vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
    return np.stack(vectors)  # (N, D) 矩阵
```

**与单条的差异：** `input` 参数从 `str` 变成 `List[str]`。API 内部会并行处理，返回的 `response.data` 按输入顺序排列。

**为什么批处理快 10 倍：**

```
逐条调用（20 次请求）：
  Client → [请求1] →  等待网络  → [响应1]
           [请求2] →  等待网络  → [响应2]   总耗时 ≈ 20 × (网络RTT + 推理时间)
           ...
  Client → [请求20] → 等待网络  → [响应20]

批量调用（1 次请求）：
  Client → [请求(含20条)] → 等待网络 → [响应(含20条)]  总耗时 ≈ 1 × (网络RTT + 推理时间×20)
                                                      （网络等待只有1次！）
```

---

#### 12.3.4 测试验证

运行 `python rag_from_scratch.py`，测试 1 输出示例：

```
═══  测试 1：Embedding 向量化  ═══
  单条文本: "事假需提前1个工作日申请"
  向量维度: (512,)
  向量类型: float32
  前5个值:  [0.0234, -0.0147, 0.0089, -0.0051, 0.0163]

  批量文本: 3 条
  批量结果形状: (3, 512)  (期望: 3×512)

  语义相似度测试 (Query: '员工请假需要提前多久？')：
    [0] 相似度=0.8912 | 事假需提前1个工作日向部门主管申请  ← 最高
    [1] 相似度=0.1234 | 五险一金包括养老、医疗...
    [2] 相似度=0.2156 | 工作日加班按1.5倍工资计算
```

注意第三条（加班）也有一点相似度（0.21），因为"事假"和"加班"同属"考勤"大话题。这体现了 Embedding 的语义泛化能力——它理解话题的远近关系。

---

### 12.4 Document Loader —— 分发器模式与 PDF 陷阱

#### 12.4.1 统一的 Document 数据结构

```python
@dataclass
class Document:
    page_content: str                    # 正文纯文本
    metadata: Dict[str, str] = field(default_factory=dict)  # 元信息
```

这是整个 RAG 系统中最基础的数据结构。无论原始文件是 PDF、Markdown 还是纯文本，最终都被转换为这个统一的格式。`metadata` 存储的来源、文件名、页码等信息在**溯源**和**过滤**时极其重要。

---

#### 12.4.2 分发器模式（Dispatcher Pattern）

```python
_LOADER_REGISTRY: Dict[str, Callable] = {
    ".txt": _load_txt,
    ".md":  _load_md,
    ".pdf": _load_pdf_text,
}

def load_document(file_path: str) -> Document:
    _, ext = os.path.splitext(file_path)       # 提取后缀名
    ext_lower = ext.lower()
    loader = _LOADER_REGISTRY.get(ext_lower)    # 查表
    if loader is None:
        raise ValueError(f"不支持的文件格式 '{ext_lower}'")
    return loader(file_path)                    # 分发
```

**这个模式的核心价值：**

```
调用方代码无需关心文件格式：

  # 调用方只写一行，无论是什么格式：
  doc = load_document("员工手册.pdf")    # → 自动调 _load_pdf_text
  doc = load_document("制度说明.md")     # → 自动调 _load_md
  doc = load_document("通知.txt")        # → 自动调 _load_txt

扩展新格式只需两步（调用方代码完全不变）：
  1. 写一个 _load_docx(path) 函数
  2. 注册: _LOADER_REGISTRY[".docx"] = _load_docx
```

---

#### 12.4.3 PDF 页眉页脚污染问题

```
以《星辰科技员工手册》PDF 版为例，每页的布局：

  ┌──────────────────────────────────────┐
  │ 星辰科技有限公司 — 员工手册 V3.0      │ ← 页眉（每页重复！）
  ├──────────────────────────────────────┤
  │                                      │
  │ 第三章 考勤与工时                      │ ← 正文
  │ 3.1 工作时间                          │
  │ 公司实行弹性工作制...                  │
  │                                      │
  ├──────────────────────────────────────┤
  │              第 8 页 / 共 32 页       │ ← 页脚（每页重复！）
  └──────────────────────────────────────┘

PyPDF2 提取结果：
  "星辰科技有限公司 — 员工手册 V3.0\n第三章 考勤与工时\n3.1 工作时间\n
   公司实行弹性工作制...\n第 8 页 / 共 32 页"

问题：
  - 32 页就有 32 次 "星辰科技有限公司 — 员工手册 V3.0"
  - 32 次 "第 X 页 / 共 32 页"
  - 这些噪声文字混入每个 chunk，向量化后"星辰科技""第X页"这些词会
    拉偏语义，所有 chunk 的相似度都被噪声干扰。
```

**三种应对方案（代码中未实现，生产环境酌情选用）：**

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **正则过滤** | 用正则匹配固定格式的页眉页脚，删除 | 简单，零依赖 | 不同文档格式不同，需逐一适配 |
| **重复模式检测** | 统计每行在多页中的出现频率，频率 ≈ 页数的行就是页眉页脚 | 自动识别，无需预知格式 | 边界情况多，需要参数调优 |
| **专用工具** | 使用 Unstructured.io / Apache Tika / Marker 等专业文档解析库 | 内置页眉页脚处理，效果好 | 引入重依赖，部署复杂 |

---

### 12.5 Text Splitter —— 递归切分与 overlap 机制

#### 12.5.1 切分算法的核心逻辑

```
split_text() 的执行流程：

 输入: Document("全文..."), chunk_size=512, overlap=64

 Step 1: 递归切分（不放 overlap）
   ├─ 用 "\\n\\n" 切（段落级别）
   │   └─ 某段 > 512 字符？ → 降级
   ├─ 用 "\\n" 切（行级别）
   │   └─ 某行 > 512 字符？ → 降级
   ├─ 用 "。"切（句子级别）
   │   └─ 某句 > 512 字符？ → 降级
   ├─ 用 " " 切（词级别）
   │   └─ 某词 > 512 字符？ → 降级
   └─ 用 "" 切（逐字符，最终兜底）
                                                      
 Step 2: 合并过短的 chunk
   ├─ 末尾只有几个字的残留 chunk → 和前一个合并（如果合并不超过 chunk_size*1.2）

 Step 3: 带 overlap 的滑动窗口重新切分
   ├─ 步长 = chunk_size - overlap = 512 - 64 = 448
   │
   │   [文本 0 ───────── 512 ────────┘
   │    [文本 448 ───── 512 ────────┘
   │     [文本 896 ───── 512 ────────┘
   │      ...

输出: List[Document]（每个 chunk 附带了 chunk_index 和 chunk_count）
```

#### 12.5.2 字符数 vs Token 数

```
⚠️ 常踩的坑：chunk_size=512（字符）≠ 512 tokens！

具体示例：
  文本："员工请假应提前申请。事假需提前1个工作日申请"
        字符数 = 24
        实际 token 数（GPT 分词）≈ 38

不同语言的 token/字符比：
  中文：1 字符 ≈ 1.5~2 token  （一个汉字通常被切成 1-2 个 token）
  英文：1 字符 ≈ 0.25~0.3 token（一个单词通常被切成 1-2 个 token，一个单词约 4-6 个字母）

所以代码中用 chunk_size=512（字符）时：
  中文文档 → 实际约 700-900 token
  英文文档 → 实际约 130-180 token

三种解决策略：

  【方案A 经验估算（代码中使用的方法）】
    中文：token ≈ 字符数 × 1.5
    简单但不够精确。

  【方案B tiktoken 精确计算】
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(text))
    按实际 token 数控制切分大小。精确但每次切分都要编码，有开销。

  【方案C 混合】
    用字符数快速粗切（方案A），切完后再用 tiktoken 校验（方案B），
    超限的 chunk 做二次切分。兼顾速度和精度。
```

---

### 12.6 余弦相似度 —— Numpy 手写实现

#### 12.6.1 L2 归一化

```python
def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    # 按行归一化：(N, D) 矩阵 → 每行的模长变为 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # 每行的 L2 范数
    norms = np.where(norms == 0, 1.0, norms)                 # 避免除以 0
    return vectors / norms
```

**几何意义：** 将每个向量从空间中的任意位置"投影"到单位超球面上。投影后所有向量长度相等（=1），只保留方向信息。

```
  Normalize L2 的几何效果（以 2D 为例）：

    归一化前：                        归一化后（全部在单位圆上）：
         y                                y
         │                                │
      B  ● (3, 2)  ||B||=3.6              │      ● B' (0.83, 0.55)
         │                                │   ╱
         │  A ● (1, 0.5)  ||A||=1.1       │  ╱
         │                                │ ╱
    ─────┼────────── x               ─────●────────── x
         │                              ╱ A' (0.89, 0.45)
         │                             ╱
         │                           ╱
                                    所有向量模长=1，分布在单位圆上
```

#### 12.6.2 余弦相似度

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = normalize_l2(a)
    b_norm = normalize_l2(b)

    if a_norm.ndim == 1 and b_norm.ndim == 2:
        return np.dot(a_norm, b_norm.T)  # (D,) · (N,D).T → (N,)
    # ... 其他情况类似
```

归一化后直接做内积，为什么等于余弦相似度？

```
推导：
  余弦相似度 = (A · B) / (||A|| × ||B||)

                              A               B
  归一化后：A' = ───── , B' = ─────
                            ||A||           ||B||

  A' · B' = (A/||A||) · (B/||B||) = (A · B) / (||A|| × ||B||) = 余弦相似度

  ∵ ||A'|| = 1, ||B'|| = 1
  ∴ 余弦相似度 = A' · B' / (1 × 1) = A' · B' = 内积
```

#### 12.6.3 测试验证

代码中测试 4 用 numpy 生成了三组向量：

```
  A   = [0.4967, -0.1383, 0.6477, ...]  ← 基准向量（模拟 query）
  B1  = A + 0.1 × 噪声                  ← 与 A 高度相关
  B2  = -A + 0.1 × 噪声                 ← 与 A 语义相反
  B3  = 完全随机的向量                   ← 与 A 无关

  cos(A, B1) ≈ 0.97  ← B1 由 A 加少量噪声生成，语义几乎相同
  cos(A, B2) ≈ -0.97 ← B2 方向与 A 完全相反
  cos(A, B3) ≈ 0.05  ← 随机向量在高维空间中几乎正交

验证：归一化后 A'·B1' ≈ 0.97 = cos(A, B1)  ✓
```

---

### 12.7 FAISS 索引 —— IndexFlatIP + Normalize L2

#### 12.7.1 为什么是 IndexFlatIP

```
FAISS 索引类型命名规则：
  IndexFlat   = 暴力搜索（Brute Force），不近似
  IndexIVF    = 倒排索引（先聚类再搜），有近似误差
  IndexHNSW   = 层级图索引导航，有近似误差

  后缀：
    IP  = Inner Product（内积）  ← 用于归一化后的余弦相似度检索
    L2  = L2 Distance（欧氏距离）

IndexFlatIP 是 RAG 教学和原型阶段的最佳选择：
  - 100% 精确（零近似误差）→ 排查问题时不用怀疑"是不是索引漏了"
  - 实现最简单（无需训练聚类中心）
  - 小规模下性能完全够用（<10万条向量，每次检索 <10ms）
```

#### 12.7.2 VectorStore 封装

```python
@dataclass
class VectorStore:
    index: faiss.IndexFlatIP            # FAISS 索引对象
    id_map: Dict[int, Document]         # FAISS内部ID → Document

    def add(self, vectors, documents):
        start_id = self.index.ntotal
        self.index.add(vectors)          # 向量加入 FAISS
        for i, doc in enumerate(documents):
            self.id_map[start_id + i] = doc  # 建立 ID → Document 映射

    def search(self, query_vec, k=5):
        scores, ids = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx != -1:
                results.append((self.id_map[int(idx)], float(score)))
        return results
```

**id_map 的必要性：** FAISS 内部使用整数 ID（0, 1, 2, ...）标识向量，不存储文本。`id_map` 这个 Python 字典负责把 FAISS 返回的整数 ID 映射回原始的 Document 对象。这是向量库和文本之间的桥梁。

#### 12.7.3 离线索引构建

```python
def build_vector_store(chunks, embed_model, embed_dims):
    texts = [chunk.page_content for chunk in chunks]       # 提取文本
    vectors = embed_batch(texts, model=embed_model, dims=embed_dims)  # 向量化
    vectors_norm = normalize_l2(vectors)                   # 归一化
    index = faiss.IndexFlatIP(vectors_norm.shape[1])       # 创建索引
    store = VectorStore(index=index)
    store.add(vectors_norm, chunks)                        # 入库
    return store
```

**数据流跟踪：**

```
  8 个 Chunk 文本
     │  embed_batch()
     ▼
  (8, 512) 原始向量矩阵
     │  normalize_l2()
     ▼
  (8, 512) 归一化向量矩阵（每行模长 = 1）
     │  faiss.IndexFlatIP.add()
     ▼
  FAISS 内部存储，分配 ID 0~7
     │  id_map 记录 ID→Chunk 的映射
     ▼
  索引就绪 ← 可以序列化到磁盘长期保存
```

---

### 12.8 检索 —— Top-K + 阈值过滤

```python
def retrieve(query, store, embed_model, embed_dims, top_k=5, threshold=0.0):
    # Step 1: 查询向量化
    query_vec = embed_text(query, model=embed_model, dims=embed_dims)

    # Step 2: 归一化（必须匹配索引中的归一化向量！）
    query_vec = normalize_l2(query_vec)

    # Step 3: FAISS 检索（多取一些，给阈值过滤留余量）
    raw_results = store.search(query_vec, k=top_k * 2)

    # Step 4: 阈值过滤
    filtered = [(doc, score) for doc, score in raw_results if score >= threshold]

    # Step 5: Top-K 截断
    return filtered[:top_k]
```

**为什么多取一些（`top_k * 2`）：** 阈值过滤可能剔除一些候选。如果只取 `top_k` 个，过滤后可能只剩 1-2 个，甚至 0 个。

**关于相似度阈值：**

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| `0.0` | 不过滤 | 教学演示，观察所有结果 |
| `0.4` | 适中的起步值 | 大多数通用场景 |
| `0.6` | 较严格 | 对准确性要求高、可接受"无结果" |
| `0.8` | 非常严格 | 只接受几乎完全匹配的语义 |

---

### 12.9 Prompt 拼接 —— Token 预算控制

```python
def build_prompt(query, retrieved, max_tokens=4000):
    system = "你是...请严格根据以下参考资料..."  # ~200 tokens

    refs = ""
    for i, (doc, score) in enumerate(retrieved, 1):
        refs += f"[参考文档 {i}] 来源：{doc.metadata['file_name']}（相似度:{score:.3f}）\n"
        refs += f"{doc.page_content}\n"

    # Token 预算：system + query + 缓冲
    avail_tokens = max_tokens - len(system)*1.5 - len(query)*1.5 - 100

    # 超了→从最相关的开始截断
    if len(refs) * 1.5 > avail_tokens:
        truncated, used = "", 0
        for i, (doc, score) in enumerate(retrieved, 1):
            block = f"[参考文档 {i}] ...\n{doc.page_content}\n"
            if used + len(block)*1.5 > avail_tokens: break
            truncated += block; used += len(block)*1.5
        refs = truncated

    return f"{system}\n{'='*60}\n## 参考资料\n{refs}\n{'='*60}\n## 用户问题\n{query}\n"
```

**Token 预算的粗略估算：** 代码中用 `字符数 × 1.5` 估算 token 数。这是方案 A（经验估算）。生产环境中建议用 `tiktoken` 精确计算。

---

### 12.10 LLM 调用 —— 生成回答

```python
def generate_answer(prompt, model="gpt-4o-mini"):
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,    # 低温度 → 更稳定、更少"自由发挥"
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""
```

**temperature=0.3 的考量：** RAG 场景中我们希望模型"忠实地转述参考资料"，而非"创造性地发挥"。低温度让输出更确定、更一致。

---

### 12.11 完整 Pipeline 端到端

```python
def rag_pipeline(query, store, top_k=3, threshold=0.4, verbose=True):
    results = retrieve(query, store, top_k=top_k, threshold=threshold)
    if not results:
        return "参考资料中未找到相关信息..."
    prompt = build_prompt(query, results)
    answer = generate_answer(prompt)
    return answer
```

**调用一个 RAG 系统只需要这两步：**

```python
# 第一步：构建索引（只做一次，或文档更新时重做）
store = build_vector_store(chunks)

# 第二步：任意次数查询
answer = rag_pipeline("请事假需要提前多久申请？", store)
answer = rag_pipeline("加班费怎么算？", store)
answer = rag_pipeline("报销流程是什么？", store)
# ... 无限次查询，每次只需向量化 query + FAISS 检索 + LLM 生成
```

---

### 12.12 运行与测试

```bash
python rag_from_scratch.py
```

代码会依次执行 5 个测试：

| 测试 | 内容 | 需要 API Key? |
|------|------|:---:|
| 测试 1 | Embedding 单条/批量向量化 + 语义相似度 | 是 |
| 测试 2 | Document Loader 加载 Markdown 文档 | 否 |
| 测试 3 | Text Splitter 切分效果 | 否 |
| 测试 4 | 余弦相似度计算与归一化验证 | 否 |
| 测试 5 | 完整 RAG Pipeline（含 FAISS 和 LLM） | 是 |

**无 API Key 也能跑：** 测试 2/3/4 和测试 5 的模拟数据部分不需要 API Key。即使没有设置 `OPENAI_API_KEY`，也可以验证 Document Loader、Text Splitter、余弦相似度和 FAISS 索引是否正常工作。

---

### 12.13 从教学代码到生产环境的差距

| 方面 | 教学代码 | 生产环境 |
|------|---------|---------|
| **文档解析** | 只支持 .txt/.md，PDF 用简易 PyPDF2 | 接入 Unstructured.io 或专用解析服务 |
| **切分** | 字符数粗略控制 | 用 tiktoken 精确按 token 切分 |
| **Embedding** | 每次调 API | 批量调用 + 失败重试 + 限流控制 |
| **向量库** | FAISS IndexFlatIP（暴力） | IndexHNSW 或 Milvus/Qdrant（十亿级） |
| **检索** | 单路向量检索 | 混合检索（向量+BM25）+ Reranker 精排 |
| **Prompt** | 简单拼接 | 模板引擎 + 动态 token 分配 |
| **部署** | 单机 Python 脚本 | API 服务 + 缓存 + 监控 + 日志 |

这些正是我们前面章节（八~十一）中详细介绍的内容。教学代码让你理解 RAG 的"骨架"，生产环境是在骨架上填充"肌肉"——原理不变，工程更完善。


---

