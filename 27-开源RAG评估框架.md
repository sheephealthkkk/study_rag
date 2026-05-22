## 第五章：开源 RAG 评估框架 —— DeepEval 与 TruLens

前四章用 LlamaIndex 内置评估器和手写代码完成了评估。但在生产环境中，你需要更专业的评估框架来做**系统化、可复现、持续集成**的评估。DeepEval 和 TruLens 是这个领域最主流的两个开源选择。

### 5.1 为什么需要专门的评估框架

```
  LlamaIndex 内置评估器:
    ✅ 与 LlamaIndex 无缝集成
    ✅ 零额外依赖
    ❌ 指标种类有限
    ❌ 没有可视化仪表盘
    ❌ 不支持 CI/CD 集成

  专业评估框架 (DeepEval / TruLens):
    ✅ 10+ 种评估指标 (覆盖 RAG、对话、毒性检测等)
    ✅ 可视化仪表盘 (逐条查看评估结果)
    ✅ CI/CD 集成 (每次代码改动自动跑评估)
    ✅ 可复现 (评估结果可导出、可对比)
    ❌ 引入额外依赖
    ❌ 学习成本
```

---

### 5.2 DeepEval —— 指标最全的 RAG 评估框架

#### 5.2.1 定位与特色

DeepEval 是一个专注于 LLM 系统评估的框架，提供了目前最全的开箱即用 RAG 指标集合。

```
  DeepEval 的 RAG 评估指标全景:

  检索相关:
    · ContextualRecall       → 相关 Chunk 有多少被找回了? (≈ Recall)
    · ContextualPrecision    → 检索到的 Chunk 有多少相关? (≈ Precision)
    · ContextualRelevancy    → 检索到的 Chunk 与 Query 相关吗?

  生成相关:
    · Faithfulness           → Answer 是否忠实于 Context?
    · AnswerRelevancy        → Answer 是否紧扣 Query?
    · Hallucination          → Answer 中是否存在幻觉?

  综合指标:
    · RAGAS                  → 一次跑完 Faithfulness + AnswerRelevancy 
                               + ContextPrecision + ContextRecall + ContextRelevancy
    · GEval                  → 自定义评估标准 (用 Prompt 定义)

  其他:
    · Toxicity               → 回答是否有毒?
    · Bias                   → 回答是否有偏见?
    · Summarization          → 摘要质量
```

#### 5.2.2 代码演示 —— 用 DeepEval 评估 RAG

```python
"""
═══════════════════════════════════════════════════════════════════════════
  DeepEval RAG 评估完整演示
═══════════════════════════════════════════════════════════════════════════

  安装: pip install deepeval
  需要 OPENAI_API_KEY (DeepEval 用 GPT-4 做裁判)
"""
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

# ═══════════════════════════════════════════════════════════════
# Step 1: 准备测试用例
# ═══════════════════════════════════════════════════════════════

# DeepEval 使用 LLMTestCase 封装一次查询的全部数据
test_case = LLMTestCase(
    # ── 基础数据 ──
    input="事假需要提前多久申请？",           # 用户 Query
    actual_output="事假需提前1个工作日向部门主管申请，经审批后交HR备案。"
                  "事假期间不发放工资。",     # LLM 实际生成的 Answer

    # ── 检索数据 (用于 Context 相关指标) ──
    retrieval_context=[
        "事假需提前1个工作日向部门主管申请，经审批后交HR备案。",
        "病假应在当日8:30前通知部门主管。连续超过2天须提供医院证明。",
        "年假按工龄计算：入职1-5年5天，5-10年10天，10年以上15天。",
    ],

    # ── Ground Truth (用于 Recall / Precision) ──
    expected_output="事假需提前1个工作日向部门主管申请，经审批后交HR备案。",

    # ── 上下文 Ground Truth (用于 ContextualRecall) ──
    # 知识库中"应该被检索到的"所有相关 Chunk
    expected_retrieval_context=[
        "事假需提前1个工作日向部门主管申请，经审批后交HR备案。",
    ],
)

# ═══════════════════════════════════════════════════════════════
# Step 2: 选择评估指标并运行
# ═══════════════════════════════════════════════════════════════

# 创建评估指标 (每个指标阈值默认=0.5)
metrics = [
    FaithfulnessMetric(threshold=0.7),        # Answer ← Context
    AnswerRelevancyMetric(threshold=0.7),     # Answer ← Query
    ContextualRecallMetric(threshold=0.7),    # 找回率
    ContextualPrecisionMetric(threshold=0.7), # 精确率
    ContextualRelevancyMetric(threshold=0.7), # 上下文相关性
    HallucinationMetric(threshold=0.7),       # 幻觉检测
]

# 运行评估
results = evaluate(
    test_cases=[test_case],
    metrics=metrics,
    # 可选: 打印详细报告
    print_results=True,
)

# ═══════════════════════════════════════════════════════════════
# Step 3: 查看结果
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  评估结果")
print("=" * 70)

for result in results:
    for metric_result in result.metrics_data:
        print(f"\n  {metric_result.name}:")
        print(f"    Score:  {metric_result.score:.3f}")
        print(f"    Passed: {metric_result.success}")
        if metric_result.reason:
            print(f"    Reason: {metric_result.reason[:150]}...")
```

#### 5.2.3 批量评估 + 基准测试

```python
"""
═══════════════════════════════════════════════════════════════════════════
  DeepEval 批量评估 —— 对比两个版本的 RAG 系统
═══════════════════════════════════════════════════════════════════════════
"""

# ── 准备多条测试用例 ─────────────────────────────────────
test_cases = [
    LLMTestCase(
        input="事假需要提前多久申请？",
        actual_output="事假需提前1个工作日申请...",
        retrieval_context=["事假需提前1个工作日向部门主管申请..."],
        expected_output="事假需提前1个工作日向部门主管申请。",
        expected_retrieval_context=["事假需提前1个工作日向部门主管申请..."],
    ),
    LLMTestCase(
        input="五险一金包括哪些？",
        actual_output="五险一金包括养老、医疗、失业、工伤、生育保险和住房公积金。",
        retrieval_context=["养老保险:公司16%个人8%...", "医疗保险:公司9.8%个人2%..."],
        expected_output="养老、医疗、失业、工伤、生育保险和住房公积金。",
        expected_retrieval_context=["养老保险:公司16%个人8%...", "医疗保险..."],
    ),
    LLMTestCase(
        input="加班费怎么计算？",
        actual_output="工作日加班1.5倍，休息日2倍，法定节假日3倍。",
        retrieval_context=["工作日加班按1.5倍工资计算...", "休息日加班按2倍..."],
        expected_output="工作日1.5倍，休息日2倍，法定节假日3倍。",
        expected_retrieval_context=["工作日加班按1.5倍...", "休息日加班按2倍...", "法定节假日按3倍..."],
    ),
]

# 一次评估所有
results = evaluate(test_cases, metrics)

# 汇总分数
print("\n  批量评估汇总:")
for metric in metrics:
    scores = []
    for r in results:
        for mr in r.metrics_data:
            if mr.name == metric.__class__.__name__:
                scores.append(mr.score)
    if scores:
        print(f"    {metric.__class__.__name__}: avg={sum(scores)/len(scores):.3f} "
              f"(min={min(scores):.3f}, max={max(scores):.3f})")
```

**DeepEval 的 RAGAS 综合指标：**

```python
# DeepEval 也支持 LlamaIndex 的集成
from deepeval.integrations.llama_index import DeepEvalAnswerRelevancyEvaluator

# 可以直接在 LlamaIndex 的评估流程中使用
evaluator = DeepEvalAnswerRelevancyEvaluator()
```

---

### 5.3 TruLens —— 可视化驱动的 RAG 评估与调试

#### 5.3.1 定位与特色

TruLens 的核心理念是 **"Feedback Functions"（反馈函数）** —— 每个反馈函数评估 RAG 系统的一个特定方面。它最大的特色是有可视化仪表盘，可以逐条查看评估结果。

```
  TruLens 的核心概念:

  反馈函数 (Feedback Functions):
    · Answer Relevance  → Query + Answer → 0~1 分数
    · Context Relevance → Query + Context → 0~1 分数
    · Groundedness      → Context + Answer → 0~1 分数 (≈ Faithfulness)

  记录器 (Recorder):
    每次查询自动记录: 输入 / 输出 / 检索结果 / 延迟 / 成本

  仪表盘 (Dashboard):
    truLens dashboard → 浏览器打开 → 看到每条查询的评估分数

  TruLens 的三件套 (RAG Triad):
    Answer Relevance  ← Answer ↔ Query      (回答是否扣题)
    Context Relevance ← Context ↔ Query     (检索是否相关)
    Groundedness      ← Answer ↔ Context    (回答是否有据)
```

#### 5.3.2 代码演示 —— TruLens 评估 LlamaIndex RAG

```python
"""
═══════════════════════════════════════════════════════════════════════════
  TruLens + LlamaIndex RAG 评估完整演示
═══════════════════════════════════════════════════════════════════════════

  安装: pip install trulens trulens-providers-openai
"""
from trulens.core import TruSession
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI as TruOpenAI
from llama_index.core import VectorStoreIndex

# ── 初始化 TruLens ────────────────────────────────────────
session = TruSession()

# TruLens 的 GPT 裁判
tru_provider = TruOpenAI(model="gpt-4o")

# ═══════════════════════════════════════════════════════════════
# Step 1: 定义反馈函数 (RAG Triad)
# ═══════════════════════════════════════════════════════════════

from trulens.core import Feedback
from trulens.feedback import Groundedness

# 初始化 Groundedness (需要先定义 provider)
grounded = Groundedness(provider=tru_provider)

# 反馈 1: Answer Relevance —— Answer 是否扣题
f_answer_relevance = Feedback(
    tru_provider.relevance_with_cot_reasons,  # 用 Chain-of-Thought 推理
    name="Answer Relevance"
).on_input_output()
# ↑ .on_input_output() = 需要 input(Query) + output(Answer)

# 反馈 2: Context Relevance —— Context 是否与 Query 相关
f_context_relevance = Feedback(
    tru_provider.context_relevance_with_cot_reasons,
    name="Context Relevance"
).on_input().on(TruLlama.select_source_nodes().text)
# ↑ 需要 input(Query) + 检索到的 source_nodes

# 反馈 3: Groundedness —— Answer 是否基于 Context
f_groundedness = Feedback(
    grounded.groundedness_measure_with_cot_reasons,
    name="Groundedness"
).on(TruLlama.select_source_nodes().text).on_output()
# ↑ 需要 source_nodes + output(Answer)

# 额外反馈 4: 延迟
f_latency = Feedback(
    lambda ret: ret, name="Latency"
).on(
    TruLlama.select_context().get_total_cost  # 查询总耗时
)

# ═══════════════════════════════════════════════════════════════
# Step 2: 用 TruLlama 包装你的 QueryEngine
# ═══════════════════════════════════════════════════════════════

index = VectorStoreIndex.from_documents(documents)  # 你的索引
query_engine = index.as_query_engine()

# TruLlama 包装 —— 之后每次查询都会被自动记录和评估
tru_engine = TruLlama(
    query_engine,
    app_name="EmployeeHandbook-RAG",   # 应用名称
    feedbacks=[
        f_answer_relevance,
        f_context_relevance,
        f_groundedness,
        f_latency,
    ],
)

# ═══════════════════════════════════════════════════════════════
# Step 3: 查询 —— 自动评估
# ═══════════════════════════════════════════════════════════════

queries = [
    "事假需要提前多久申请？",
    "加班费怎么计算的？",
    "五险一金包括哪些？",
    "报销的流程是怎样的？",
    "公司年假有多少天？",
]

for query in queries:
    response = tru_engine.query(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")

# 每次查询后，TruLens 自动:
#   1. 记录: 输入/输出/检索结果/延迟
#   2. 运行三个反馈函数 → 计算分数
#   3. 存入 SQLite 数据库

# ═══════════════════════════════════════════════════════════════
# Step 4: 启动可视化仪表盘
# ═══════════════════════════════════════════════════════════════
# 在终端运行: trulens dashboard
# 浏览器打开 → http://localhost:8501
# 可以看到:
#   · 每条查询的 Answer Relevance / Context Relevance / Groundedness 分数
#   · 时间趋势图 (分数是上升还是下降)
#   · Leaderboard (对比不同配置的 RAG 系统)

print(f"\n  ── TruLens 仪表盘 ──")
print(f"  在终端运行: trulens dashboard")
print(f"  浏览器打开: http://localhost:8501")
print(f"  数据存储在: {session.connector.db_file}")
```

#### 5.3.3 TruLens 的可视化仪表盘

TruLens 最大的特色就是可视化。运行 `trulens dashboard` 后：

```
  仪表盘展示的内容:

  1. 应用概览
     总查询数 / 平均延迟 / 平均成本

  2. 每条查询的详细记录
     Q: "事假需要提前多久申请？"
     A: "事假需提前1个工作日..."
     Answer Relevance:  ★★★★☆  0.85
     Context Relevance: ★★★★★ 0.92
     Groundedness:      ★★★★☆ 0.78
     Latency: 234ms

  3. 时间趋势图
     如果你调整了 chunk_size 或换了 Embedding 模型,
     可以直观看到分数随时间的变化 —— 
     是变好了还是变坏了？

  4. Leaderboard (对比实验)
     如果你同时运行了两个配置的 RAG:
       · config_A: chunk_size=256, Reranker=on
       · config_B: chunk_size=512, Reranker=off

     Leaderboard 会并排显示两者的平均分数
     → 一眼看出哪个配置更好
```

---

### 5.4 DeepEval vs TruLens vs LlamaIndex 内置

| 维度 | DeepEval | TruLens | LlamaIndex 内置 |
|------|----------|---------|:---:|
| **指标数量** | 15+ (最全) | 3+ RAG Triad | 5 (Faithfulness/Relevancy/Correctness/MRR/HitRate) |
| **可视化仪表盘** | 有 (Web) | ✅ 有 (Web, 最丰富) | ❌ 无 |
| **CI/CD 集成** | ✅ 原生支持 | ✅ 支持 | ⚠ 需手动集成 |
| **学习曲线** | 中 (需理解 LLMTestCase) | 中 (需理解 Feedback Functions) | 低 (一行代码) |
| **与 LlamaIndex 集成** | ✅ 有集成 | ✅ TruLlama 专用包装 | ✅ 原生 |
| **自定义指标** | ✅ GEval (Prompt定义) | ✅ 自定义 Feedback 函数 | ⚠ 有限 |
| **适用阶段** | 深度测评、上线前验证 | 持续监控、实验对比 | 快速原型验证 |
| **安装** | `pip install deepeval` | `pip install trulens` | 无需额外安装 |

**选型建议：**

```
  你的需求 → 推荐

  快速验证 (刚写完代码试一下) → LlamaIndex 内置评估器
    理由: 零依赖，一行代码
    限制: 没有仪表盘，指标少

  深度测评 (上线前全面测试) → DeepEval
    理由: 指标最全 (15+)，覆盖所有 RAG 维度
    限制: 需要理解 LLMTestCase 的数据结构

  持续监控 (上线后持续追踪) → TruLens
    理由: 可视化仪表盘最丰富，时间趋势图一目了然
          每次查询自动记录 + 评估，零侵入
    限制: 指标数量少 (RAG Triad 三个)，但最核心的三个都覆盖了

  两者可以组合:
    DeepEval 做上线前的"全部指标走一遍"
    TruLens 做上线后的"持续监控核心指标"

  都不需要:
    手写代码 (4.8 节) + Excel/Grafana → 最灵活但最费时间
```
