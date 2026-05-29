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


---

## 5.5 大厂面试题深度讲解

### 5.5.1 框架对比与选型类

#### Q1：DeepEval、TruLens、LlamaIndex 内置评估器三者分别适用于什么场景？从快速原型到生产监控，评估工具链应该怎么演进？（字节跳动 / 腾讯 综合选型题）

**面试官考察点：** 评估工具链的分级选型——知道什么阶段用什么工具。

**回答框架——三级评估工具链的递进关系：**

**三者一句话定位：**

| 工具 | 定位 | 一句话 |
|------|------|--------|
| **LlamaIndex 内置** | 轻量级、零依赖 | 开发中快速验证——一行代码看 MRR/Hit Rate |
| **DeepEval** | 指标最全的深度测评 | 上线前压力测试——15+ 指标全面覆盖 |
| **TruLens** | 可视化持续监控 | 上线后持续追踪——每次查询自动记录+评估 |

**三阶段评估工具演进路径：**

```
Phase 1（开发迭代，周级别）→ LlamaIndex 内置评估器
  目标：快速验证每次改动的效果
  频率：每次调参后跑一轮（每天 N 次）
  指标：MRR、Hit Rate（零 LLM 成本）
  工具：RetrieverEvaluator + FaithfulnessEvaluator
  时间：5 分钟一轮

Phase 2（上线前验证，月级别）→ DeepEval
  目标：全面覆盖所有 RAG 维度的深度测评
  频率：准备上线前跑一次完整评估
  指标：ContextualRecall/Precision/Relevancy + Faithfulness + AnswerRelevancy + Hallucination
  工具：DeepEval 的 evaluate() + 50-100 条 LLMTestCase
  时间：30-60 分钟（含 LLM 裁判打分）

Phase 3（线上监控，持续运行）→ TruLens
  目标：持续追踪线上质量，快速发现退化
  频率：每次线上查询自动记录
  指标：RAG Triad（Answer Relevance / Context Relevance / Groundedness）+ 延迟
  工具：TruLlama 包装 QueryEngine + TruLens Dashboard
  时间：零侵入（查询时自动评估）
```

**面试官追问："三个工具能不能只用其中一个？"**

"可以但会有盲区。只用一个的代价：
- 只用 LlamaIndex 内置 → 指标少、无仪表盘、无法做历史对比 → 上线后出问题不知道什么时候开始退化的
- 只用 DeepEval → 可以做深度测评，但没有持续监控 → 上线后文档库更新了/用户 query 分布变了 → 评估基线失效了也不知道
- 只用 TruLens → 可以持续监控，但只有 3 个指标 → 如果 Recall 下降导致 Groundedness 低，TruLens 能告诉你'回答质量下降了'但无法定位到'是检索漏了'还是'LLM 编造了'

最佳实践：三个都用——LlamaIndex 做开发迭代、DeepEval 做上线验证、TruLens 做线上监控。三者分工明确，互不替代。"

---

#### Q2：TruLens 的 RAG Triad（Answer Relevance / Context Relevance / Groundedness）为什么选择这三个指标？它们和 RAG 的三层架构有什么对应关系？（阿里巴巴 / 腾讯）

**面试官考察点：** 是否理解 TruLens 选择这三个指标的设计哲学——它们恰好对应 RAG 的三个核心环节。

**回答框架——RAG Triad 的三指标对应 RAG 三层架构：**

**三层架构与三指标的精确对应：**

```
RAG 三层架构              TruLens RAG Triad          衡量的是什么
─────────────────────────────────────────────────────────────
① 检索层（Retriever）     Context Relevance          检索到的 Chunk 与 Query 相关吗？
                          Context ↔ Query            衡量"检索质量"

② 生成层（LLM）          Groundedness               Answer 是否基于 Context？
                          Answer ↔ Context           衡量"生成忠实度"

③ 端到端（整体体验）       Answer Relevance          Answer 是否紧扣 Query？
                          Answer ↔ Query             衡量"整体用户体验"
```

**三者之间的逻辑链——"一个环节出问题，下游全受影响"：**

```
Context Relevance 低：
  → 检索返回了不相关的内容
  → 即使 LLM 忠实于 Context（Groundedness 可能高）
  → Answer 也与 Query 脱节（Answer Relevance 低）
  → 根因：检索环节

Context Relevance 高 + Groundedness 低：
  → 检索找到了正确的内容
  → 但 LLM 没有忠实使用——在编造或自由发挥
  → Answer 可能跑题或幻觉
  → 根因：生成环节（Prompt 约束不够 / temperature 过高）

Context Relevance 高 + Groundedness 高：
  → 检索正确 + LLM 忠实
  → Answer Relevance 应该高
  → 如果 Answer Relevance 仍然低 → Context 找到了"相关但不完整"的信息
  → LLM 忠实于不完整的信息 → 回答没有完全解决用户问题
```

**为什么恰好选了这三个——"最小必要集"：**

```
TruLens 的设计哲学：用最少的指标覆盖 RAG 最关键的三环。

为什么没有 Recall / Precision？
  → Recall 需要全量标注（太贵），不适合"每次查询自动评估"
  → Context Relevance 是 Recall/Precision 的 LLM-as-Judge 近似

为什么没有 NDCG / MRR？
  → 排序质量指标需要 Ground Truth（太贵）
  → Context Relevance 只判断"相关/不相关"（二值判断）
  → Groundedness 辅以检查 LLM 是否用到了返回的 Context

三个指标的标注成本为零（全部 LLM 自动判断），
但能覆盖 RAG 系统 80% 的质量问题。
这就是"最小必要集"的设计哲学。
```

---

### 5.5.2 工程实践类

#### Q3：DeepEval 的 LLMTestCase 需要准备哪些数据？如果知识库有 10 万条 Chunk，如何为 DeepEval 准备评估数据集？（拼多多 / 美团 工程实践题）

**面试官考察点：** 评估数据准备是生产中最耗时的环节。这道题考察务实的工程经验。

**回答框架（先讲 LLMTestCase 结构，再讲大规模数据集的准备策略）：**

**LLMTestCase 的完整数据结构——五个字段对应四个评估角色：**

```python
LLMTestCase(
    input="事假需要提前多久申请？",         # 用户 Query（必填）
    actual_output="事假需提前1个工作日...",  # LLM 实际生成（必填）
    
    retrieval_context=[                    # 检索返回的上下文
        "事假需提前1个工作日申请...",        # 实际检索结果
        "病假应在8:30前通知...",
    ],
    
    expected_output="事假需提前1个工作日...", # 参考答案（用于评估 Correctness）
    
    expected_retrieval_context=[           # 应检索到的上下文（用于 Recall/Precision）
        "事假需提前1个工作日申请...",        # Ground Truth
    ],
)
```

**每个字段与评估指标的对应关系：**

```
ContextualRecall    → retrieval_context vs expected_retrieval_context
ContextualPrecision → retrieval_context vs expected_retrieval_context  
Faithfulness        → actual_output vs retrieval_context
AnswerRelevancy     → actual_output vs input
Hallucination       → actual_output vs retrieval_context
```

**10 万条 Chunk 的评估数据集准备策略——不可能全部标注：**

```
策略一（推荐）：分层抽样 + 混合标注

  Step 1：按文档类别分层抽样 20 份文档（覆盖 HR/财务/技术等各类别）
  Step 2：每份文档用 generate_question_context_pairs 自动生成 5 条 QA
  Step 3：人工从 20×5=100 条中筛选 30 条，修正不合理问题，补充 5 条口语化 query
  Step 4：对最终 35 条做全量标注（expected_retrieval_context）
  
  成本：~5 小时人工标注 + ~$0.50 LLM 生成
  覆盖：35 条 query × 各类别 → 基本覆盖核心场景

策略二（大规模自动评估——适合需要 100+ 条评估集的场景）：
  
  全部用 generate_question_context_pairs 生成 200 条
  人工快速扫描一遍（不做精标注）
  只用 ContextRelevancy + Faithfulness + AnswerRelevancy（不需要 expected_output）
  这三个指标完全不需要 Ground Truth → LLM 自动打分
  
  成本：~$1 LLM 生成 + ~30 分钟人工扫描
  覆盖：200 条 → 数量上有统计显著性
  劣势：缺少 Recall/Precision（需要 expected_retrieval_context 才能算）
```

**面试官追问："expected_retrieval_context 怎么标？每条 query 都要看完 10 万个 Chunk 吗？"**

"不需要。实际做法：
1. 先用 ANN 检索 Top-50 → 标注者只在这 50 个中标注哪些是'真正相关的'——而不是浏览全部 10 万个
2. 这等价于 Pool-based 标注——假设 ANN 已经筛掉了大部分明显不相关的内容，正确答案大概率在这 50 个中
3. 这种标注方式下 Recall 的计算变为 Recall@Pooled——衡量 'ANN 粗排后的 Recall' 而非 '全库 Recall'
4. 虽然不是完美的 Recall@All，但工程上是可操作的折中方案"

---

#### Q4：如何把 RAG 评估集成到 CI/CD 流水线中？每次代码改动自动跑评估，具体怎么做？（字节跳动 / 华为 DevOps 实践题）

**面试官考察点：** 评估的自动化——从"开发完跑一次"到"每次 push 自动跑"。

**回答框架——CI/CD 集成评估的四层设计：**

**第一层——快速冒烟测试（每次 commit，< 2 分钟）：**

```yaml
# .github/workflows/rag-eval-smoke.yml
# 每次 push 自动触发，只跑 10 条 query + 2 个关键指标
# 目标：快速发现"致命错误"（检索返回空、LLM 返回 None 等）

name: RAG Smoke Test
on: [push]

jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install deepeval llama-index
      - name: Run smoke test (10 queries)
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python tests/rag_smoke_test.py
      - name: Check thresholds
        run: |
          # ContextRelevancy 必须 > 0.5（检索至少返回了相关内容）
          # Faithfulness 必须 > 0.6（LLM 没有严重编造）
          python -c "
          import json
          with open('eval_results.json') as f:
              results = json.load(f)
          assert results['avg_context_relevancy'] > 0.5, '⚠ 检索失败！'
          assert results['avg_faithfulness'] > 0.6, '⚠ 幻觉严重！'
          print('✓ 冒烟测试通过')
          "
```

**第二层——完整评估（PR 合并前，< 15 分钟）：**

```yaml
# 跑 50 条 query + 全部指标
# 对比 PR 前后的分数变化 → 分数下降 > 5% → 阻断合并

name: RAG Full Evaluation
on: [pull_request]

jobs:
  full-eval:
    runs-on: ubuntu-latest
    steps:
      - name: Run full evaluation (50 queries, all metrics)
        run: python tests/rag_full_eval.py
      - name: Compare with baseline
        run: |
          # 如果 Faithfulness 下降 > 5%，阻断 PR 合并
          python scripts/compare_eval.py \
            --baseline eval_baseline.json \
            --current eval_results.json \
            --threshold 0.05
```

**第三层——性能回归检测：**

```python
# 除了质量指标，还要检测性能回归
LATENCY_THRESHOLDS = {
    "p50": 500,    # P50 延迟必须 < 500ms
    "p95": 1000,   # P95 延迟必须 < 1s
    "p99": 2000,   # P99 延迟必须 < 2s
}

# 如果延迟超过阈值 → 同样阻断 PR
```

**第四层——基线管理：**

```
基线存储：每次 main 分支 merge 后，跑一次完整评估 → 保存为 eval_baseline.json
PR 对比：每次 PR 跑评估 → 对比 eval_baseline.json → 分数下降 > 阈值 → 阻断

基线更新策略：
  · 手动触发（不自动更新）——需要人工确认新基线是可接受的
  · 基线 ⚓ = 当前生产环境的"质量锚点"
  · 任何 PR 的质量不能低于基线的 95%
```

**面试官追问："如果评估分数波动很大（同一配置跑两次分数不同），CI/CD 不是会频繁误报吗？"**

"这确实是 LLM-as-Judge 的固有问题。应对策略：
1. **多次运行取中位数**——每个指标跑 3 次，取中位数，减少 LLM 随机性的影响
2. **放宽阈值**——不从 5% 开始，先从 10% 开始——适应 LLM 评分的自然波动
3. **引入统计检验**——用 Mann-Whitney U 检验比较新旧分数分布是否有显著差异——而非直接比平均值
4. **人工复核机制**——CI 阻断后自动发 Slack 通知，人工判断是真退化还是随机波动"

---

### 5.5.3 面试高频知识点速查

#### 一句话答案系列

| 问题 | 一句话答案 |
|------|-----------|
| DeepEval 的最大优势？ | 指标最全（15+），覆盖 RAG 所有维度，且支持 CI/CD 集成 |
| TruLens 的最大优势？ | 可视化仪表盘 + 零侵入持续监控——每次查询自动记录+评估 |
| LlamaIndex 内置评估器什么时候用？ | 开发快速迭代——零依赖、一行代码、MRR/Hit Rate 零 LLM 成本 |
| RAG Triad 是哪三个指标？ | Answer Relevance（回答扣题）、Context Relevance（检索相关）、Groundedness（回答有据） |
| RAG Triad 为什么恰好是三个？ | 三个指标恰好对应 RAG 三层架构——检索层/生成层/端到端 |
| LLMTestCase 最关键的字段？ | retrieval_context（实际检索结果）和 expected_retrieval_context（应检索的结果） |
| 10 万条 Chunk 怎么准备评估集？ | 分层抽样 20 份文档 → 自动生成 100 条 QA → 人工精选 30-35 条做全量标注 |
| CI/CD 中评估的防误报策略？ | 多次运行取中位数 + 放宽阈值 + 统计检验 + 人工复核 |

#### 三框架全景速查表

| 维度 | LlamaIndex 内置 | DeepEval | TruLens |
|------|:---:|:---:|:---:|
| **指标数** | 5 | 15+ | 3+ |
| **仪表盘** | ✗ | 有 | ✅ 最丰富 |
| **CI/CD** | ⚠ 需手写 | ✅ 原生 | ✅ 支持 |
| **学习曲线** | 低 | 中 | 中 |
| **上线前测评** | ★★ | ★★★★★ | ★★★ |
| **线上监控** | ✗ | ★★ | ★★★★★ |
| **快速迭代** | ★★★★★ | ★★★ | ★★★ |
| **安装** | 0 依赖 | `pip install deepeval` | `pip install trulens` |

---
