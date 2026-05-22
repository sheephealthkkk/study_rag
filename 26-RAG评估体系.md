## 第四章：RAG 评估体系 —— 检索质量与生成质量的度量

前三章讲了怎么构建检索管线。但你怎么知道你的检索管线"好"还是"不好"？你换了 Embedding 模型、调了 chunk_size、加了 Reranker——这些改动到底有没有用？

评估体系就是回答这些问题的。它用**数据**而非**直觉**来衡量 RAG 系统的质量。

### 4.1 评估的四个基础要素

任何一个 RAG 评估都需要四个数据：

```
═══════════════════════════════════════════════════════════════════════
              RAG 评估的四个基础要素
═══════════════════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────────────────┐
  │  1. Question (用户问题)                                       │
  │     例: "事假需要提前多久申请？"                                │
  │     来源: 人工编写 / 用户日志 / LLM 自动生成                   │
  │     要求: 覆盖各种查询类型 (短查询/长查询/精确编码/模糊语义)     │
  ├───────────────────────────────────────────────────────────────┤
  │  2. Ground Truth (参考答案 / 标准答案)                         │
  │     例: "事假需提前1个工作日向部门主管申请。"                     │
  │     来源: 人工标注 (最可靠) / LLM 生成 (需人工审核)              │
  │     要求: 准确、完整、可追溯 (能对应到具体的 Chunk)               │
  ├───────────────────────────────────────────────────────────────┤
  │  3. Context (检索到的上下文)                                    │
  │     例: [Node("事假需提前1个工作日..."), Node("病假应在8:30前...")] │
  │     来源: 你的 RAG 检索系统实际返回的 Chunk                     │
  │     用途: 评估"检索"是否有用——Context 是否包含了回答问题所需信息？│
  ├───────────────────────────────────────────────────────────────┤
  │  4. Response Answer (RAG 系统的生成回答)                        │
  │     例: "根据公司规定，事假需提前1个工作日申请。"                  │
  │     来源: 你的 RAG 系统用 LLM 生成的最终回答                    │
  │     用途: 评估"生成"是否有用——Answer 是否正确？是否忠实于Context？│
  └───────────────────────────────────────────────────────────────┘

  这四者之间的评估关系:

         Question ──▶ Context (检索阶段) ──▶ Answer (生成阶段)
            │              │                      │
            │         评估指标:                评估指标:
            │      Recall / Precision         Faithfulness
            │      Hit Rate / MRR            Answer Relevance
            │      Context Relevance          Correctness
            │
            └────── Ground Truth (用于对比) ──────────┘
```

---

### 4.2 generate_question_context_pairs —— 自动生成评估数据集

手写评估数据集（Question + Ground Truth）是最可靠的，但也是最贵的——每条问题需要领域专家认真标注。LlamaIndex 提供了 `generate_question_context_pairs` 来自动生成评估数据，降低评估门槛。

#### 4.2.1 工作原理与流程

```
═══════════════════════════════════════════════════════════════════════
        generate_question_context_pairs 内部流程
═══════════════════════════════════════════════════════════════════════

  输入: 你的知识库中的所有 Node/Chunk

  Step 1: 遍历每个 Node
    对每个 Chunk，用 LLM 生成一个"可以用这个 Chunk 回答"的问题

    Prompt 模板 (简化):
      "下面是一段文档内容:
       {node_text}
       
       请基于这段内容生成一个用户可能提出的问题。
       这个问题应该能够用这段内容来回答。
       只输出问题本身，不要输出答案。"

    LLM 输出: "事假需要提前多久申请？"

  Step 2: 自动生成 Ground Truth
    生成问题的同时，将该 Node 的文本作为"参考答案"

    → 也就是: Question = LLM生成的问题, Ground Truth = 该Node的原文

  Step 3: 输出评估数据集
    每条数据 = {
      "question": "事假需要提前多久申请？",       ← LLM生成的
      "ground_truth": "事假需提前1个工作日...",    ← 该Node的原文
      "source_node_id": "node_001"                ← 来源Node的ID
    }
    
  输出: List[QuestionGroundTruthPair]
═══════════════════════════════════════════════════════════════════════
```

#### 4.2.2 优点与缺点

| 优点 | 缺点 |
|------|------|
| ✅ 速度快——每条Node几秒，100条Node几分钟 | ❌ LLM 生成的问题可能不自然（"请根据以下内容造句"） |
| ✅ 成本低——比人工标注便宜 100 倍 | ❌ Ground Truth = Node原文 → 假设"整个Node都是正确答案" |
| ✅ 覆盖全——每个Node都有对应问题，不会漏 | ❌ 生成的问题偏向"事实提取"型 → 缺少复杂推理/多跳问题 |
| ✅ 适合初始评估——快速了解检索基线 | ❌ 如果Node本身质量差 → 生成的问题也差 |

#### 4.2.3 代码演示

```python
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.llms.openai import OpenAI

# ── 自动生成评估数据集 ────────────────────────────────────
# 用 GPT-4o 为每个 Node 生成一个相关问题
qa_pairs = generate_question_context_pairs(
    nodes=nodes,                    # 你的知识库 Node 列表
    llm=OpenAI(model="gpt-4o"),
    num_questions_per_chunk=1,      # 每个 Node 生成 1 个问题
    max_questions=50,               # 最多生成 50 个问题 (限制成本)
)

print(f"生成了 {len(qa_pairs)} 条评估数据")
for i, pair in enumerate(qa_pairs[:3]):
    print(f"\n  数据 {i+1}:")
    print(f"    Question: {pair.question}")
    print(f"    Ground Truth: {pair.context[:80]}...")
    print(f"    Source Node ID: {pair.source_node.node_id[:12]}...")
```

---

### 4.3 评估的两大维度

```
═══════════════════════════════════════════════════════════════════════
        评估维度全景
═══════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │  维度 1: 检索质量 (Retrieval Quality)                           │
  │                                                                 │
  │  回答的问题: "检索系统有没有找到正确的信息？"                     │
  │                                                                 │
  │  评估对象: Context (检索到的 Chunk) vs Ground Truth (正确答案)   │
  │  不需要: LLM 生成的 Answer                                       │
  │                                                                 │
  │  指标: Recall / Precision / Hit Rate / MRR / Context Relevance  │
  ├─────────────────────────────────────────────────────────────────┤
  │  维度 2: 生成质量 (Generation Quality)                           │
  │                                                                 │
  │  回答的问题: "LLM 基于检索到的信息生成的回答是否正确？"           │
  │                                                                 │
  │  评估对象: Answer (LLM回答) vs Ground Truth (参考答案)           │
  │  需要: Context 作为"中间桥梁" (判断 Answer 是否忠实于 Context)   │
  │                                                                 │
  │  指标: Faithfulness / Answer Relevance / Correctness            │
  └─────────────────────────────────────────────────────────────────┘

  为什么评估要分两个维度?

    检索好 ≠ 回答好。检索到的信息是对的，但 LLM 可能"读错"了。
    回答好 ≠ 检索好。LLM 可能用自己的知识 (而非检索到的知识) 回答了问题。
    
    所以需要分别评估——先看检索有没有找到对的信息，再看 LLM 有没有用对。
═══════════════════════════════════════════════════════════════════════
```

---

### 4.4 检索质量指标 —— 无序评估详解

本节聚焦检索质量的四个核心指标。它们都是"无序评估"——不按检索器给出的分数（score）来判断质量，只看"哪些被找回了、哪些没被找回"。

#### 4.4.0 理解每个指标需要的"数据"

在开始之前，先明确每个指标是**对什么数据做评估**：

| 指标 | 需要的数据 | 数据长什么样 |
|------|-----------|------------|
| Recall | 每条Query → 知识库中"所有相关的Chunk ID列表"（全量标注） | Query1 → [chunk_0, chunk_15, chunk_23] |
| Precision | 每条Query → 检索系统返回的Top-K Chunk ID列表 + 其中哪些是相关的 | 返回[chunk_0, chunk_8, chunk_15] → 其中2个相关 |
| Hit Rate | 每条Query → 知识库中"至少1个相关的Chunk ID"（部分标注即可） | Query1 → chunk_0 是相关的 |
| Context Relevance | 每条Query → LLM 对检索结果的评判 | 不需要人工标注! |

**关键区别：** Recall 需要你标注"所有相关的Chunk"（全量标注，最贵），Hit Rate 只需要你标注"至少1个相关Chunk"（部分标注，便宜）。Context Relevance 不需要人工标注，用 LLM 做裁判。

---

#### 4.4.1 Recall（召回率）—— 衡量"找得全不全"

```
═══════════════════════════════════════════════════════════════════════
              Recall —— "该找的都找到了吗？"
═══════════════════════════════════════════════════════════════════════

  定义:
    Recall = 检索到的相关 Chunk 数 / 知识库中所有相关的 Chunk 数

  公式:
    Recall = TP / (TP + FN)
    
    TP (True Positive):  检索系统找回了、确实相关的 Chunk
    FN (False Negative): 检索系统没找回、但实际相关的 Chunk

  针对什么数据:
    需要: 每条 Query → 知识库中 ALL 相关 Chunk 的列表
    这是最贵的标注 —— 你需要从头到尾看完所有 Chunk, 
    标注出每一个"和这条Query相关"的。

  具体例子 (Query: "请假有哪些类型？"):

    知识库共有 20 个 Chunk:
    ┌────┬─────────────────────────────────┬──────────┐
    │ ID │ 内容                             │ 是否相关？│
    ├────┼─────────────────────────────────┼──────────┤
    │ C0 │ 公司实行弹性工作制                 │    否    │
    │ C1 │ 事假需提前1个工作日申请            │    是 ←  │
    │ C2 │ 病假应在当日8:30前通知             │    是 ←  │
    │ C3 │ 年假按工龄计算：1-5年5天           │    是 ←  │ ← 总共 5 个相关
    │ C4 │ 婚假为3天                         │    是 ←  │
    │ C5 │ 产假按国家规定执行                 │    是 ←  │
    │ C6 │ 五险一金缴纳比例                   │    否    │
    │ C7 │ 住宿标准一线城市500元/天            │    否    │
    │ ... │ ...                              │   ...    │
    └────┴─────────────────────────────────┴──────────┘

    检索系统返回 Top-5: [C1, C2, C3, C6, C8]
                           是   是   是   否   否
    
    TP = 3 (C1, C2, C3 被找回了)
    FN = 2 (C4, C5 没被找回)
    
    Recall = 3 / (3+2) = 3/5 = 60%

  这意味着什么:
    Recall = 60% → "5个相关Chunk只找到了3个，漏了2个(c4婚假, c5产假)"
    Recall = 100% → "所有相关Chunk都找到了"
    Recall = 0% → "一个都没找到"

  为什么 Recall 重要:
    · 漏掉的信息 LLM 永远看不到 → 无法生成完整回答
    · 如果 C4(婚假) 和 C5(产假) 没被检索到, 
      LLM 的回答就只会说 "请假有三种: 事假、病假、年假" —— 少了两种!
```

---

#### 4.4.2 Precision（精确率）—— 衡量"搜得准不准"

```
═══════════════════════════════════════════════════════════════════════
              Precision —— "返回的有多少是真正有用的？"
═══════════════════════════════════════════════════════════════════════

  定义:
    Precision = 检索到的相关 Chunk 数 / 检索返回的所有 Chunk 数

  公式:
    Precision = TP / (TP + FP)
    
    TP (True Positive):  检索到、确实相关的
    FP (False Positive): 检索到、但不相关的 (噪声!)

  针对什么数据:
    需要: 每条 Query → 检索返回的 Top-K Chunk 中哪些是相关的
    比 Recall 标注便宜 —— 只需要标注返回的 K 个, 不用标注全部 Chunk

  同一个例子 (Query: "请假有哪些类型？"):

    检索返回 Top-5: [C1, C2, C3, C6, C8]
                      相关 相关 相关 无关 无关
                      
    TP = 3
    FP = 2 (C6"五险一金", C8"住宿标准")
    
    Precision = 3 / (3+2) = 3/5 = 60%

  这意味着什么:
    Precision = 60% → "返回的5个结果中只有3个是真正相关的，2个是噪声"
    Precision = 100% → "返回的全部都是相关的，没有一条噪声"
    Precision = 20% → "5个结果中只有1个相关，4个是噪声——严重!"

  为什么 Precision 重要:
    · 噪声 Chunk 挤占了 Top-K 的位置 → 真正相关的 Chunk 可能被挤出 Top-5
    · LLM 看到无关内容 → 被噪声干扰 → 回答跑偏
    · 噪声增加了 LLM 的输入 token → 提高了成本但无产出
```

---

#### 4.4.3 Hit Rate（命中率）—— 衡量"找不找得到"

```
═══════════════════════════════════════════════════════════════════════
              Hit Rate —— "至少找到一个相关的结果了吗？"
═══════════════════════════════════════════════════════════════════════

  定义:
    Hit Rate = 至少命中了 1 个相关 Chunk 的 Query 数 / 总 Query 数

  不同于 Recall 和 Precision (单条Query的计算):
    Hit Rate 是在多条 Query 上的聚合指标 —— 看的是"有多少条Query至少找到了1个相关结果"

  针对什么数据:
    需要: 每条 Query → 至少 1 个相关的 Chunk ID
    这是最便宜的标注 —— 不需要标注"所有相关Chunk"，只需标注"至少1个"
    甚至可以不用人工标注: 直接用 LLM 判断 "这个 Chunk 能否回答这个问题?"

  具体例子 (5条 Query, 每次返回 Top-5):

    Query 1 "事假提前多久":    返回包含 C1(相关) → 命中 ✓
    Query 2 "加班费怎么算":    返回包含 C10(相关) → 命中 ✓
    Query 3 "报销流程怎样":    返回包含 C15(相关) → 命中 ✓
    Query 4 "竞业限制多久":    返回的5个都不相关 → 未命中 ✗
    Query 5 "公司成立时间":    返回包含 C20(相关) → 命中 ✓

    Hit Rate = 4 / 5 = 80%

  这意味着什么:
    Hit Rate = 80% → "80%的查询至少找到了一个相关结果"
    Hit Rate = 100% → "每条查询都至少有一个相关结果" (理论上可能, 实际很难)
    Hit Rate = 50% → "一半的查询什么都没找到" (严重问题!)

  为什么 Hit Rate 重要且实用:
    · 标注成本最低 —— 不需要全量标注
    · 直接反映用户体验 —— "用户问了10个问题，几个能找到答案？"
    · 适合快速评估 —— 生成100条问题 → 跑检索 → 用LLM评判 → 拿到Hit Rate

  Hit Rate 的"宽松"和"严格"两种计算:

    宽松 (Hit Rate@K):  Top-K 中至少有 1 个相关 → 就算命中
      K=5, 返回5个结果 → 只要有1个相关就行
    
    严格 (Hit Rate@1):  只有排名第 1 的结果相关 → 才算命中
      → 这个衡量的是"系统能不能一次给对"
```

---

#### 4.4.4 Context Relevance（上下文相关性）—— 衡量"检索到的内容是否有用"

```
═══════════════════════════════════════════════════════════════════════
        Context Relevance —— "检索到的内容真的能帮助回答问题吗？"
═══════════════════════════════════════════════════════════════════════

  定义:
    Context Relevance = 检索到的 Chunk 中"真正对回答有用的"的比例

  与前三个指标的核心区别:
    Recall / Precision / Hit Rate 都需要人工标注"相关 vs 不相关"。
    Context Relevance 不需要人工标注——它用 LLM 做裁判!

  针对什么数据:
    只需要: Query + 检索到的 Context (Chunk列表)
    评测方式: LLM (作为裁判) 读取 Query + Context → 判断"这些Context能否回答这个问题？"

  LLM 评判 Prompt (简化):
    "用户问题: {query}
     
     以下是从知识库中检索到的文档片段:
     [1] {chunk_1_text}
     [2] {chunk_2_text}
     [3] {chunk_3_text}
     
     请逐条判断这些片段是否与用户问题相关。输出 1 (相关) 或 0 (不相关):"

    LLM 输出: [1, 1, 0]  ← 第1,2条相关, 第3条无关
    → Context Relevance = 2/3 = 67%

  同一例子, Context Relevance vs Precision 的区别:

    检索返回 Top-3: [C1(事假), C2(病假), C6(五险一金)]
    
    Precision (需要人工标注):
      事先人工判断 C1 → 相关, C2 → 相关, C6 → 不相关
      Precision = 2/3 = 67%

    Context Relevance (LLM 自动评判):
      LLM 读取 Query(请假有哪些类型?) + 三个Chunk
      LLM 判断: C1→相关(1), C2→相关(1), C6→不相关(0)
      Context Relevance = 2/3 = 67%

    在这个例子中两者一致。但实际中可能不一致:
      · LLM 可能把 C6 也判为相关 (因为"五险一金是公司福利, 和请假话题不那么远")
      · 人工可能更严格 (认为 C6 与"请假类型"完全无关)
      → LLM 评分的"宽容度"取决于 Prompt 怎么写

  优点: 不需要人工标注, 成本最低, 可以大规模评估
  缺点: LLM 评判本身有误差 (LLM 也有判断不准的时候)
```

---

### 4.5 四个指标的区别与联系 —— 一张表看清

```
═══════════════════════════════════════════════════════════════════════
        检索质量四指标对比总表
═══════════════════════════════════════════════════════════════════════

  指标     │  问的问题           │  针对什么数据       │  计算方式          │  标注成本
  ─────────┼────────────────────┼───────────────────┼──────────────────┼──────────
  Recall   │ "该找的都找到了吗？" │ 知识库中ALL相关Chunk│ TP/(TP+FN)        │  最高
           │                    │ (全量标注)          │ 相关中找回了多少    │  (需全量标注)
  ─────────┼────────────────────┼───────────────────┼──────────────────┼──────────
  Precision│ "返回的有多少有用？" │ 检索返回的Top-K      │ TP/(TP+FP)        │  中等
           │                    │ (只标注返回的)       │ 返回的有多少相关    │  (只标注K个)
  ─────────┼────────────────────┼───────────────────┼──────────────────┼──────────
  Hit Rate │ "至少找到一个了吗？"│ 每条Query至少1个相关 │ 命中的Query数/总Query│ 最低
           │                    │ (部分标注或LLM评判)  │                    │  (可LLM自动)
  ─────────┼────────────────────┼───────────────────┼──────────────────┼──────────
  Context  │ "检索到的是否有用？" │ 检索返回的Top-K      │ LLM评判(0/1)      │  零
  Relevance│                    │ (LLM自动评判)       │ 相关条数/返回条数   │  (不需标注)
  ─────────┴────────────────────┴───────────────────┴──────────────────┴──────────

  同一个 Query 的四个指标同时计算 (Query: "请假有哪些类型？"):

    知识库: 20个Chunk, 其中5个相关 (C1事假, C2病假, C3年假, C4婚假, C5产假)
    检索返回 Top-5: [C1(相关), C2(相关), C3(相关), C6(无关), C8(无关)]

    Recall    = TP/(TP+FN) = 3/(3+2) = 60%
    → "5个相关Chunk中找回了3个，漏了婚假和产假"

    Precision = TP/(TP+FP) = 3/(3+2) = 60%
    → "返回的5个Chunk中3个相关，2个是噪声"

    Hit Rate  = 至少1个相关? = Yes
    → "这次查询至少找到了1个相关结果"

    Context Relevance = LLM评判 = 2/3 (如果LLM认为C6也相关)
    或 = 3/5 = 60% (如果LLM判断精确)

  为什么四个指标要一起看?

    · 高Recall + 低Precision: 找得全但混入很多噪声
      → Reranker 最有效 (把相关的排到前面, 噪声排到后面)

    · 高Precision + 低Recall: 返回的都是相关的但漏了很多
      → 增大top_k / 混合检索 / 多路召回

    · Hit Rate=90% + Recall=40%: 大部分查询能找到东西, 但找不全
      → chunk_size 太大 → 语义稀释 → 增大top_k或减小chunk_size

    · Hit Rate=50% + Context Relevance很高: 有一半查询完全找不到
      → 知识库覆盖不全 / Embedding模型不合适 / 切分策略需调整
═══════════════════════════════════════════════════════════════════════
```

---

### 4.6 总结：评估的实践路径

```
  评估的推荐步骤 (从便宜到贵):

  Step 1: Context Relevance (LLM自动, 零成本)
    跑 50-100 条查询 → 用 LLM 评判每条查询的检索结果是否相关
    → 如果 Context Relevance < 60% → 先不要往后评估, 调检索参数

  Step 2: Hit Rate (LLM自动 或 少量人工标注)
    每条查询标注"是否至少找到一个相关Chunk"
    → 如果 Hit Rate < 80% → 检查知识库覆盖和 Embedding

  Step 3: Precision (需标注返回的Top-K)
    每条查询标注"返回的K个Chunk中哪些相关"
    → 如果 Precision < 50% → 加 Reranker / 调整 chunk_size

  Step 4: Recall (需全量标注, 最贵)
    选取 10-20 条最重要的查询做全量标注
    → 这是最准确的评估, 但只在"确定要上线"时做
```


---

### 4.7 有序评估 —— 不仅看"找到了吗"，还要看"排得对不对"

前四个指标（Recall / Precision / Hit Rate / Context Relevance）都是"无序"的——它们不关心检索结果中的**排列顺序**。只要正确答案在 Top-K 中，不管排在第 1 位还是第 10 位，Hit Rate 都算"命中"。

但实际上**顺序非常重要**。如果正确答案排在第 5 位，而 LLM 只使用 Top-3 作为上下文，那这个正确答案虽然"被检索到了"，但 LLM 根本看不到它。

有序评估就是回答这个问题的：**正确答案排在什么位置？越靠前越好。**

---

#### 4.7.1 MRR（Mean Reciprocal Rank）—— 平均倒数排名

**定义：** MRR 衡量的是"第一个正确答案的平均排名位置"。

```
═══════════════════════════════════════════════════════════════════════
              MRR 的核心概念
═══════════════════════════════════════════════════════════════════════

  倒数 (Reciprocal): 数学中的倒数 = 1/数字

    排名第 1 位 → Reciprocal Rank = 1/1 = 1.000
    排名第 2 位 → Reciprocal Rank = 1/2 = 0.500
    排名第 3 位 → Reciprocal Rank = 1/3 = 0.333
    排名第 4 位 → Reciprocal Rank = 1/4 = 0.250
    排名第 5 位 → Reciprocal Rank = 1/5 = 0.200
    排名第 10 位 → Reciprocal Rank = 1/10 = 0.100
    找不到      → Reciprocal Rank = 0.000

  关键理解:
    · "倒数"是数学上的 1/排名，不是"排名的倒数"!!
    · 排名越靠前，RR 值越大
    · 从第 1 到第 2: RR下降 0.500 (从 1.0 到 0.5) —— 最大的降幅!
    · 从第 5 到第 6: RR下降 0.033 (从 0.2 到 0.167) —— 降幅逐渐变小
    · 这说明 MRR 极度强调"第一个正确答案排在第 1 位"

  Mean (平均): 对所有 Query 的 RR 取算术平均值

═══════════════════════════════════════════════════════════════════════
```

**MRR 与 Hit Rate 的组合使用：**

MRR 和 Hit Rate 必须一起看，因为它们回答的是不同的问题：

| | Hit Rate | MRR |
|------|----------|-----|
| 回答的问题 | "至少有没有找到？" | "找到的排在什么位置？" |
| 如果 HR 很低 | → 系统根本找不到答案，MRR 无意义 | |
| 如果 HR 很高但 MRR 低 | → 能找到，但排名靠后 | |
| 如果两者都高 | → 能找到，而且排在最前面 ✓ | |

**完整计算示例：**

```
═══════════════════════════════════════════════════════════════════════
              MRR 完整计算 —— 5 条 Query 的实例
═══════════════════════════════════════════════════════════════════════

  查询 1: "事假需要提前多久申请？"
  检索返回 Top-10:
    排名 1: "公司实行弹性工作制..."           ← 无关
    排名 2: "事假需提前1个工作日申请..."       ← 第一个正确答案! rank=2
    排名 3: "病假应在8:30前通知..."
    排名 4: "加班按1.5倍工资计算..."
    排名 5: "五险一金缴纳比例..."
    ...
    RR = 1/2 = 0.500

  查询 2: "加班费怎么计算？"
  检索返回 Top-10:
    排名 1: "工作日加班按1.5倍，休息日2倍，节假日3倍"  ← 第一个正确答案! rank=1
    排名 2: "公司实行弹性工作制..."
    ...
    RR = 1/1 = 1.000  ← 完美! 第一个就是正确答案!

  查询 3: "竞业限制的期限是多长？"
  检索返回 Top-10:
    排名 1-10: 全都是无关内容
    知识库中确实有相关内容，但没被检索到 (在排名 15 才出现)
    RR = 0.000  ← 在 Top-10 中没找到 (Hit Rate 也 = 0)

  查询 4: "报销流程是怎样的？"
  检索返回 Top-10:
    排名 1: "住宿标准一线城市500元/天..."
    排名 2: "餐饮补贴每日80元..."
    排名 3: "报销需在30日内提交..."            ← 第一个正确答案! rank=3
    RR = 1/3 = 0.333

  查询 5: "年假有多少天？"
  检索返回 Top-10:
    排名 1: "入职1-5年5天，5-10年10天，10年以上15天"  ← rank=1
    RR = 1/1 = 1.000

  ═══════════════════════════════════════════════════════

  MRR = (0.500 + 1.000 + 0.000 + 0.333 + 1.000) / 5
       = 2.833 / 5
       = 0.567

  解读:
    · 平均来看，第一个正确答案的排名约在 1/0.567 ≈ 第 1.76 位
    · Query 2 和 5 表现完美 (排名第 1)
    · Query 3 完全失败 (排名在 Top-10 之外)
    · Query 1 和 4 分别排名第 2 和第 3

  配合 Hit Rate:
    Hit Rate@10 = 4/5 = 80% (5条查询中4条找到了)
    MRR = 0.567
    → 结论: 大部分能找到，但平均排名约在第 2 位 → 可以接受
           如果 MRR < 0.3 → 虽然能找到，但排名太靠后 → 需要 Reranker

═══════════════════════════════════════════════════════════════════════
```

**MRR 的适用场景：** 你的应用只关心"第一个正确答案排在哪里"——比如搜索场景，用户只看第一页结果，第一个正确答案的位置决定了是否被看到。

**MRR 的局限：** 它**只看第一个正确答案**。如果一条 Query 有 5 个相关文档（多跳推理），MRR 只关心第 1 个在哪里，不关心另外 4 个是否也被找回了。这就是为什么还有 NDCG。

---

#### 4.7.2 NDCG（Normalized Discounted Cumulative Gain）

**定义：** NDCG 衡量的是"整个检索列表的质量"——不仅看第一个正确答案，而是看**所有相关文档都在什么位置**，并且排名越靠前贡献越大。

NDCG 的完整理解需要先拆解它的三个组成部分：CG → DCG → IDCG → NDCG。

##### 4.7.2.1 CG（Cumulative Gain）—— 累积收益

```
  CG@K = Σ relevance_i     (对前 K 个结果的"相关性分数"求和)
  
  例子 (相关性分数: 3 = 高度相关, 2 = 相关, 1 = 弱相关, 0 = 无关):

  检索返回 Top-5, 相关性标注为:
    排名 1: 高度相关 → relevance = 3
    排名 2: 无关     → relevance = 0
    排名 3: 高度相关 → relevance = 3
    排名 4: 弱相关   → relevance = 1
    排名 5: 无关     → relevance = 0

  CG@5 = 3 + 0 + 3 + 1 + 0 = 7

  问题: CG 不关心顺序! 
    把排名1(3分)和排名5(0分)交换 → CG还是7
    但实际上，用户更关注排名靠前的结果
```

##### 4.7.2.2 DCG（Discounted Cumulative Gain）—— 折损累积收益

DCG 在 CG 的基础上加了**位置折扣（Discount）**——排名越靠后，相关性分数的贡献越小。

```
  折扣函数: discount(rank) = 1 / log₂(rank + 1)
  
  通俗理解:
    排名 1 的相关性 → 权重 1.000  (不打折)
    排名 2 的相关性 → 权重 0.631  (打 63%)
    排名 3 的相关性 → 权重 0.500  (打 50%)
    排名 4 的相关性 → 权重 0.431  (打 43%)
    排名 5 的相关性 → 权重 0.387  (打 39%)
    排名 10 的相关性 → 权重 0.289 (打 29%)

  为什么用 log 而不用线性?
    · log 曲线前陡后平: 前几个位置的折扣大, 后面趋于平缓
    · 这模拟了用户行为: 用户对第 1 和第 2 之间的差别很敏感,
      但对第 9 和第 10 之间的差别不太敏感
```

**DCG 的计算：**

```
  DCG@K = Σ (relevance_i / log₂(i + 1))   for i = 1 to K

  同一个例子:

  排名 1: relevance=3 → 3 / log₂(1+1) = 3 / 1.000 = 3.000
  排名 2: relevance=0 → 0 / log₂(2+1) = 0 / 1.585 = 0.000
  排名 3: relevance=3 → 3 / log₂(3+1) = 3 / 2.000 = 1.500
  排名 4: relevance=1 → 1 / log₂(4+1) = 1 / 2.322 = 0.431
  排名 5: relevance=0 → 0 / log₂(5+1) = 0 / 2.585 = 0.000

  DCG@5 = 3.000 + 0.000 + 1.500 + 0.431 + 0.000 = 4.931

  对比:
    CG@5 = 7.000  (不关心顺序)
    DCG@5 = 4.931 (考虑了顺序, 排名靠后的分数被打折)

  如果把排名 3 的 "高度相关(3分)" 换到排名 1:
    排名 1: 3/1.000 = 3.000
    排名 2: 0/1.585 = 0.000
    排名 3: 3/2.000 = 1.500  ← 原来在排名1的3分被换到排名3
    → 同一组相关性分数, 顺序不同, DCG 完全不变!

  这看起来还是有问题——因为我们想要 DCG 反映"理想排序"和"实际排序"的差距。
  这就是为什么还需要 IDCG。
```

##### 4.7.2.3 IDCG（Ideal DCG）—— 理想折损累积收益

IDCG 是"如果系统完美排序"时的 DCG——也就是把所有相关文档按相关性分数从高到低排列，计算此时的 DCG。

```
  同一条查询, 相关性分数为 [3, 0, 3, 1, 0]

  理想排序 (按相关性降序):
    排名 1: relevance=3 → 3/1.000 = 3.000  (最好的放第1)
    排名 2: relevance=3 → 3/1.585 = 1.893  (次好的放第2)
    排名 3: relevance=1 → 1/2.000 = 0.500
    排名 4: relevance=0 → 0/2.322 = 0.000
    排名 5: relevance=0 → 0/2.585 = 0.000

  IDCG@5 = 3.000 + 1.893 + 0.500 + 0.000 + 0.000 = 5.393

  这就是"这条查询能达到的最高 DCG 值"——完美排序下的得分。
```

##### 4.7.2.4 NDCG —— 归一化折损累积收益

```
  NDCG@K = DCG@K / IDCG@K

  同一例子:
    实际 DCG@5 = 4.931 (系统的实际排序)
    理想 IDCG@5 = 5.393 (完美排序的得分)

    NDCG@5 = 4.931 / 5.393 = 0.914

  解读:
    NDCG = 1.000 → 系统的排序 = 完美排序! 
    NDCG = 0.914 → 系统排序是完美排序的 91.4%
    NDCG = 0.500 → 系统只达到了理想的一半
    NDCG = 0.000 → 系统完全乱排

  为什么 NDCG 比 DCG 更有意义:
    · 不同查询的 DCG 不可比 (一条查询有10个高度相关, 另一条只有2个)
    · NDCG 归一化到 [0, 1]，不同查询之间可比
    · 可以对所有查询的 NDCG 取平均 → 得到系统的整体排名质量
```

##### 4.7.2.5 NDCG 完整计算示例 —— 两条查询的对比

```
═══════════════════════════════════════════════════════════════════════
        NDCG 完整计算 (两条查询, K=5)
═══════════════════════════════════════════════════════════════════════

  Query A: "请假有哪些类型？" (应该有 4 个相关文档)

  系统实际返回 (相关性标注):
    排名 1: "事假需提前1个工作日..."    relevance = 3 (高度相关)
    排名 2: "公司实行弹性工作制..."     relevance = 0 (无关)
    排名 3: "病假应在8:30前通知..."    relevance = 3 (高度相关)
    排名 4: "五险一金缴纳比例..."       relevance = 0 (无关)
    排名 5: "年假按工龄计算..."         relevance = 2 (相关)

  DCG@5:
    = 3/log₂(2) + 0/log₂(3) + 3/log₂(4) + 0/log₂(5) + 2/log₂(6)
    = 3/1.000 + 0/1.585 + 3/2.000 + 0/2.322 + 2/2.585
    = 3.000 + 0.000 + 1.500 + 0.000 + 0.774
    = 5.274

  理想排序 (按相关度降序): [3, 3, 2, 0, 0]
  IDCG@5:
    = 3/1.000 + 3/1.585 + 2/2.000 + 0/2.322 + 0/2.585
    = 3.000 + 1.893 + 1.000 + 0.000 + 0.000
    = 5.893

  NDCG@5 (Query A) = 5.274 / 5.893 = 0.895

  ═══════════════════════════════════════════════════════════

  Query B: "公司提供哪些福利？" (只有 1 个相关文档)

  系统实际返回:
    排名 1: "五险一金缴纳比例..."   relevance = 3 (高度相关)
    排名 2: "住宿标准500元/天..."   relevance = 0
    排名 3: "餐补每日80元..."       relevance = 0
    排名 4: "报销流程..."           relevance = 0
    排名 5: "弹性工作制..."         relevance = 0

  DCG@5 = 3/1.000 + 0 + 0 + 0 + 0 = 3.000
  IDCG@5 = 3/1.000 + 0 + 0 + 0 + 0 = 3.000
  NDCG@5 (Query B) = 3.000 / 3.000 = 1.000  ← 完美!

  ═══════════════════════════════════════════════════════════

  两条查询的 NDCG 对比:
    Query A: NDCG = 0.895 (排序质量很好，但有一些可以改进的地方)
    Query B: NDCG = 1.000 (完美排序)
    
    平均 NDCG = (0.895 + 1.000) / 2 = 0.948

  注意: Query B 只有 1 个相关文档，NDCG=1.0 很容易达成。
        Query A 有多个相关文档，NDCG=0.895 需要更多排序能力。
        
        所以看 NDCG 时, 需要同时看"平均相关文档数"——
        如果大部分查询只有 1 个相关文档, NDCG 高是正常的。
═══════════════════════════════════════════════════════════════════════
```

---

#### 4.7.3 MRR vs NDCG —— 什么时候用哪个

```
═══════════════════════════════════════════════════════════════════════
        MRR vs NDCG 对比
═══════════════════════════════════════════════════════════════════════

  维度           │  MRR                      │  NDCG
  ───────────────┼───────────────────────────┼─────────────────────────────
  关注什么        │  第一个正确答案的排名       │  所有相关文档的排名 + 分数
  ───────────────┼───────────────────────────┼─────────────────────────────
  相关性分数      │  只看"相关/不相关" (二值)  │  需要"相关性等级" (多级)
                 │                          │  如: 3=高度, 2=相关, 1=弱, 0=无关
  ───────────────┼───────────────────────────┼─────────────────────────────
  一个查询多个     │  不关心 (只看第一个)       │  全关心! 排序靠前的权重更大
  相关文档时       │                          │
  ───────────────┼───────────────────────────┼─────────────────────────────
  计算复杂度      │  简单 (只需标记第一个)      │  较复杂 (需要相关性等级标注)
  ───────────────┼───────────────────────────┼─────────────────────────────
  什么时候用      │  用户只想要"一个答案"       │  用户想要"全面了解"
                 │  场景: 查政策、问规定       │  场景: 文献综述、市场调研
                 │  搜索结果页、QA系统         │  推荐系统、内容聚合
  ───────────────┼───────────────────────────┼─────────────────────────────
  典型值         │  MRR > 0.7 较好           │  NDCG > 0.8 较好
                 │  MRR > 0.5 及格           │  NDCG > 0.6 及格
  ───────────────┴───────────────────────────┴─────────────────────────────

  同一个查询, 两个指标的视角不同:

  检索返回 Top-5 (标注: ✓=相关, ✓✓✓=高度相关):
    排名 1: "年假按工龄计算..."   ✓✓✓ 高度相关
    排名 2: "事假需提前1个..."    ✓✓✓ 高度相关
    排名 3: "五险一金缴纳..."     ✗ 无关
    排名 4: "病假应在8:30前..."   ✓ 相关
    排名 5: "弹性工作制..."       ✗ 无关

  MRR: 第一个正确答案在排名 1 → RR = 1.000
       MRR 非常满意: "第一个就是正确答案!"

  NDCG: 实际 DCG = 3/1 + 3/1.585 + 0/2.0 + 1/2.322 + 0/2.585 = 5.323
        理想 DCG = 3/1 + 3/1.585 + 1/2.0 + 0/2.322 + 0/2.585 = 5.393
        NDCG = 5.323/5.393 = 0.987
        
        NDCG 也很高, 但比 MRR 多了一层理解:
        "排名 4 还有一个相关文档, 如果能排到前面就更好了"

═══════════════════════════════════════════════════════════════════════

  总结: 
    · 如果用户只需要一个答案 → 用 MRR (关注第一个对不对)
    · 如果用户需要全面信息 → 用 NDCG (关注所有相关信息是否排到前面)
    · 如果是 RAG 系统 → 两个都需要!
      MRR 告诉你 "LLM 能不能一眼看到正确答案"
      NDCG 告诉你 "所有相关信息是否都排在了 LLM 会看的地方"
═══════════════════════════════════════════════════════════════════════
```


---

### 4.8 LlamaIndex 评估实战 —— 用代码计算每一个指标

前面讲了每个指标的理论和公式。这一节用代码实现每个指标的计算，展示数据在每一步的变化。

#### 4.8.1 准备评估数据

```python
"""
═══════════════════════════════════════════════════════════════════════════
  检索评估指标 —— 完整代码实战 (纯 Python + Numpy)
═══════════════════════════════════════════════════════════════════════════
  演示 Recall / Precision / Hit Rate / MRR / NDCG 的计算全过程。
  每个指标都有: 公式注释 → 数据变化展示 → 手算验证 → 汇总报告
"""
import numpy as np
from typing import List, Dict, Set

# ── 模拟场景 ──────────────────────────────────────────────────
# 10 条测试查询，每条有人工标注的 "相关 Chunk" 和 "相关性等级"

# 每条查询的 "相关 Chunk 集合" (用于 Recall / Precision / Hit Rate)
GROUND_TRUTH = {
    0: {1, 5},               # Q0: Chunk 1 和 5 是相关的
    1: {3, 7, 12},           # Q1: 3 个相关
    2: {0, 8},               # Q2: 2 个相关
    3: {15},                 # Q3: 1 个相关
    4: {2, 6, 9, 14},       # Q4: 4 个相关
    5: {11},                 # Q5: 1 个相关
    6: {4, 10, 13},          # Q6: 3 个相关
    7: {16, 17, 18, 19},    # Q7: 4 个相关
    8: {0, 1, 2, 3, 4},     # Q8: 5 个相关
    9: {7},                  # Q9: 1 个相关
}

# 每条查询的 "相关性等级" (用于 NDCG)
# 3 = 高度相关, 2 = 相关, 1 = 弱相关, 0 = 无关
RELEVANCE_GRADES = {
    0: {1: 3, 5: 2},
    1: {3: 3, 7: 2, 12: 1},
    2: {0: 3, 8: 3},
    3: {15: 3},
    4: {2: 3, 6: 3, 9: 2, 14: 1},
    5: {11: 3},
    6: {4: 2, 10: 2, 13: 3},
    7: {16: 3, 17: 3, 18: 2, 19: 1},
    8: {0: 3, 1: 3, 2: 2, 3: 2, 4: 1},
    9: {7: 3},
}

# 模拟检索系统返回的 Top-10 (按排名排列的 Chunk ID)
RETRIEVED = {
    0: [1, 8, 5, 3, 12, 0, 15, 7, 9, 2],
    1: [7, 0, 12, 3, 6, 9, 4, 8, 15, 1],
    2: [15, 3, 0, 8, 5, 1, 7, 12, 9, 4],
    3: [15, 3, 7, 0, 8, 1, 5, 12, 6, 2],
    4: [2, 6, 10, 14, 9, 0, 1, 3, 5, 8],
    5: [3, 7, 0, 8, 11, 1, 15, 5, 2, 6],
    6: [7, 0, 4, 10, 13, 3, 8, 1, 6, 15],
    7: [0, 1, 16, 17, 18, 19, 3, 5, 2, 4],
    8: [10, 8, 15, 0, 1, 2, 3, 4, 5, 6],
    9: [5, 8, 3, 1, 15, 7, 0, 12, 2, 4],
}
```

#### 4.8.2 Recall@K 完整计算

```python
print("=" * 70)
print("  指标 1: Recall@K —— 所有相关中有多少被找回了")
print("=" * 70)

def compute_recall(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Recall@K = |Top-K中相关的| / |所有相关的|

    数据变化 (以 Q0, K=5 为例):
      输入: retrieved[:5] = [1, 8, 5, 3, 12]
            relevant = {1, 5}
      处理: Top-K集合 = {1, 8, 5, 3, 12}
            交集 = {1, 5}     ← 2个相关
            全集 = {1, 5}     ← 总共2个相关
      返回: 2/2 = 1.000  ← 全部找回了!
    """
    top_k_set = set(retrieved[:k])
    return len(top_k_set & relevant) / len(relevant) if relevant else 0.0

# 逐条计算并展示
for k in [5, 10]:
    print(f"\n  ── Recall@{k} ──")
    all_recalls = []
    for qid in range(10):
        r = compute_recall(RETRIEVED[qid], GROUND_TRUTH[qid], k)
        all_recalls.append(r)
        top_set = set(RETRIEVED[qid][:k]) & GROUND_TRUTH[qid]
        total = len(GROUND_TRUTH[qid])
        print(f"    Q{qid}: {len(top_set)}/{total} → {r:.2f}  "
              f"(找到的: {top_set})")
    print(f"    平均 Recall@{k} = {np.mean(all_recalls):.4f}")
```

**运行输出示例：**
```
  ── Recall@5 ──
    Q0: 2/2 → 1.00  (找到的: {1, 5})
    Q1: 3/3 → 1.00  (找到的: {3, 7, 12})
    Q2: 2/2 → 1.00  (找到的: {0, 8})
    Q3: 1/1 → 1.00  (找到的: {15})
    Q4: 4/4 → 1.00  (找到的: {2, 6, 9, 14})
    Q5: 0/1 → 0.00  (找到的: set() ← 相关Chunk 11排在Top-5之后!
    Q6: 3/3 → 1.00  (找到的: {4, 10, 13})
    Q7: 2/4 → 0.50  (找到的: {16, 17} ← 漏了18和19!
    Q8: 0/5 → 0.00  (找到的: set() ← 5个相关都排在5之后!
    Q9: 0/1 → 0.00  (找到的: set() ← 相关Chunk 7排在第6位
    平均 Recall@5 = 0.6500

  ── Recall@10 ──
    Q0-Q9: 全部 = 1.00  (Top-10 足够大, 覆盖了所有相关)
    平均 Recall@10 = 1.0000
```

#### 4.8.3 Precision@K 完整计算

```python
print("\n" + "=" * 70)
print("  指标 2: Precision@K —— 返回的结果中有多少相关")
print("=" * 70)

def compute_precision(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Precision@K = |Top-K中相关的| / K

    数据变化 (以 Q0, K=5 为例):
      输入: retrieved[:5] = [1, 8, 5, 3, 12]
            relevant = {1, 5}
      处理: 交集 = {1, 5} → 2个相关
            分母 = 5 (固定的K)
      返回: 2/5 = 0.400  ← 5个结果中2个相关, 3个是噪声
    """
    return len(set(retrieved[:k]) & relevant) / k

for k in [5, 10]:
    print(f"\n  ── Precision@{k} ──")
    all_prec = []
    for qid in range(10):
        p = compute_precision(RETRIEVED[qid], GROUND_TRUTH[qid], k)
        all_prec.append(p)
        hits = set(RETRIEVED[qid][:k]) & GROUND_TRUTH[qid]
        noise = set(RETRIEVED[qid][:k]) - GROUND_TRUTH[qid]
        print(f"    Q{qid}: {len(hits)}/{k} → {p:.2f}  "
              f"(相关: {hits if hits else '无'}, 噪声: {len(noise)}个)")
    print(f"    平均 Precision@{k} = {np.mean(all_prec):.4f}")
```

**对比 Recall 和 Precision 的关键差异：**
```
  Q0: Recall@5 = 2/2 = 1.00  (所有相关都找回了) ✓
       Precision@5 = 2/5 = 0.40 (但返回的一半以上是噪声) ⚠

  Q8: Recall@5 = 0/5 = 0.00  (一个都没找回) ✗
       Precision@5 = 0/5 = 0.00 (自然也是0)

  → Recall 告诉你"找得全不全"，Precision 告诉你"找得准不准"
  → 两者必须同时看!
```

#### 4.8.4 Hit Rate@K 完整计算

```python
print("\n" + "=" * 70)
print("  指标 3: Hit Rate@K —— 至少找到1个相关了吗")
print("=" * 70)

def compute_hit(retrieved: List[int], relevant: Set[int], k: int) -> bool:
    """Hit Rate 单条: 返回 True/False"""
    for item in retrieved[:k]:
        if item in relevant:
            return True
    return False

for k in [1, 5, 10]:
    print(f"\n  ── Hit Rate@{k} ──")
    hits = [compute_hit(RETRIEVED[q], GROUND_TRUTH[q], k) for q in range(10)]
    hr = sum(hits) / len(hits)
    for qid in range(10):
        status = "命中了" if hits[qid] else "没找到"
        first_rank = None
        for i, item in enumerate(RETRIEVED[qid][:k]):
            if item in GROUND_TRUTH[qid]:
                first_rank = i + 1
                break
        rank_info = f"第1个在rank={first_rank}" if first_rank else "Top-K中无"
        print(f"    Q{qid}: {status} ({rank_info})")
    print(f"    Hit Rate@{k} = {hr:.2%} → {sum(hits)}/10 条查询找到了")
```

**对比不同 K 值：**
```
  Hit Rate@1  → 如果 Q4 第1位就是相关 → 命中; 如果第2位才出现 → 不命中
  Hit Rate@5  → 更宽松, 大部分查询都能命中
  Hit Rate@10 → 几乎全部命中
  
  K=1 vs K=5 的差距告诉你系统"一眼能不能给对"。
```

#### 4.8.5 MRR@K 完整计算

```python
print("\n" + "=" * 70)
print("  指标 4: MRR@K —— 第一个正确答案的排名")
print("=" * 70)

def compute_rr(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    RR = 1 / rank_of_first_relevant  (rank 从 1 开始)
    找不到 → RR = 0

    数据变化 (Q0, K=10):
      retrieved = [1, 8, 5, 3, 12, 0, 15, 7, 9, 2]
      遍历: rank1=1→相关! → RR = 1/1 = 1.000

    数据变化 (Q5, K=10):
      retrieved = [3, 7, 0, 8, 11, 1, 15, 5, 2, 6]
      遍历: rank1=3→无关, rank2=7→无关, ..., rank5=11→相关!
      → RR = 1/5 = 0.200
    """
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            return 1.0 / (i + 1)   # i从0开始, rank=i+1
    return 0.0

for k in [5, 10]:
    print(f"\n  ── MRR@{k} ──")
    rrs = [compute_rr(RETRIEVED[q], GROUND_TRUTH[q], k) for q in range(10)]
    print(f"    每条 RR: ", end="")
    for qid, rr in enumerate(rrs):
        # 找具体排名
        rank = None
        for i, item in enumerate(RETRIEVED[qid][:k]):
            if item in GROUND_TRUTH[qid]:
                rank = i + 1
                break
        rank_str = f"rank={rank}" if rank else "未命中"
        print(f"Q{qid}={rr:.3f}({rank_str})  ", end="" if qid % 3 != 2 else "\n     ")
    print(f"\n    MRR@{k} = {np.mean(rrs):.4f}")
    if np.mean(rrs) > 0:
        print(f"    平均排名 ≈ 第 {1/np.mean(rrs):.1f} 位")
```

#### 4.8.6 NDCG@K 完整计算

```python
print("\n" + "=" * 70)
print("  指标 5: NDCG@K —— 整体排序质量 (考虑所有相关文档)")
print("=" * 70)

def compute_dcg(retrieved: List[int], grades: Dict[int, int], k: int) -> float:
    """
    DCG@K = Σ rel_i / log₂(rank_i + 1)
            rank从1开始, 所以第i个位置 (i=0-based) 的分母 = log₂(i+2)

    数据变化 (Q0, K=5):
      rank1: Chunk1, rel=3 → 3/log₂(2) = 3/1.000 = 3.000
      rank2: Chunk8, rel=0 → 0/log₂(3) = 0/1.585 = 0.000
      rank3: Chunk5, rel=2 → 2/log₂(4) = 2/2.000 = 1.000
      rank4: Chunk3, rel=0 → 0
      rank5: Chunk12, rel=0 → 0
      DCG@5 = 3.000 + 0.000 + 1.000 + 0.000 + 0.000 = 4.000
    """
    dcg = 0.0
    for i, item_id in enumerate(retrieved[:k]):
        rel = grades.get(item_id, 0)
        dcg += rel / np.log2(i + 2)    # i+2 因为 rank=i+1
    return dcg

def compute_ndcg(retrieved: List[int], grades: Dict[int, int], k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    IDCG = 理想排序下的 DCG (所有相关按 rel 降序排到最前面)
    """
    dcg = compute_dcg(retrieved, grades, k)
    ideal = sorted(grades.values(), reverse=True)
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(min(k, len(ideal))))
    return dcg / idcg if idcg > 0 else 0.0

for k in [5, 10]:
    print(f"\n  ── NDCG@{k} ──")
    ndcgs = []
    for qid in range(10):
        ndcg_val = compute_ndcg(RETRIEVED[qid], RELEVANCE_GRADES[qid], k)
        ndcgs.append(ndcg_val)
        dcg_val = compute_dcg(RETRIEVED[qid], RELEVANCE_GRADES[qid], k)
        ideal_vals = sorted(RELEVANCE_GRADES[qid].values(), reverse=True)[:k]
        idcg_val = sum(ideal_vals[i] / np.log2(i + 2) for i in range(len(ideal_vals)))
        print(f"    Q{qid}: DCG={dcg_val:.3f} / IDCG={idcg_val:.3f} → NDCG={ndcg_val:.3f}")
    print(f"    平均 NDCG@{k} = {np.mean(ndcgs):.4f}")

# ── 手算详解 Q0 ──────────────────────────────────────────
print(f"\n  [手算详解] Q0 NDCG@5:")
q = 0
ret = RETRIEVED[q][:5]
grades = RELEVANCE_GRADES[q]
print(f"    检索 Top-5: {ret}")
print(f"    相关性: Chunk1=3(高), Chunk5=2(中), 其余=0")
print(f"")
print(f"    DCG 步进计算:")
dcg_sum = 0.0
for i, item_id in enumerate(ret):
    rel = grades.get(item_id, 0)
    discount = 1.0 / np.log2(i + 2)
    contrib = rel * discount
    dcg_sum += contrib
    print(f"      rank{i+1}: Chunk{item_id}(rel={rel}) × {discount:.4f} = {contrib:.3f}  → 累计DCG={dcg_sum:.3f}")
print(f"    DCG@5 = {dcg_sum:.3f}")
print(f"")
print(f"    IDCG (理想: 相关分数 [3,2,0,0,0]):")
ideal = [3, 2, 0, 0, 0]
idcg_sum = 0.0
for i, rel in enumerate(ideal):
    discount = 1.0 / np.log2(i + 2)
    contrib = rel * discount
    idcg_sum += contrib
    print(f"      rank{i+1}: rel={rel} × {discount:.4f} = {contrib:.3f}  → 累计IDCG={idcg_sum:.3f}")
print(f"    IDCG@5 = {idcg_sum:.3f}")
print(f"    NDCG@5 = {dcg_sum:.3f}/{idcg_sum:.3f} = {dcg_sum/idcg_sum:.3f}")

# ═══════════════════════════════════════════════════════════════
# 汇总报告
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  六项评估指标汇总报告")
print("=" * 70)

results = {}
for k in [5, 10]:
    results[f"Recall@{k}"] = np.mean([compute_recall(RETRIEVED[q], GROUND_TRUTH[q], k) for q in range(10)])
    results[f"Precision@{k}"] = np.mean([compute_precision(RETRIEVED[q], GROUND_TRUTH[q], k) for q in range(10)])
    results[f"Hit Rate@{k}"] = np.mean([compute_hit(RETRIEVED[q], GROUND_TRUTH[q], k) for q in range(10)])
    results[f"MRR@{k}"] = np.mean([compute_rr(RETRIEVED[q], GROUND_TRUTH[q], k) for q in range(10)])
    results[f"NDCG@{k}"] = np.mean([compute_ndcg(RETRIEVED[q], RELEVANCE_GRADES[q], k) for q in range(10)])

print(f"\n  {'指标':<20} {'@5':<12} {'@10':<12} {'解读'}")
print(f"  {'-'*55}")
for k in [5, 10]:
    print(f"  {'Recall@'+str(k):<20} {results[f'Recall@{k}']:<12.4f} {'相关中有多少被找回' if k==5 else ''}")
    print(f"  {'Precision@'+str(k):<20} {results[f'Precision@{k}']:<12.4f} {'返回中有多少相关' if k==5 else ''}")
    print(f"  {'Hit Rate@'+str(k):<20} {results[f'Hit Rate@{k}']:<12.4f} {'至少找到1个的比例' if k==5 else ''}")
    print(f"  {'MRR@'+str(k):<20} {results[f'MRR@{k}']:<12.4f} {'第一个正确答案的排名' if k==5 else ''}")
    print(f"  {'NDCG@'+str(k):<20} {results[f'NDCG@{k}']:<12.4f} {'整体排序质量' if k==5 else ''}")
```


---

### 4.9 LlamaIndex 内置评估器 —— 一行代码完成评估

4.8 节手写的指标是为了让你理解计算逻辑。生产环境中，LlamaIndex 提供了内置的评估器，直接调用即可。

#### 4.9.1 LlamaIndex 的评估器全景

```
═══════════════════════════════════════════════════════════════════════
        LlamaIndex 内置评估器分类
═══════════════════════════════════════════════════════════════════════

  检索评估:
  ┌─────────────────────────────────────────────────────────────┐
  │  RetrieverEvaluator                                         │
  │    · evaluate() → 对单条 Query 的检索结果打分               │
  │    · 内置指标: MRR, Hit Rate                                │
  │    · 返回: RetrievalEvalResult (含 score, passing, metadata)│
  │                                                             │
  │  MultiModalRetrieverEvaluator                                │
  │    · 同上，针对多模态(文本+图片)检索                         │
  └─────────────────────────────────────────────────────────────┘

  生成评估 (LLM-as-Judge):
  ┌─────────────────────────────────────────────────────────────┐
  │  FaithfulnessEvaluator                                       │
  │    · 评估: 生成的 Answer 是否忠实于 Context?                  │
  │    · 原理: 逐句拆分 Answer → 在 Context 中找证据             │
  │    · 返回: EvaluationResult (passing=True/False, score)      │
  │                                                             │
  │  RelevancyEvaluator                                          │
  │    · 评估: Context 是否与 Query 相关?                         │
  │    · 原理: LLM 读取 Query+Context → 判断相关性                │
  │    · 返回: EvaluationResult                                  │
  │                                                             │
  │  CorrectnessEvaluator                                        │
  │    · 评估: Answer 与 Ground Truth 是否一致?                    │
  │    · 原理: LLM 对比 Answer vs Reference → 打分 1-5          │
  │    · 返回: EvaluationResult                                  │
  │                                                             │
  │  PairwiseComparisonEvaluator                                 │
  │    · 评估: 两个 Answer 哪个更好? (A/B 对比)                   │
  └─────────────────────────────────────────────────────────────┘

  批处理:
  ┌─────────────────────────────────────────────────────────────┐
  │  BatchEvalRunner                                             │
  │    · 批量运行多条 Query → 汇总 → 生成报告                    │
  └─────────────────────────────────────────────────────────────┘
═══════════════════════════════════════════════════════════════════════
```

#### 4.9.2 RetrieverEvaluator —— 检索质量评估（MRR + Hit Rate）

```python
"""
═══════════════════════════════════════════════════════════════════════════
  LlamaIndex 内置: RetrieverEvaluator
═══════════════════════════════════════════════════════════════════════════

  RetrieverEvaluator 内部做了什么:
    1. 接收一个 Retriever 对象 + 一组测试 Query
    2. 对每条 Query: retriever.retrieve(query) → 得到检索结果
    3. 对每条结果: 检查 Ground Truth 是否在 Top-K 结果中
    4. 计算 MRR 和 Hit Rate
    5. 返回 RetrievalEvalResult 列表

  注意: RetrieverEvaluator 内置只算了 MRR 和 Hit Rate!
        Recall / Precision 需要自己写 (如 4.8 节所示)，
        或用 BatchEvalRunner + 自定义评估函数。
"""
from llama_index.core.evaluation import RetrieverEvaluator

# ── Step 1: 准备你的 Retriever ─────────────────────────────
retriever = index.as_retriever(similarity_top_k=5)

# ── Step 2: 创建 RetrieverEvaluator ───────────────────────
evaluator = RetrieverEvaluator.from_metric_names(
    metric_names=["mrr", "hit_rate"],   # 要计算的指标
    retriever=retriever,
)

# ── Step 3: 逐条评估 ──────────────────────────────────────
# 对每条 Query 执行检索 → 评估
eval_results = evaluator.evaluate(
    queries=["事假需要提前多久申请？", "加班费怎么算？", "五险一金有哪些？"],
    expected_ids=[        # ← 每条 Query 对应的"正确答案 Node ID"
        ["node_03"],      # Query 0 的正确答案是 node_03
        ["node_12"],      # Query 1 的正确答案是 node_12
        ["node_07"],      # Query 2 的正确答案是 node_07
    ],
)

# ── Step 4: 查看结果 ──────────────────────────────────────
for i, result in enumerate(eval_results):
    print(f"\n  Query {i}: {result.query}")
    print(f"    MRR:       {result.metric_vals_dict.get('mrr', 'N/A'):.4f}")
    print(f"    Hit Rate:  {result.metric_vals_dict.get('hit_rate', 'N/A')}")
    print(f"    检索到的 Node 数: {len(result.retrieved_nodes)}")

# ── 汇总 ──────────────────────────────────────────────────
avg_mrr = np.mean([r.metric_vals_dict.get("mrr", 0) for r in eval_results])
avg_hit = np.mean([r.metric_vals_dict.get("hit_rate", 0) for r in eval_results])
print(f"\n  平均 MRR: {avg_mrr:.4f}")
print(f"  平均 Hit Rate: {avg_hit:.4f}")
```

**RetrievalEvalResult 返回的数据结构：**

```python
# result 的内部字段:
class RetrievalEvalResult:
    query: str                          # "事假需要提前多久申请？"
    expected_ids: List[str]             # ["node_03"] — 人工标注的正确答案
    retrieved_ids: List[str]            # ["node_03", "node_08", "node_01", ...] — 系统返回的
    retrieved_nodes: List[NodeWithScore] # 完整的 Node 对象列表
    metric_vals_dict: Dict[str, float]  # {"mrr": 1.0, "hit_rate": 1.0}
    passing: bool                       # 是否通过 (Hit Rate > 0 即通过)
```

---

#### 4.9.3 FaithfulnessEvaluator —— 生成质量评估（回答是否忠实于上下文）

```python
"""
═══════════════════════════════════════════════════════════════════════════
  LlamaIndex 内置: FaithfulnessEvaluator
═══════════════════════════════════════════════════════════════════════════

  FaithfulnessEvaluator 的工作方式:
    1. 将 LLM 生成的 Answer 逐句拆分
    2. 对每句话: 在 Context (检索到的 Chunk) 中搜索证据
    3. 如果有证据 → 这句话是 "faithful" (忠实的)
    4. 如果找不到证据 → 这句话可能是 LLM 编造的!
    5. faithfulness = 有证据的句数 / 总句数

  关键: 判断的是 "Answer 是否基于 Context", 不是 "Answer 是否正确"!
"""
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.llms.openai import OpenAI

# ── 创建评估器 (用 GPT-4o 做裁判) ─────────────────────────
faith_evaluator = FaithfulnessEvaluator(
    llm=OpenAI(model="gpt-4o", temperature=0),
)

# ── 模拟: Query + Context + 生成的 Answer ────────────────
query = "事假需要提前多久申请？"
contexts = [
    "事假需提前1个工作日向部门主管申请，经审批后交HR备案。",
    "病假应在当日8:30前通知部门主管。",
]
# LLM 生成的回答 (模拟)
response_text = "事假需提前1个工作日向部门主管申请，期间不发放工资。"

# ── 评估 ──────────────────────────────────────────────────
faith_result = faith_evaluator.evaluate_response(
    query=query,
    response=response_text,      # LLM 实际生成的
    contexts=contexts,           # 检索到的 Chunk
)

print(f"  Faithfulness: {faith_result.passing}")  # True or False
print(f"  Score: {faith_result.score}")           # 0.0 ~ 1.0
print(f"  Feedback: {faith_result.feedback}")     # LLM 的解释
# 输出示例:
#   Faithfulness: True
#   Score: 0.67
#   Feedback: "第一句'提前1个工作日'在上下文中找到了证据。
#              第二句'期间不发放工资'在上下文中没有找到支持."
```

**Faithfulness 分数的含义：**

| Score | 含义 | 处理建议 |
|:---:|------|------|
| 1.0 | 所有陈述都在 Context 中有证据 | 完美 |
| 0.6-0.9 | 大部分有证据，少量无法验证 | 检查 Context 是否包含足够信息 |
| 0.3-0.6 | 接近一半是编造的 | 检查 Prompt 是否约束不够 / LLM temperature 过高 |
| <0.3 | 大量编造 | 严重幻觉问题，检查检索质量和 Prompt |

---

#### 4.9.4 RelevancyEvaluator —— 检索相关性评估

```python
"""
═══════════════════════════════════════════════════════════════════════════
  LlamaIndex 内置: RelevancyEvaluator
═══════════════════════════════════════════════════════════════════════════

  RelevancyEvaluator 的工作方式:
    1. LLM 读取 Query + Context (检索到的 Chunk)
    2. LLM 判断: "这些 Chunk 能否帮助回答这个问题?"
    3. 返回: passing (True/False) + score

  与 FaithfulnessEvaluator 的区别:
    Faithfulness → "Answer 是否忠于 Context?"  (Context → Answer)
    Relevancy    → "Context 是否与 Query 相关?" (Query → Context)
    两者方向相反!
"""
from llama_index.core.evaluation import RelevancyEvaluator

relevancy_evaluator = RelevancyEvaluator(
    llm=OpenAI(model="gpt-4o", temperature=0),
)

relevancy_result = relevancy_evaluator.evaluate_response(
    query=query,
    response=response_text,
    contexts=contexts,
)

print(f"  Relevancy: {relevancy_result.passing}")
print(f"  Score: {relevancy_result.score}")
# 输出示例: Relevancy: True, Score: 0.85
# → 检索到的 Chunk 与 Query 高度相关
```

---

#### 4.9.5 BatchEvalRunner —— 批量评估 + 汇总报告

```python
"""
═══════════════════════════════════════════════════════════════════════════
  LlamaIndex 内置: BatchEvalRunner
═══════════════════════════════════════════════════════════════════════════

  作用: 将多条 Query 批量送入评估器，自动汇总分数。
"""
from llama_index.core.evaluation import BatchEvalRunner

# ── 准备测试数据 ─────────────────────────────────────────
eval_queries = [
    "事假需要提前多久申请？",
    "病假怎么规定的？",
    "加班费怎么计算？",
    "五险一金包括哪些？",
    "报销流程是什么？",
]

# ── 创建 Runner (同时跑多个评估器) ────────────────────────
runner = BatchEvalRunner(
    evaluators={
        "faithfulness": faith_evaluator,
        "relevancy": relevancy_evaluator,
        "retrieval": evaluator,  # RetrieverEvaluator
    },
    workers=1,  # 并发数 (避免 API 限流)
)

# ── 批量运行 (需要 QueryEngine 对每条查询生成回答) ────────
# 注意: BatchEvalRunner 需要你传入 QueryEngine 或 Response 对象
# 简化示例:
# eval_results = runner.evaluate_queries(
#     queries=eval_queries,
#     query_engine=query_engine,
# )
#
# for evaluator_name, results in eval_results.items():
#     avg_score = np.mean([r.score for r in results])
#     print(f"  {evaluator_name}: avg_score={avg_score:.3f}")
```

---

#### 4.9.6 手写 vs LlamaIndex 内置对比

| 方面 | 4.8 节手写 | LlamaIndex 内置评估器 |
|------|----------|---------------------|
| **指标覆盖** | Recall/Precision/HitRate/MRR/NDCG 全部 | RetrieverEvaluator只支持MRR+HitRate；Faithfulness/Relevancy是LLM裁判 |
| **标注需求** | 你需要准备全部标注数据 | RetrieverEvaluator 需要 expected_ids；Faithfulness/Relevancy 不需要标注 |
| **计算方式** | 数学公式直接算 | RetrieverEvaluator 是数学公式；Faithfulness/Relevancy 是 LLM 打分 |
| **灵活性** | 极高——你想怎么算就怎么算 | 受限于评估器支持的指标 |
| **成本** | 零（纯数学计算） | Faithfulness/Relevancy 每次调用 LLM → 有 API 费用 |
| **适合阶段** | 深度测评（上线前） | 快速验证（开发中） |

**建议的使用顺序：**
1. 开发阶段 → 用 `RetrieverEvaluator` 快速看 MRR/Hit Rate（零 LLM 成本）
2. 调优阶段 → 用 `FaithfulnessEvaluator` + `RelevancyEvaluator`（LLM 裁判，少量抽样）
3. 上线前 → 用手写方法（4.8 节）全量计算 Recall/Precision/NDCG


---

### 4.10 生成质量三指标 —— 忠实度、答案相关性、事实正确性

检索评估回答的是"检索系统有没有找到对的信息"。生成评估回答的是另外一个问题：**LLM 基于找到的信息，有没有生成好的回答？**

这三个指标从不同角度衡量"好"：

```
═══════════════════════════════════════════════════════════════════════
        三个指标分别衡量什么
═══════════════════════════════════════════════════════════════════════

  Faithfulness (忠实度):
    Question: "LLM 的回答是否忠实地反映了 Context？"
    检查对象: Answer ← Context
    问的是:    "LLM 有没有编造 Context 里不存在的东西？"

  Answer Relevance (答案相关性):
    Question: "LLM 的回答是否紧扣用户问题？"
    检查对象: Answer ← Question
    问的是:    "LLM 有没有答非所问？"

  Factual Correctness (事实正确性):
    Question: "LLM 的回答与客观事实是否一致？"
    检查对象: Answer ← Ground Truth (参考答案)
    问的是:    "LLM 说的到底对不对？"
```

#### 4.10.1 用一个例子区分三个指标

```
═══════════════════════════════════════════════════════════════════════
        同一个例子，三个指标给出完全不同的分数
═══════════════════════════════════════════════════════════════════════

  场景: 你有一个知识库，里面的文档是这样写的:

    Context (知识库中的原文):
      "巴黎是法国的首都。法国位于西欧，人口约6700万。"

  用户问: "法国的首都是哪里？"

  现在有两个 LLM 回答，我们来看三个指标分别给什么分数:

  ═══════════════════════════════════════════════════════════

  回答 A: "法国的首都是米兰。"

    Faithfulness (忠实度): ★★★★★ 高分!
      检查: Context 里有"巴黎是法国的首都" → LLM 说"米兰是法国首都" → 
      但这在 Context 中找不到证据 → 实际上忠实度应该是 低! 
      等一等——如果 LLM 说的"米兰"在每个字上都遵循了 Context 的模式
      (首都 + 城市名)，但从 Context 中**找不到"米兰"这个词** →
      所以 Faithfulness = 0/1 = 0。LLM 编造了"米兰"。

      等等, 再想: 如果 Context 中恰好有 "米兰是意大利的时尚之都" 这段文字呢?
      → Faithfulness 检查的是 "米兰" 这个词是否在 Context 中出现过。
      它在另一段 Context 中出现了 → Faithfulness 可能被判为高!
      → 但"米兰是法国首都"这个事实是错的!
      → 这就是为什么 Faithfulness 高 ≠ 答案正确!

    Answer Relevance (答案相关性): ★★★★★ 高分!
      检查: 问题问"首都" → 回答说了"首都" → 高度相关!
      即使答案错了(米兰不是法国首都)，它也确实在回答"首都"这个问题。

    Factual Correctness (事实正确性): ✗ 低分
      检查: Ground Truth = "巴黎是法国首都"
            回答 = "米兰是法国首都"
            → 错误! 0分。

  ═══════════════════════════════════════════════════════════════

  回答 B: "巴黎是法国首都，法国还有埃菲尔铁塔、卢浮宫。卢浮宫收藏了蒙娜丽莎。"

    Faithfulness (忠实度):
      检查每句话:
        "巴黎是法国首都" → Context 中有 ✓
        "法国有埃菲尔铁塔" → Context 中没有 ✗ (编造! Context没提铁塔)
        "卢浮宫收藏蒙娜丽莎" → Context 中没有 ✗ (编造!)
      Faithfulness = 1/3 ≈ 0.33  ← 低! 三句话中两句是编的

    Answer Relevance (答案相关性):
      检查: 问题问"首都" → 回答花了大篇幅讲"铁塔""博物馆" → 
      这些与"首都"不直接相关 → Relevance 中等偏低

    Factual Correctness (事实正确性):
      检查: "巴黎是法国首都" → 正确 ✓
            另外两句虽然与问题无关, 但客观上也是正确的事实。
            但评分的通常只看与问题直接相关的部分 → 高分

  ═══════════════════════════════════════════════════════════════

  回答 C (理想): "法国的首都是巴黎。"

    Faithfulness: ★★★★★ 高分 (Context 中能找到)
    Answer Relevance: ★★★★★ 高分 (直接回答问题)
    Factual Correctness: ★★★★★ 高分 (与事实一致)

  ═══════════════════════════════════════════════════════════════

  核心理解:

    这三个指标的关系不是"谁更好"——而是"从不同角度检查"。

    可能的情况:
    ┌──────────────────┬──────────────┬──────────────────┬──────────────────┐
    │ 场景              │ Faithfulness │ Answer Relevance │Factual Correctness│
    ├──────────────────┼──────────────┼──────────────────┼──────────────────┤
    │ 编造了不存在的信息  │ 低           │ 高 (还是回答了问题) │ 低 (编造=错误)    │
    │ 答非所问           │ 高 (照着念的) │ 低               │ 高 (念的是对的)    │
    │ 照着错的文档生成   │ 高 (忠实于错文档)│ 高              │ 低 (文档就是错的)  │
    │ 完美回答           │ 高           │ 高               │ 高               │
    └──────────────────┴──────────────┴──────────────────┴──────────────────┘

    最关键的一行: "照着错的文档生成" → Faithfulness=高, Correctness=低
    这就是 Faithfulness 的局限性 —— 它只管"LLM有没有照抄Context",
    不管"Context本身对不对"。
═══════════════════════════════════════════════════════════════════════
```

---

#### 4.10.2 Faithfulness（忠实度）—— 计算逻辑详解

```
  完整计算流程:

  Step 1: 提取主张 (Claim Extraction)
    将 LLM 的回答拆解为独立的"主张"(claims)
    
    Answer: "事假需提前1个工作日向部门主管申请，期间不发放工资。"
    拆解为:
      Claim 1: "事假需提前1个工作日向部门主管申请"
      Claim 2: "事假期间不发放工资"

  Step 2: 逐一验证 (Claim Verification)
    对每个 Claim, 在 Context 中搜索是否有证据支持:
    
    Claim 1: "提前1个工作日向部门主管申请"
      Context 中搜索 → 找到: "事假需提前1个工作日向部门主管申请" → 支持 ✓
      → verdict = 1
    
    Claim 2: "期间不发放工资"
      Context 中搜索 → 没找到 "不发放工资" 
      (Context 只说了提前1个工作日申请, 没提工资)
      → verdict = 0  ← 编造的!

  Step 3: 计算忠实度
    Faithfulness = 有证据的 Claim 数 / 总 Claim 数
                 = 1 / 2 = 0.50

    LLM 只忠实于了一半 —— 一半是照抄Context的, 另一半是自己编的。
```

---

#### 4.10.3 Answer Relevance（答案相关性）—— 计算逻辑详解

```
  完整计算流程:

  Step 1: 语义相关度计算
    将 Answer 向量化 → 与 Query 向量做余弦相似度
    
    Query 向量:  [0.23, -0.45, 0.67, ...]  (关于"事假申请")
    Answer 向量: [0.21, -0.42, 0.63, ...]
    → 余弦相似度 = 0.94  (高度相关)

  Step 2: 生成"反向问题" (Reverse Question)
    让 LLM 根据 Answer 生成一个问题: 
    "如果这个答案是回答, 什么样的问题会引出这个答案?"

    Answer: "事假需提前1个工作日向部门主管申请。期间不发放工资。"
    LLM 生成的反向问题: "事假的申请流程和工资政策是什么？"

  Step 3: 反向问题与原始问题的语义相似度
    cos(原始问题 "事假需要提前多久申请？", 反向问题 "事假的申请流程和工资政策是什么？")
    → 如果反向问题 ≈ 原始问题 → 回答紧扣原问题 → 高相关性
    → 如果反向问题 偏离 原始问题 → 回答跑题了 → 低相关性

  Step 4: 冗余度惩罚
    如果 Answer 中包含大量重复或无意义的词 → 扣分
    "好的好的，让我来回答你的问题，关于这个事假，嗯，根据规定...就是...嗯..."
    → 大量冗余 → 惩罚

  Step 5: 最终得分
    Answer Relevance = 语义相似度 - 冗余度惩罚
```

---

#### 4.10.4 Factual Correctness（事实正确性）—— 计算逻辑详解

```
  完整计算流程:

  Step 1: 提取关键事实
    对 Answer 和 Ground Truth 分别提取关键事实:
    
    Answer:       "事假需提前3个工作日申请"
    Ground Truth: "事假需提前1个工作日申请"
    
    提取出的事实: 3个工作日 vs 1个工作日

  Step 2: LLM 逐条对比
    用 LLM 判断每个事实是否与 Ground Truth 一致:
    
    "3个工作日" vs "1个工作日" → 不一致 → 0分

  Step 3: 打分 (通常 1-5 分制)
    1分: 完全错误
    3分: 部分正确
    5分: 完全正确
    
    这里: 3个工作日 vs 1个工作日 → 1-2分 (基本信息框架对, 细节错)

  与 Faithfulness 的关键区别:
    Faithfulness: Answer vs Context     (不管Context本身对不对)
    Correctness:  Answer vs Ground Truth (关心"正确答案"是什么)
```

---

#### 4.10.5 LlamaIndex 代码实战 —— 计算三个指标

```python
"""
═══════════════════════════════════════════════════════════════════════════
  LlamaIndex 内置: 生成质量三指标完整计算
═══════════════════════════════════════════════════════════════════════════

  依赖: pip install llama-index-core llama-index-llms-openai
"""
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from llama_index.llms.openai import OpenAI

# ── 创建三个评估器 (都用 GPT-4o 做裁判) ────────────────────
judge_llm = OpenAI(model="gpt-4o", temperature=0)

faith_evaluator    = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)
correctness_evaluator = CorrectnessEvaluator(llm=judge_llm)

# ═══════════════════════════════════════════════════════════════
# 场景设置: 同一个 Query，三个不同的 Answer
# ═══════════════════════════════════════════════════════════════

query = "事假申请规定是什么？"

# Ground Truth (标准答案)
reference = (
    "事假需提前1个工作日向部门主管申请，经审批后交HR备案。"
    "紧急情况可在当日8:30前电话通知部门主管，事后补办手续。"
)

# Context (知识库中检索到的内容)
context = [
    "事假需提前1个工作日向部门主管申请，经审批后交HR备案。",
    "病假应在当日8:30前通知部门主管。",
]

# ── 回答 A: 忠实于Context但事实不全 ──────────────────────
answer_a = "事假需提前1个工作日向部门主管申请，经审批后交HR备案。"

# ── 回答 B: 完全不忠实，编造的 ───────────────────────────
answer_b = "事假需提前3个工作日向部门主管和HR共同申请，"
answer_b += "期间工资按50%发放，还需要医院证明。"

# ── 回答 C: 答非所问 ─────────────────────────────────────
answer_c = "公司实行弹性工作制，核心工作时间10:00-17:00，"
answer_c += "午休时间为12:00-13:30。五险一金包括养老、医疗..."

# ═══════════════════════════════════════════════════════════════
# 分别评估
# ═══════════════════════════════════════════════════════════════

def evaluate_all(query, response_text, contexts, reference):
    """一个函数跑完三个评估"""
    print(f"  回答: {response_text[:60]}...")
    print()

    # 1. Faithfulness —— Answer 是否忠于 Context
    faith = faith_evaluator.evaluate_response(
        query=query,
        response=response_text,
        contexts=contexts,
    )
    print(f"  [Faithfulness]  passage={faith.passing}  score={faith.score}")
    if faith.feedback:
        print(f"    反馈: {faith.feedback[:120]}...")

    # 2. Answer Relevance —— Answer 是否紧扣 Question
    relevancy = relevancy_evaluator.evaluate_response(
        query=query,
        response=response_text,
    )
    print(f"  [Relevancy]     passage={relevancy.passing}  score={relevancy.score}")

    # 3. Factual Correctness —— Answer 是否与 Ground Truth 一致
    correctness = correctness_evaluator.evaluate(
        query=query,
        response=response_text,
        reference=reference,
    )
    print(f"  [Correctness]   passage={correctness.passing}  score={correctness.score}")
    print(f"  {'─'*50}")
    return faith, relevancy, correctness

print("=" * 70)
print("  生成质量评估: 三种不同回答的对比")
print("=" * 70)

print("\n  [回答 A] 忠实+相关+正确")
evaluate_all(query, answer_a, context, reference)

print("\n  [回答 B] 编造的 (低忠实度, 低正确性)")
evaluate_all(query, answer_b, context, reference)

print("\n  [回答 C] 答非所问 (低相关性)")
evaluate_all(query, answer_c, context, reference)

# ═══════════════════════════════════════════════════════════════
# 预期结果对比
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  三指标对比解读")
print("=" * 70)
print("""
  预期得分:

                     Faithfulness  Relevancy   Correctness
  回答 A (忠实+对)      高 ✓         高 ✓        高 ✓
  回答 B (编造)         低 ✗         中~高       低 ✗
  回答 C (跑题)         中~高        低 ✗        中

  关键解读:
    · 回答 B 可能 Relevance 不低 —— 因为它确实在讲"事假"
    · 回答 A 三项都高 —— 这是理想状态
    · 回答 C Faithfulness 可能不低 —— 
      它照抄了Context(弹性工作制)，所以是"忠实"的，但文不对题

  这个对比完美展示了为什么三个指标要一起看:
    · Faithfulness 高 + Correctness 低 → Context 本身是错的
    · Faithfulness 低 + Relevancy 高 → LLM 在编造，编的还在点上
    · Relevancy 低 → 检索可能有问题，返回了不相关的内容
""")
```

---

#### 4.10.6 三指标联合诊断矩阵

```
═══════════════════════════════════════════════════════════════════════
        生成质量诊断矩阵 —— 三指标组合的含义
═══════════════════════════════════════════════════════════════════════

  Faithfulness │ Relevancy │ Correctness │ 诊断
  ─────────────┼───────────┼────────────┼──────────────────────────────
      高       │    高     │     高      │ ★ 完美! LLM+检索+知识都是对的
  ─────────────┼───────────┼────────────┼──────────────────────────────
      高       │    高     │     低      │ Context 本身有错!
              │           │            │ → 检查知识库的准确性
              │           │            │ 可能是旧版本文档/过期政策
  ─────────────┼───────────┼────────────┼──────────────────────────────
      低       │    高     │     低      │ LLM 在编造!
              │           │            │ → 检查Prompt约束/temperature
              │           │            │ 或 Context 信息不足→LLM 补充
  ─────────────┼───────────┼────────────┼──────────────────────────────
      高       │    低     │     中      │ 检索不相关!
              │           │            │ → 检索系统返回了无关Context
              │           │            │ LLM老实照着念了，但文不对题
  ─────────────┼───────────┼────────────┼──────────────────────────────
      低       │    低     │     低      │ 全部崩盘!
              │           │            │ → 检索错 + LLM编造
              │           │            │ 需要从检索到生成全链路排查
═══════════════════════════════════════════════════════════════════════

  调优路径 (按优先级):
    1. Faithfulness低 → 加强Prompt约束(强制引用)/降低temperature
    2. Relevancy低   → 检查检索质量/Reranker/增大top_k或chunk_size
    3. Correctness低 → 更新知识库/检查Ground Truth时效性
```


---

