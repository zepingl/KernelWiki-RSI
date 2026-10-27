# KernelWiki-RSI：Replay Objective、最优策略与晋级阻塞分析

> 核对日期：2026-10-25。代码参考：GEAK `KernelWiki-RSI` 分支，`755a0bcc`。
> 实验证据：2026-10-22 的归档 schema-1 实验与 schema-2 reconstruction 工件。
> 本文整理机制、证据和待验证建议；不是一次新实验，也不改变目标函数、门禁或活动知识。

## 结论速览

| 问题 | 结论 |
| --- | --- |
| Dream-RSI 的 replay objective 有几项？ | 三项：发现质量、执行成本、并行奖励。 |
| 本地是否漏写第三项？ | 没有。公式已实现，但 GLM7 配置的 `beta_parallel=0`，当前不产生并行奖励。 |
| `0.01` 是实测成本吗？ | 不是。它是质量与方向预算之间的手工权衡系数，没有 GPU 时间、token 或货币标定。 |
| 固定 buffer 是否存在最优策略？ | 在非空、有限的可评价行为集合内存在最大值；不保证唯一、严格优于 incumbent、可泛化或已被搜索到。 |
| 当前为什么没有晋级？ | Schema-1 候选的 held-out 质量回退；schema-2 则在 primary-support gate 就停止，未进入候选搜索。 |
| 最值得质疑的耦合是什么？ | 当前把严格的 primary 支持同时用作离线搜索前置条件；“不能晋级”不应自动等同于“不能做有边界的离线学习”。 |

本文区分三种陈述：**已核实事实**来自代码或原始工件；**机制推论**说明这些事实意味着什么；
**改进建议**是尚未实施、仍需实验验证的方向。

## 1. 正在优化什么

### 1.1 KnowledgePolicy 不等于一棵事后挑出的最优树

本地的核心对应关系是：

**Knowledge-as-a-Policy = Harness 的探索决策层。**

- **KnowledgePolicy**：根据当前可见状态决定继续哪个父节点、如何安排批次、何时停止。
- **Harness**：解释结构化策略，执行预算、动作合法性和在线/replay 共用的状态转换。
- **Discovery tree**：某次探索实际产生的源码、上下文、调用和测量之间的可追溯关系。
- **Replay buffer**：能够支持确定性重放的历史 world，以及保留的原始 attempts、逆向报告和证据缺口。

树是策略与环境交互的结果，不是策略本身。事后知道哪个节点最好，可以选出漂亮路径，
却未必能写出一个仅凭当时可见信息就能选择该路径的通用策略。

当前策略由固定解释器执行结构化规则；它不是任由实验 Agent 重写整个 Harness、生成器或评测器。
策略看到的是已揭示的节点状态、质量、预算和停滞等信息，而不是未访问分支的未来测量结果。

### 1.2 Dream-RSI、reverse engineering 与 replay 的关系

```text
工作策略 θ(t)
    ↓
固定 Harness + GEAK 在线探索
    ↓
原始源码 / 父节点上下文 / 调用 / 独立测量
    ↓
reverse engineering：恢复关系、核实事实、定位证据缺口
    ↓
冻结 replay buffer B(t+1)
    ↓
Developer 提案 → Reviewer 审查 → train replay → 反馈与下一次修改
    ↓
validation 选择工作策略 θ(t+1)，否则保留 incumbent
    ↓
下一轮在线探索，扩大历史

最终发布是另一条门禁：
冻结最终候选 → 新鲜配对 replay / real online → 显式 release
```

这是本地实现的机制路径，不是声称该路径已经在真实实验中完整成功运行。

Reverse engineering 为策略学习提供可信关系与方法线索，但不生成新的 kernel 性能测量。
Analyst review 不能把“可能的优化方法”变成 qualified outcome，也不能把未保存的源码或上下文补成事实。
一次跨轮共享上下文更新或 merge，不能仅凭时间顺序就认定为可重放的父子 continuation。

Replay 本身不调用 discovery Agent 或 GPU；Developer/Reviewer 的模型调用发生在策略修订环节。
Replay 只在已有结果上改变可执行探索决策，不能凭空创造未记录的新 kernel。

## 2. Dream-RSI 原论文的三项 replay objective

### 2.1 原始公式

[Dream-RSI 第 3 节，Replay objective，公式（1）](https://arxiv.org/html/2609.14858v1#S3.E1)：

```text
V = Q − β₁ × N + β₂ × N / max(1, K)
```

| 符号 | 含义 | 激励 |
| --- | --- | --- |
| `Q` | Replay 已到达节点中的最佳解质量，包含根节点 | 找到更好的解 |
| `N` | 已揭示的非根节点数，代表 generation–evaluation requests | 减少尝试成本 |
| `K` | Replay 停止时的决策轮数 | 与 `N` 一起衡量批处理程度 |
| `β₁ × N` | 执行成本惩罚 | 少做无效探索 |
| `β₂ × N / max(1, K)` | 平均每轮尝试数对应的并行奖励 | 将可并行的探索放入同一批，而不是逐个串行执行 |

论文允许固定的 `β₁, β₂ ≥ 0`。因此，第三项取零在数学上是合法特例，
但不能把它称为“三项都已参与优化”的实验。

论文的策略选择包含 incumbent，因此在固定历史和同一评分下，选出的版本平均 replay score 不低于 incumbent。
这是**已评价候选集合内的非回退选择**，不是全局最优证明，也不保证每轮严格改善或真实在线改善。

### 2.2 本地公式与对应关系

实现见 [ReplayObjective](../replay_buffer/objective.py)：

```text
J = best_quality
    − beta_cost × charged_budget
    + beta_parallel × attempted_nodes / max(1, decision_batches)
```

| 本地字段 | 实际语义 | 与论文的关系 |
| --- | --- | --- |
| `best_quality` | 最佳 qualified 主分数相对 root 的归一化增益，最低为零 | 质量项的本地归一化版本 |
| `charged_budget` | 实际消耗的方向预算单位 | 成本代理；不直接等于论文的 `N` |
| `attempted_nodes` | 已记录并揭示的尝试节点数，失败和 unknown 尝试也计数 | 对应尝试数 |
| `decision_batches` | 非空决策批次数 | 对应决策轮数 |
| `parallel_bonus` | `beta_parallel × attempted_nodes / max(1, decision_batches)` | 并行奖励 |

质量由 [Rollout](../harness.py) 更新：

```text
quality = max(0, best_qualified_primary_score / root_primary_score − 1)
```

失败与 unknown outcome 不提高质量，但已记录的尝试仍消耗预算。
完全没有记录支持的转移与“记录了失败结果”不同：不能把缺失转移当作零成本成功，
也不能用它取得合格的策略改善；qualification 会拒绝不完整的 replay 支持。

### 2.3 当前实际上启用了哪些项

[GLM7 配置](../replay_buffer/configs/glm7.yaml) 为：

```yaml
budgets: [1, 2, 4, 6]
objective:
  beta_cost: 0.01
  beta_parallel: 0
```

代入后：

```text
J = best_quality − 0.01 × charged_budget
```

因此，之前只写“两项公式”是在描述**当前生效配置**，不是完整实现或论文公式。
当前优化直接偏好“质量更高、探索更省”；同样的质量和消耗，仅改变批次安排不会获得直接奖励。

文档将默认关闭并行项的理由表述为：不能把 batch counts 宣称为真实 wall-clock savings。
这个证据边界应当保留，但它不意味着并行结构指标本身不可使用：

- 论文第三项本来就是逻辑指标，不要求先测得 GPU 耗时才能定义。
- 启用前需要核实 simulator 的批次确实对应 GEAK 可以执行的并行决策。
- 本地正常模式的 worker limit 为一或二，deep exploration 固定为一；奖励不能超出实际动作能力。
- 即使指标提高，也只能先声称批处理结构改善，不能直接声称在线时间或 GPU 吞吐改善。

## 3. `−0.01` 的含义与优化偏向

### 3.1 它是权衡系数，不是物理成本

`beta_cost=0.01` 表示：**每多消耗一个方向预算单位，扣除 0.01 的归一化质量分数。**
也就是将一个普通探索方向的惩罚设为相当于 1 个百分点的 root-relative 主分数增益。
这不是说该方向消耗了 1% 的 GPU 时间，也不是说它必然使 latency 降低 1%。

[GenerationSpec.cost](../discovery.py) 定义普通探索为一个方向单位、deep exploration 为两个。
这些单位不是模型调用数、token、GPU seconds 或货币；deep 的两个单位也不证明实际耗时恰好翻倍。
目前没有完成可据以物理标定这个系数的真实成本校准。

### 3.2 为什么 objective 改善可以伴随质量下降

以下是解释公式的**假设算例，不是实验结果**，并行项取零：

| 策略 | 最佳质量 `Q` | 方向消耗 `C` | `J = Q − 0.01C` |
| --- | ---: | ---: | ---: |
| A：多探索，质量较高 | 0.03 | 3 | 0.00 |
| B：更早停止，质量较低 | 0.02 | 1 | 0.01 |

标量 objective 偏好 B，因为节省的成本超过损失的质量。
如果这些质量同时代表 full-budget quality，独立的质量非回退门禁仍会拒绝 B。
这说明“成本节省是否允许交换质量损失”是目标与约束共同决定的，不是公式计算错误。

同理，负 objective 本身不是实验失败。资格判定比较的是候选相对 incumbent 的变化，
并同时检查支持与质量，而不是要求分数变为正数。

### 3.3 并行奖励会改变什么

另一个**假设算例**：在可合法执行这些批次的条件下，两种安排同样尝试四个节点，
具有相同最佳质量、相同方向消耗：

| 安排 | 决策轮数 `K` | 尝试数 `N` | 并行指标 `N/K` |
| --- | ---: | ---: | ---: |
| 逐个串行 | 4 | 4 | 1 |
| 每批两个 | 2 | 4 | 2 |

当 `beta_parallel=0` 时，两者 objective 相同；当系数为正时，第二种获得更多奖励。
这不保证第二种安排能在任意 discovery tree 中实现：父子依赖和已揭示状态仍限制可并行性。

启用第三项也不是自动修复方案。权重过大可能奖励为了凑批次而做的低价值尝试；
需要在固定 train/validation 上分析质量、成本、批次安排之间的 trade-off，再冻结权重。
不能根据 held-out 失败反复改系数，直到制造一次“晋级”。

## 4. 当前评分与 qualification 不是同一件事

### 4.1 多预算、world 与 kernel 的聚合

[aggregate](../replay_buffer/objective.py) 按以下顺序计算：

1. 同一 world 内，平均各个冻结预算的 objective。
2. 同一 kernel 内，平均各个 world 的结果。
3. 对 kernel 做 macro-average，而不是让历史记录更多的 kernel 占据更大权重。

`full_budget_quality` 单独计算：取每个 world 最大预算对应的最佳质量，再按 world/kernel 聚合。
即使策略提前停止，仍使用该最大预算 replay 实际达到的质量，不假设它后来还能得到未访问节点的收益。

### 4.2 当前代码确实有独立非回退约束

[qualifies](../replay_buffer/evaluate.py) 要求：

- 候选与 incumbent 使用完全相同的 world、kernel、budget 配对；不能按候选挑选或丢弃不利样本。
- 双方都具有合格的 replay 支持。
- 普通 train/validation 比较要求 objective 严格改善，阈值为 `1e-12`。
- Macro full-budget quality 不回退，容差为 `1e-12`。
- Held-out 比较允许 objective 持平，但还要求每个 kernel 的 full-budget quality 不回退。

当前 [inner.evolve](../loop/inner.py) 用这个资格函数决定是否推进 training editing base，
validation 也用它选择工作策略。因此，不能把当前训练机制描述为“只优化标量 objective、完全没有质量约束”。

内循环不评估 test、不发布。最终 [qualification / release](../loop/qualification.py) 另外要求新鲜配对
replay 和真实在线证据；通过 CPU fixture 或一次诊断 replay 不等于策略发布资格。

## 5. 两版实验分别为什么没有晋级

### 5.1 归档 schema-1：找到了 replay 分数更好的候选，但质量回退

归档实验使用 `claude-opus-5.5 --effort high`，执行四次 Developer 和四次独立 Reviewer 调用。
候选 1 的核心行为是停滞后提前停止。

| 指标 | Incumbent | 候选 1 | 判断 |
| --- | ---: | ---: | --- |
| Validation objective | -0.0093096814 | -0.0085650006 | 改善 |
| Validation full-budget quality | 0.0057945626 | 0.0057945626 | 持平 |
| Held-out objective | -0.0084097556 | -0.0080863459 | 改善 |
| Held-out full-budget quality | 0.0055477347 | 0.0054050484 | 回退 |
| Held-out `o-proj` quality | 0.0067211577 | 0.0062930989 | 回退；其余两个 kernel 持平 |

结果为 `no_change`、`published=false`，保留 incumbent，没有根据 held-out 结果重选候选。
这是“replay 成本—质量权衡没有通过最终质量约束”的负实验，不是“搜索没有产生任何更高分候选”。

这些都是归档实现上的 replay 指标，不是新 GPU kernel speedup。
Schema 2 改变了上下文、动作合法性、支持规则和分数归一化；不能把旧实验移作新 loop 的验收结果。
原 test 已经被检查过，只能继续作为 regression 数据，不能重新标记为 fresh holdout。

详见[归档实验说明](../../../docs/kernelwiki-rsi-v1-experiment.md)及第 9 节原始工件。

### 5.2 Schema-2：primary support 为零，搜索尚未启动

已核实的 `pool-schema2-04`：

| 数据 | 数量 |
| --- | ---: |
| 历史 attempts | 1,498 |
| Replay-ready attempts | 528 |
| Qualified node outcomes | 153 |
| Diagnostic worlds | 337 |
| Primary worlds | 0 |
| Reverse reports | 3 |

[qualification_gaps](../replay_buffer/cohorts.py) 在策略评价之前要求：

1. 至少两个记录分支，并且至少一个分支有 continuation。
2. 记录节点的方向成本总量覆盖最大冻结预算，目前为六。
3. 没有 unknown evaluation outcome。

原始 world 工件的只读统计进一步确认：

| 支持缺口 | 涉及 world 数 |
| --- | ---: |
| 没有支持“新分支与 continuation”选择的结构 | 337 |
| 记录成本不足以覆盖最大预算 | 337 |
| 含 unknown outcome | 284 |

528 条可重放分支的记录深度全部为一。上表缺口相互重叠，不能相加解释为不同 world 的总数。
这不证明原始优化从未存在长期探索；它表明当前可验证的 source/context 关系尚不足以将其组成受支持的 continuation。

原始 provenance 缺口还包括：缺少 private-workspace source observations（836），
初始 `codebase_context` 版本（84）、`profiling_summary` 版本（31），
private workspace 不从观察到的 parent 开始（9），以及不受支持或复用的 dispatch 输入（10）。
这些是 manifest 的缺口计数，不是新实验测量。

[inner.evolve](../loop/inner.py) 因 train/validation primary 覆盖不足而提前返回：

```json
{
  "status": "insufficient_primary_support",
  "primary_worlds": 0,
  "diagnostic_worlds": 337,
  "agent_calls": 0,
  "opportunities": [],
  "selected": null,
  "test_evaluated": false,
  "published": false
}
```

所以，schema-2 的直接阻塞不在候选 objective 比较，而在搜索前的支持门。
仅调整 `0.01` 或开启第三项，不会使已经提前返回的 loop 开始搜索。

### 5.3 真实校准：环境边界失败，没有形成新测量

2026-10-22 的单方向真实校准在 root qualification 阶段失败。
失败工件记录约 6.695 秒、零 generation requests、零模型调用。
原因是 workspace 的 `unittest.py` 遮蔽标准库，使 adapter 导入 Torch 失败，不是 kernel 性能失败。

Adapter 修正后，CPU 回归通过；这不能追认失败的 GPU trial 为一次成功成本校准。
当前这些证据不支持声称获得了新真实 kernel 收益、完成 fresh holdout 或发布了新知识。
记录中的活动知识版本仍为：

```text
d166eedd01c6ceadacfa8f9549d551e617ff6be62efa994e2efec3c7f880a8da
```

## 6. “给定 buffer，肯定应该有 optimal policy”的正确部分与边界

### 6.1 存在性：这个直觉在限定问题中成立

固定 buffer、预算、动作规则和 objective，定义非空的可评价策略集合 `Π_B`：

```text
π* ∈ argmax(π ∈ Π_B) J_B(π)
```

有限节点、有限动作和有界决策轮数使 replay 可产生的不同轨迹组合有限。
即使策略文本有许多写法，只要可评价行为集合非空，评分最大值可以被某种行为达到。

但“可评价”必须先定义清楚。若坚持当前所有 primary 资格要求，而给定数据无法满足，
就不能直接假定合格的比较集合非空。应区分更窄的诊断优化问题与严格晋级问题。

### 6.2 四个不能跳过的区别

| 区别 | 为什么重要 |
| --- | --- |
| 存在最优值 ≠ 当前搜索找到了它 | 四次 LLM 提案不是穷举，也不提供 optimality certificate。 |
| 最优值存在 ≠ 严格优于 incumbent | Incumbent 可能已经最优，或多种策略产生同一轨迹与分数。 |
| 事后最优路径 ≠ 可部署策略 | Oracle 看见隐藏结果；合法策略只能利用当时可见 observation 和可泛化规则。 |
| Buffer-optimal ≠ 可晋级或在线最优 | 还要满足支持、质量、独立验证和真实在线证据，且 replay 无法创造 buffer 外的新结果。 |

如果允许按 world ID 记忆最佳分支，就容易“构造”离线赢家；但这绕过了策略所需的信息边界。
更合理的区分是：全信息 oracle 作为诊断上界，合法候选策略作为实际优化对象。
二者的差距可以帮助识别信息不足、搜索不足或策略表达能力不足，但 oracle 本身不能晋级。

### 6.3 因而不能从“没有晋级”推导“buffer 没有价值”

已有证据只支持：

- 归档搜索发现了更高 replay 分数的候选，但它没有满足最终质量要求。
- 当前重构 loop 因 primary 支持不足，没有执行新的候选搜索。
- 尚无证据证明 incumbent 是 buffer-optimal，也没有证据证明不存在更好的合法离线策略。

“可用于晋级的数据不足”与“没有任何可利用的离线学习信号”不是同一个结论。

## 7. 需要质疑的工程耦合：离线学习与发布资格

### 7.1 当前耦合

```text
没有完整 primary train/validation
    → 不创建 Developer/Reviewer 实验
    → 不搜索离线候选
    → 保留 incumbent
```

严格的 primary gate 保护正式资格判定，尤其防止利用缺失转移、unknown outcome 或不配对覆盖制造高分。
但是，是否必须用同一个严格门禁阻断所有离线学习，仍值得作为独立需求审查。

### 7.2 建议拆分的两个结果

以下是**设计建议，当前未实施**：

| 结果 | 所回答的问题 | 不承诺什么 |
| --- | --- | --- |
| `best_found_on_buffer` | 在预先声明的数据支持与动作范围内，搜索到了什么更好的合法策略？ | 不承诺全局最优，也不直接发布 |
| `promotion_eligible` | 这个候选是否具备正式非回退和发布所需证据？ | 不因离线高分而豁免支持或 fresh holdout |

两者不需要引入新的数据库、workflow engine 或复杂状态注册表；可以先用已有实验结果表达。
关键是允许“有边界的离线学习成功、发布资格仍不足”成为真实可报告的状态。

### 7.3 更窄的离线实验也必须有统一边界

若要在现有浅层数据上探索，需在候选产生前固定：

- 可使用的 kernel、world、预算与动作集合。
- 何种已记录 outcome 足以进入该实验，unknown 如何处理，以及哪些行为不可评价。
- 候选与 incumbent 共用的覆盖，不按策略动态丢弃不利 world。
- 对结果可作出的声明，例如只验证早停或批次结构，不宣称学会长程 continuation。

这可以定义一个新的、受限的学习问题，但不能通过删除未知事实、补造分支，
或事后缩小样本来把原本不合格的实验改称正式晋级。

## 8. 后续验证方向与优先级

本节是分析得到的建议，不是本次文档任务已执行的重构或实验。

| 优先级 | 待回答问题 | 最小证据 | 成功后的合理结论 |
| --- | --- | --- | --- |
| P0 | Primary 缺口究竟缺什么，哪些可从原始追溯恢复？ | 对 source/context、continuation、outcome 与预算缺口逐项核实 | 明确可恢复支持和必须新采集的数据 |
| P1 | 现有数据能否定义非空、合法、统一覆盖的受限学习问题？ | 冻结 train/validation 范围与动作边界 | 可以开始诊断优化，但不自动具备发布资格 |
| P2 | 搜索能力是否足够，incumbent 距离 buffer 上界多远？ | 简单合法基线、受限枚举/搜索及独立的 oracle 上界 | 定位搜索、信息或表达能力差距 |
| P3 | 三项目标如何权衡，`0.01` 是否合适？ | Train/validation 上的质量—成本—批次 trade-off 和系数敏感性 | 冻结有解释依据的实验 objective |
| P4 | Batch 指标是否对应真实 GEAK 执行能力？ | 对比共享 Harness 的合法动作、父子依赖与在线批次记录 | 可以声称逻辑批处理改善 |
| P5 | 离线候选是否具有真实在线收益并可发布？ | 独立授权的新校准、fresh holdout 与配对在线 gate | 仅按实际通过的门禁声明收益与发布 |

应保留的硬约束：

- 原始 evidence、失败结果和 unknown 不被改写；source/context 关系不能由模型推测替代。
- Discovery Agent、解释器、评测器、参考知识与 inner-loop objective 保持冻结。
- 权重与候选选择不使用 fresh holdout 反馈；失败后不能在同一 holdout 上反复重选。
- 工作策略选择与活动 KernelWiki 发布分离。
- 模拟改善、历史逆向恢复、CPU 回归和真实在线收益分别报告，不相互冒充。

**综合判断：问题不只是“objective 少了一项”，也不能简单归结为“buffer 太差”。
首先要区分学习边界、搜索过程和发布资格，再用三项清晰的目标与可追溯证据评价策略。**

## 9. 代码与证据索引

### 9.1 代码所有权

| 代码 | 所负责的事实或机制 |
| --- | --- |
| [harness.py](../harness.py) | 可见 observation、动作合法性、质量与预算计数、策略解释 |
| [discovery.py](../discovery.py) | Discovery 领域值与普通/deep 方向成本 |
| [objective.py](../replay_buffer/objective.py) | 三项公式、默认系数、macro 聚合 |
| [glm7.yaml](../replay_buffer/configs/glm7.yaml) | 冻结预算、实际 objective 参数、kernel splits |
| [cohorts.py](../replay_buffer/cohorts.py) | 策略无关的 primary 支持条件 |
| [simulator.py](../replay_buffer/simulator.py) | 已记录转移的确定性 replay，不生成新结果 |
| [evaluate.py](../replay_buffer/evaluate.py) | 配对评价、支持资格、objective 与质量非回退 |
| [inner.py](../loop/inner.py) | 搜索前 primary gate、Developer/Reviewer 反馈、editing base 和 validation 选择 |
| [qualification.py](../loop/qualification.py) | 新鲜配对 replay/online 判定与显式 release |
| [reverse_engineering/](../reverse_engineering/) | 原始追溯、逐缺口调查、事实与方法分析的边界 |
| [当前机制 README](../README.md) | 整体代码地图和使用入口 |
| [完整工程说明](../../../docs/kernelwiki-rsi-v1.md) | 重构验收、校准失败与修正记录 |

### 9.2 原始实验工件

Schema-2 证据根目录：

```text
/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_reconstruction_20261022/
├── pool-schema2-04/manifest.json
├── pool-schema2-04/worlds.jsonl
├── historical-primary-gate/result.json
├── REPORT-schema2.md
├── reverse-one-trip/result.json
├── reverse-moe/result.json
├── reverse-absorb/result.json
└── calibration-work/calibration-01/failure.json
```

Schema-1 归档根目录：

```text
/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_v1_20261022/
└── round-opus-01/
    ├── manifest.json
    ├── result.json
    ├── incumbent-validation.json
    ├── candidate-1-validation.json
    ├── incumbent-test.json
    └── candidate-1-test.json
```

两套工件必须分别解释。本文的统计来自只读核对，数值展示保留十位小数；
没有重新运行模型/GPU，没有重新选择候选，也没有修改实验或知识状态。
