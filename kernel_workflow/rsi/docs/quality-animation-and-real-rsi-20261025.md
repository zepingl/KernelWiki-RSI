# KernelWiki-RSI：Quality 动图的含义与真实自进化实验

> 记录时间：2026-10-25 13:41:00 UTC。
> 代码快照：GEAK `KernelWiki-RSI` 分支，HEAD `755a0bccbca0f7caf2edd3dc19a9d7dcf80cf346`；核对时包含本地未提交的可视化和文档。
> 历史证据：2026-10-22 的归档实验及 schema-2 reconstruction 工件。
> 本次范围：只读核对并新增本文；不改案例、objective、正式 buffer 或活动 Knowledge，不启动模型/GPU 实验。

本文记录本轮关于动图、quality-only objective 和真实 KernelWiki 自进化的分析。
目标函数、最优策略与晋级约束的完整论证见[已有分析](replay-objective-analysis.md)；
下文明确区分**已核实事实**、**机制推论**和**尚未执行的实验建议**。

## 1. 直接回答：当前动图是不是只看 kernel perf？

**大号的 `1.08% → 7.78%` quality 对比是纯 kernel 性能分量；但整个实验并未改成 quality-only，也不是一次真实自进化成功。**

| 层次 | 当前含义 | 不能据此声称 |
| --- | --- | --- |
| 大号 quality 与 `+6.71 pp` | 相同预算下，构造 world 内到达的最佳 kernel 性能增益 `Q` | 已执行新 kernel 优化，或提升了端到端模型吞吐 |
| 策略卡片、侧栏 Replay Objective、策略选择 | 当前 Python replay 的 `J = Q − 0.01 × C`；并行系数为零 | 成本项已经移除 |
| 动图中的 loop | 手写策略与构造 discovery tree 上的离线对比 | Developer/Reviewer 已实际迭代出策略，或 KernelWiki 已发布新版本 |
| 归档实验卡片 | 旧 schema-1 实验的局部 replay 分数改善，最终 `no_change` | 当前 schema-2 loop 已通过真实验收 |

因此，准确表述是：

> 现有案例展示“在同一构造树和相同方向预算上，换一种探索决策，可以到达性能更好的历史测量节点”。
> 它既不是单纯由节省成本产生的 quality 增长，也不是本地 Dream-RSI 已完成 KernelWiki 自进化的证明。

## 2. Quality 增长的物理意义

### 2.1 当前案例采用的性能指标

[归档测量适配器](../reverse_engineering/sources/archive.py)为选中的 DSA sparse MLA 历史测量计算：

```text
L_process = Σ_case calls_case × optimized_ms_case
L         = median_process(L_process)
S         = 1 / L
Q         = S_best / S_root − 1 = L_root / L_best − 1
```

- `S` 的指标名称是 `inverse-call-weighted-latency-v2`，越大越好。
- `L` 是固定 workload 的调用次数加权 kernel GPU 时间，不是单次调用延迟。
- [Harness](../harness.py)只将已到达的 qualified 测量用于最佳质量；根节点也参与，初始 `Q=0`。
- `Q` 是无量纲的 root-relative 增益，不含预算惩罚、并行奖励或 Knowledge 版本数量。
- 归档的 `e2e_verified=false`；没有据此验证 GLM 端到端吞吐、tokens/s 或整体模型加速。

这解释了“为什么 quality 涨了”：策略在 replay 中到达了 `S` 更大的已有节点。
它度量探索结果的 kernel 性能，不直接度量知识内容是否更正确、更丰富或已被发布。

### 2.2 T₁ 的可追溯数值

原始 world：`node039/dsa-sparse-mla/f8dcf1b80680e86f`，正式 `worlds.jsonl` 第 127 行。

| 原始测量 | Primary score `S` | 相对 root 的 `Q` |
| --- | ---: | ---: |
| Root | 5.56792731178922e-06 | 0 |
| 弱节点，案例 A1 | 5.627899930851435e-06 | 0.01077108512807512 |
| 强节点，案例 A2 | 6.001370655346108e-06 | 0.0778464443382978 |

在构造 world 的预算 `2`、单 worker、每方向成本 `1` 下，现有 replay 得到：

| 手写策略 | 最佳 quality `Q` | 已消耗方向预算 `C` | `J = Q − 0.01 × C` |
| --- | ---: | ---: | ---: |
| 基线轮转 | 1.0771085% | 2 | -0.0092289149 |
| 第一方向后停止 | 1.0771085% | 1 | +0.0007710851 |
| 续探已测优叶 | 7.7846444% | 2 | +0.0578464443 |

大号 quality 的差是 **6.7075359 个百分点**，显示为 **+6.71 pp**。
基线与续探的 `C` 相同，扣除相同的 `0.02` 不改变这两者的排序；
提前停止策略的 objective 则确实受到成本项影响。

注意百分比的口径：`Q=7.7846%` 表示逆加权延迟 score 相对 root 增长 7.7846%；
对应加权延迟降低约 **7.2224%**，不是延迟降低 7.7846%。

## 3. 为什么保留案例，却不能把它改称“真实自进化”

[现有动图](../../docs/visuals/kernelwiki-rsi/index.html)与[构造代码](../../docs/visuals/kernelwiki-rsi/build_case.py)明确保留以下边界：

1. A1、A2 在原始记录中是 root 的兄弟节点；图中的 `A1 → A2` 是教学构造，不是 reverse engineering 恢复出的 continuation。
2. B0 复用 root 测量，不代表额外发生过一次真实优化尝试。
3. 构造节点的 source/context 身份来自既有 replay 单元测试 fixture，不冒充原始 kernel。
4. 三条策略为手写；实际执行的是 Python `evaluate()` 和现有 `ReplayObjective`，浏览器仅播放冻结的 `case.js`。
5. Developer、Reviewer、在线探索、逆向调查、冻结和 outer cycle 的动画顺序是编排，没有为该案例实际调用 Agent 或 GPU。
6. 第二轮加入另一个构造 world 后保留续探策略；不是每轮都强行产生改善。不同 buffer 之间不承诺绝对 objective 单调增长。

原始 attempts、测量、正式 buffer 与活动 Knowledge 均不因该案例改变。
案例可继续承担直观说明机制的作用；真实实验必须另建工件，不将教学连线混入正式 discovery tree。

## 4. 去掉 replay objective 后两项：能做什么，不能做什么

**可接入 quality-only 实验，但应新建并冻结配置，不删除已存在的 objective 机制，也不重写旧实验的评分。**

尚未实施的配置建议：

```yaml
objective:
  beta_cost: 0
  beta_parallel: 0
```

[Objective 实现](../replay_buffer/objective.py)已支持这两个系数为零；此时每条 trace 的 `J=Q`。
当前正式 manifest 和现有案例仍为 `beta_cost=0.01`、`beta_parallel=0`。

候选产生前还必须选择预算口径：

| 冻结预算 | 实际优化目标 |
| --- | --- |
| `[1, 2, 4, 6]` | 对多个预算下的质量做平均，偏向更早发现好节点，即 quality–budget 曲线 |
| `[6]` | 在固定最大方向预算 6 下的最终质量，不直接奖励更早发现 |

实际聚合顺序是预算 → world → kernel 的嵌套平均；不是将记录较多的 kernel 加权放大。
不得看完候选或 held-out 后更换预算，以制造赢家。

即使质量是唯一软目标，也必须保留方向预算、模型调用、GPU 资源和 wallclock 硬上限，
并继续记录成本、验证 correctness、支持完整性、配对覆盖与非回退。
将成本系数置零不等于这些约束消失，也不补出未记录的分支结果。

尤其需要区分：

- **事后 oracle 最优路径**：知道全树测量后挑最佳叶子，是诊断上界。
- **合法 KnowledgePolicy**：只依据当时已揭示的 observation 决定下一步，不能读取隐藏结果或按 world ID 记忆答案。
- **实际候选搜索**：有界 Opus 提案可能找到更好策略，也可能 `no_change`；不提供全局最优保证。

Knowledge-as-a-Policy 对应 Harness 的探索决策层。
Discovery tree 是策略与环境交互的产物，而不是把最好路径写进 KernelWiki 就等价完成策略进化。

## 5. 当前真实数据的阻塞点

已核实的正式 pool：

```text
/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_reconstruction_20261022/pool-schema2-04
worlds.jsonl SHA-256:
44a7d2d7cca7378e481e064054e82d4b523adff18573091b75129870ff776ed9
```

| 工件内容 | 数量 |
| --- | ---: |
| 历史 attempts | 1,498 |
| Replay-ready attempts | 528 |
| Qualified node outcomes | 153 |
| Diagnostic worlds | 337 |
| Primary worlds | 0 |
| Reverse reports | 3 |

本轮对话中的只读统计显示：528 条 recorded branches 的深度全部为 1；
337 个 world 均缺少受支持的“新分支 vs continuation”选择，284 个含 unknown outcome。

| 方向预算 | 记录成本不足的 world | 合格 primary world |
| --- | ---: | ---: |
| 1 | 0 | 0 |
| 2 | 198 | 0 |
| 4 | 336 | 0 |
| 6 | 337 | 0 |

缺口计数相互重叠。上述事实不证明历史工作从未长期探索，
只证明当前可核实的 source/context provenance 尚不能构成可重放的延续关系。

[Primary gate](../replay_buffer/cohorts.py)要求至少两个分支、至少一个 continuation、
记录成本覆盖最大冻结预算，且没有 unknown outcome；已测量失败可以保留，未知不能伪装为成功。
[Inner loop](../loop/inner.py)因此返回 `insufficient_primary_support`，Agent calls 为 0，
保留 incumbent、不评估 test、不发布。
**仅置零系数，甚至只把预算降为 1，均不能解除当前的结构支持缺口。**

另有旧 schema-1 实验确实得到更高 validation objective，但 full-budget quality 持平，
held-out 的 `o-proj` 质量回退，最终 `no_change`、`published=false`。
它是应保留的负实验，不是当前重构版的成功证据；已看过的旧 test 只能作 regression，不能重新称为 fresh holdout。

## 6. 怎样取得真实 Dream-RSI 驱动 KernelWiki 自进化的结果

以下是**待执行方案**，不是本文新增的实验结果。

```text
工作策略 θ_t
  → 固定 Harness / GEAK 真实在线探索
  → reverse engineering：核实源码、输入上下文、调用和独立测量
  → 冻结真实 replay buffer B_(t+1)
  → Developer / Reviewer / train replay 反馈 / 独立 validation
  → 选择 θ_(t+1)，或保留 θ_t
  → 被选策略实际控制下一轮在线探索

最终候选另走：fresh 配对 replay + online → qualification → 显式 CAS release
```

| 阶段 | 最小动作 | 通过条件 / 不通过时的下一步 |
| --- | --- | --- |
| 真实校准 | 一张 GPU、一个正常方向、至多 30 分钟的小试验 | Root qualification、生成与独立 host 测量实际成功；失败先缩小到环境或测量边界，不启动大实验 |
| 恢复 provenance | 从原始日志和快照核实 parent source、精确输入 context、raw calls 和 outcomes | 每条延续关系有真实依据；找不到就保留 unknown/gap，不凭时间顺序连边或重接兄弟节点 |
| 补采真实 tree | 用 GEAK workflow 记录新分支和真实 continuation | 满足冻结预算和 outcome 支持；不足则补采，不在已知不合格数据上反复调 objective |
| 冻结实验 | 固定 buffer 身份、split、预算、objective、解释器、测量协议与资源上限 | 候选和 incumbent 使用完全相同的评价问题；旧 GLM7 数据不冒充新 held-out |
| 运行真实 inner | 固定 `claude-opus-5.5 --effort high`，实际 Developer 提案、Reviewer 审查、train replay 反馈 | Validation 独立选择工作策略；内循环不读取 test、不发布；真实拒绝或 `no_change` 也保留 |
| 验证 outer 反馈 | 被选工作策略控制下一轮 online，再 reverse/freeze/evolve | 同一冻结 B 内配对比较旧/新 θ；不同 B 的绝对分数不当作因果改善证据 |
| 最终发布 | 冻结最终候选，取得 fresh 配对 replay 与真实 online 证据 | 通过质量及逐 kernel 非回退等 qualification 后，才显式 CAS release；记录真实 Knowledge revision |

校准特别注意：2026-10-22 的 trial 在 root qualification 约 6.695 秒失败，
原因是 workspace 的 `unittest.py` 遮蔽标准库、阻碍 Torch 导入；该 trial 的 generation/model calls 都为零。
Adapter 修复和 CPU 回归不等于已完成成功的 fresh GPU 校准，应重新完成这一窄门。

建议先做机制 pilot：三个 kernel 分置 train/validation/test、两轮 outer、每轮至多两个候选，
采用新冻结的 quality-only 配置。该 pilot 是新协议，不代表完成原七 kernel 的发布验证。
总模型调用、GPU 时间与 wallclock 上限需在启动前明确并冻结，不能只凭“两个候选”推算真实成本。

严格 primary 晋级资格不必自动禁止所有有界离线学习。
若先利用浅树做诊断学习，应另行冻结受支持的动作问题，保证候选/incumbent 覆盖一致，
不宣称学到了 continuation，也不放宽正式发布门禁。这是需要另行设计和验证的选项。

## 7. 什么证据才足以展示“成功案例”

必须分别报告三种状态，而不以一条上涨曲线代替它们：

| 状态 | 必需证据 |
| --- | --- |
| 找到更好的离线候选 | 相同 B、split、预算和 objective 下的配对结果；policy revision、规则 diff、Q/J 与支持检查 |
| 工作策略形成真实闭环并在线验证 | 实际 Agent 提案/审查/反馈记录；被选 θ 控制下一轮 online 的调用、真实 discovery tree、独立测量与配对结果 |
| KernelWiki 已实际发布进化版本 | Fresh qualification 通过、显式 release 记录和新 Knowledge revision；不能只显示 selected-policy 文件 |

结果工件应保留：真实 source/context/parent 引用、raw calls、host 测量收据、
buffer 身份与冻结协议、方向预算/模型调用/GPU 时间、候选拒绝原因和各轮选策决定。
实验可能只有 offline 改善，也可能真实 `no_change`；不得通过构造新连线或筛掉失败 world 把它变成发布成功。

**当前结论：现有 quality 展示是纯性能分量上的构造 replay 对比；尚无当前 schema-2 Dream-RSI 完整闭环成功、在线收益和 KernelWiki 新发布的证据。**

## 8. 核对入口与本次变更边界

- [现有案例说明](../../docs/visuals/kernelwiki-rsi/README.md)、[冻结案例数据](../../docs/visuals/kernelwiki-rsi/case.js)：数值来源、构造边界和两轮选策。
- [Harness](../harness.py)、[测量归档适配器](../reverse_engineering/sources/archive.py)、[Replay Objective](../replay_buffer/objective.py)：quality、物理指标与目标函数。
- [Cohorts](../replay_buffer/cohorts.py)、[Evaluate / qualifies](../replay_buffer/evaluate.py)：支持、配对、聚合与非回退。
- [Inner](../loop/inner.py)、[Outer](../loop/outer.py)、[Qualification / release](../loop/qualification.py)：工作策略选择、在线反馈及发布边界。
- [工程设计](../../../docs/kernelwiki-rsi-v1.md)、[归档实验说明](../../../docs/kernelwiki-rsi-v1-experiment.md)、[完整 objective 分析](replay-objective-analysis.md)：更完整的设计与原始工件索引。

本次仅新增时间戳分析文档；保留现有案例、配置、正式历史和活动 Knowledge。
真实实验的下一道门仍是 fresh 小规模校准与可验证的 branching/continuation 数据支持，不是修改展示数字。
