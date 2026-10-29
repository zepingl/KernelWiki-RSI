# KernelWiki-RSI v2：历史 GLM 数据的 replay 诊断与修复计划

> 时间戳：2026-10-29
>
> 本文是 [discovery-tree-design.md](discovery-tree-design.md) 第一步"用现有数据只读重组"的结果，并给出修复计划。全程只读原始数据，没有改代码。
>
> 数据来源：
> - 冻结的 schema-2 buffer `pool-schema2-04`；
> - 5 个 GLM7 campaign 的原始 audit run 索引；
> - 两个可用 campaign（node039、node056-0922）的 `plan_round` 结果。
>
> 脚本和产物在 `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_schema3_shape_20261029/`（见文末"复现"）。

## 结论先行

1. **贪心不是问题。** lane 每一步从当前最优节点出发、规划几个方向、只继续最好的那个。这仍然是一棵树，而且原始日志把它完整记下来了：主干、每一步的分叉、每个方向的 `kk_refs`、派发顺序和 stop 都能精确恢复。
2. **buffer 缩水有三个独立原因：**
   - 建 buffer 时没把每一步接成链；
   - 兄弟的结果被判为"没有测量"；
   - 贪心本身让"换一条路走下去"无法评估。

   前两个是表示方式和规则造成的，可以修；只有第三个是贪心固有的。
3. **在数量上，测量规则是主因，而且大部分不用 GPU 就能补回。** 有 168 个方向其实已经用同一个 `audit_bench` 做过合格测量，只是因为发起者是 engineer 而不是 verify 调用，被判为没有测量。补回之后，能完整 replay 的多方向 expansion 预估从 **15 个增加到 111 个**（共 150 个）。第 5 节第 1 步已经实现，按新规则重新 ingest 后实测为 **123 个**，其中 validation 22 个。
4. **knowledge 标签齐全，但多数只是"指向"，不是"读到"。** 815 条 `kk_refs` 中，执行方向的 engineer 读过的只占 10%。

## 1. 贪心 lane 形成的树

```
P0 ─┬─ a1（最好）─┬─ b1（最好）─┬─ c1 …
    ├─ a2         ├─ b2         └─ c2
    └─ a3         └─ b3
```

每一步是一次 Expansion（一个 plan 及其派发的方向），兄弟节点带着各自的方向和 knowledge。被继续的那一个构成主干。

replay 不跑 GPU，只揭示历史上记录过的结果（`simulator.py`：策略选中一个没有记录子节点的节点时，这一步记为 unsupported）。所以：

- **单步选择可以评估：** 这一步派哪些方向、带哪些 knowledge、派几个、要不要停。前提是每个兄弟都有可信的结果。
- **换一条路径不能评估：** 比如第 1 步改成继续 a2。a2 下面从来没被展开过。这是贪心固有的限制，只能靠在线收集。

## 2. 诊断

### 2.1 v1 把每一步存成了孤立的深度为 1 的 world

| 现象 | 数据 |
| :-- | :-- |
| world 和 branch | 337 个 world；528 条 branch，长度**全部为 1** |
| primary | 0。337 个 world 全部因 "no supported choice between a new branch and continuation" 被排除（`replay_buffer/cohorts.py:14`） |
| 链接字段 | 1,498 条 attempt 的 `parent_attempt_id` **全部为空**，ingest 从不赋值，`reconstruct()` 因此无法连接延续（`discovery.py:253`） |
| context 拆分 | 337 个 world 中有 300 个和别的 world 共享 root 源码，同一源码最多被拆成 17 个 world |

光给 `parent_attempt_id` 赋值也不够。v1 要求子节点的 `input_context` 等于父节点的 `context.revision`（`discovery.py:185`、`harness.py:93`）。而父节点的 revision 把它自己的 call id 也算进了 hash（`archive.py:336`），下一步的输入 context 又包含所有兄弟写入的 memory，所以这条检查**按构造永远通不过**。这正是 discovery-tree-design 第 5 节要删掉的限制。

按源码 hash（父源码 → 产出源码）直接重建，真实形状如下（21 组 campaign × kernel）：

- 最长主干 1–13 步；每个父节点派出 2–22 个子节点。
- 每个父节点**最多只有一个**子节点被继续：80 个父节点各继续了 1 个，35 个是死胡同，没有一个同时继续两个子节点。
- 能比较分数的 63 次继续中，60 次选的是兄弟里分数最高的。

### 2.2 方向和 knowledge 可以恢复，但没有入库

5 个 run 共 746 个 plan，对应的原始索引如下：

| 项目 | 数据 |
| :-- | :-- |
| 每个 plan 的方向数（1 / 2 / 3 / 4） | 483 / 183 / 74 / 1 |
| 规划的方向全部派发 | 739 / 746 |
| 方向带 `kk_refs` 或 `learned_refs` | 1,075 个方向中 778 个 |
| 多方向 plan 中，兄弟的 knowledge 不同 | 258 个中 239 个 |
| engineer 收到的方向与 plan 原文精确匹配 | 1,141 / 1,141 |

但这些信息进 buffer 时都丢了：

- `archive.py:334` 计算 `input_context` 时排除了 `DIRECTION`、`SPECIALTY`、`KK_REFS`。把它们当作动作而不是状态，这本身是对的，但排除后它们没有存到任何地方。
- `Attempt` 和 `DiscoveryNode` 没有方向或 knowledge 字段。
- replay 的 `Action` 只有 `parents` 和 `rule_ids`，策略无法选择"派哪个方向"。

另外，node058 没有 `knowledge_lineage` 索引，node001 的 `agent_calls` 只有 1 行，这两个 run 无法做 knowledge 归因。

### 2.3 按 Expansion 重建（node039 + node056-0922）

| 项目 | 数据 |
| :-- | :-- |
| plan | 622 个 = 363 个 Expansion + 259 个显式 stop。另有 5 个 plan 的 JSON 无法解析，其下 10 个方向未纳入 |
| Expansion 宽度（1 / 2 / 3 / 4 / 6） | 213 / 96 / 51 / 1 / 2 |
| 只派发 plan 前缀的 Expansion | 2 个 |
| Node 等级（共 574） | **E 121 · A 397 · T 56** |
| integrate | 15 次调用，本次未重建 |

等级按 discovery-tree-design 第 3 节判定：
- **E**：有独立测量，兄弟共享输入 context，且处于 launch 的第 1 轮。
- **A**：父子关系是观测到的，但缺测量，或 context 混入了兄弟写入的 memory。
- **T**：父源码或 context 没有观测到。

E 是上限估计：跨 launch 时 memory 是否继承，在索引里无法验证。

### 2.4 knowledge 证据：大多停留在"指向"

815 条 `kk_refs`，按 discovery-tree-design 中 `KnowledgeUse.channel` 的口径归类：

| 证据 | 数量 | 对应 channel |
| :-- | --: | :-- |
| 执行方向的 engineer 读过正文 | 78（10%） | `read`，观测到 |
| 同一 launch 中 tech lead 在 plan 之前读过 | 379（46%） | 规划阶段的 `read`；投递给 engineer 是推断的 |
| 只出现在 plan 中 | 358（44%） | `pointed`，属于声明 |

另有 206 个方向没带任何 `kk_refs`。

### 2.5 主干依据的结果大多不可见

被继续的方向共 226 个：
- 74 个是测得最好的；
- 1 个不是；
- **151 个没有独立测量**：选它们依据的是 engineer 自报，或没落盘的 verify 结论。

262 个 Expansion 的父节点后来又被规划了一次：148 次在同一 launch 内（本轮没被采纳），114 次在后续 launch 中重新出发。

### 2.6 每个方向的结果从哪里来

把 574 个方向的结果来源逐一核对（用 `archive.py` 自己的 `measurement()` 验证）：

| 结果来源 | 方向数 | 能否补回 |
| :-- | --: | :-- |
| 已有 verify 测量 | 151 | 已合格 |
| 输出源码与父节点完全相同（没有改动） | 174 | 结果等于父节点的分数，不需要 GPU |
| engineer 发起的 `audit_bench` 测量 | 168 | 已合格，只因发起者不是 verify 调用而被排除（`archive.py:347`），不需要 GPU |
| 正确性失败 | 10 | 本身是已知结果（`incorrect`） |
| 源码已归档，但没有合格测量 | 56 | 需要 GPU 重测，每个约 24 秒（p90 36 秒） |
| 没有记录源码 | 15 | 无法恢复 |

那 168 个方向的测量都满足：两个独立进程、正确性通过、逐 case 有效、源码和合同未变，评分合同与该 kernel 的根合同一致。其中 77 个方向被测过不止一次，各次结果最大和最小之间的差距中位数是 0.66%（p90 为 2.05%），所以挑运气好的那次结果的风险很小。

## 3. 各类决策的 replay 支持

只看多方向 Expansion（150 个），要求其中**所有方向**都有可用结果：

| 补齐方式 | 能完整 replay | train / validation / test |
| :-- | --: | :-- |
| 现状 | 15 | 7 / 1 / 7 |
| + 没改动的方向记为父节点分数 | 35 | 16 / 4 / 15 |
| + 采用 engineer 发起的测量 | **111** | 50 / 19 / 42 |
| + GPU 重测（上限） | 146 | 62 / 24 / 60 |
| **第 5 节第 1 步实施后实测** | **123** | 50 / 22 / 51 |

其他决策在现状下的支持情况：

| 决策 | 有记录 | 现状下能完整 replay |
| :-- | :-- | :-- |
| 有 knowledge 和无 knowledge 的兄弟对比 | 17 个 Expansion | 3 个 |
| 宽度（按派发顺序取前 k 个） | 150 个 | 前 2 个都已测量的 19 个；91 个连第一个都没有测量 |
| 早停（launch 内 ≥2 轮） | 109 个 launch，259 次显式 stop | 5 个 launch |
| normal 与 deep 的选择 | 3 个父节点两种都试过 | 3 |
| 从非最优子节点继续 | 1 次 | 不可评估（贪心固有） |

## 4. 三个原因汇总

| 原因 | 后果 | 和贪心有关吗 | 解决办法 |
| :-- | :-- | :-- | :-- |
| 没把每一步接成链 | 深度全是 1，primary 为 0 | 无关 | 按 discovery-tree-design 改为 Expansion 结构，用源码 hash 连接 |
| 兄弟的结果被判为没有测量 | 150 → 15 | 无关 | 修正测量采信规则；没有改动的方向直接继承父节点分数；剩余的重测 |
| 只有主干被继续 | 无法评估换一条路径 | **有关** | 只能靠在线收集模式（偶尔从第二名继续） |

## 5. 修复计划

每一步都有通过条件，前一步没通过就不进入下一步。

### 第 1 步：修正历史测量的采信规则（不需要 GPU）· 已完成，2026-10-29

**实现**（`reverse_engineering/sources/archive.py`）：测量结果属于"源码 + 评分合同"，不属于发起测量的那次调用。新增的 `AuditReader.candidate_measurement(source, contract)` 取代了原来两处"只认 verify 调用"的筛选（`attempt()` 和 `independent_pair()`）：

- **采用 engineer 发起的测量：** 该源码所有通过 `measurement()` 全部检查、且合同相同的测量都算数。测了多次时取下中位数（一次真实测量，带完整证据），不取最好值。
- **没有改动的方向：** 不需要特殊处理。源码和父节点相同，父节点自己的 baseline 测量自然会被采用，没有凭空构造任何结果。
- **正确性失败：** 每个进程都在源码和合同未变的情况下报告不正确时，记为 `failed`。如果同一源码既有合格测量又有正确性失败，记为 unknown（"conflicting correctness evidence"），不替它下结论。
- **证据缺失的测量：** 某次测量的回执文件已不存在时跳过这一次，不让它把整个 attempt 变成 trace_only。只捕获"文件缺失"；内容与 hash 不符仍然报错。

**通过条件与结果：**

| 条件 | 结果 |
| :-- | :-- |
| 能完整 replay 的多方向 Expansion ≥ 100，且 validation 上 > 0 | **123 / 150**（train 50 · validation **22** · test 51），原来是 15 |
| 新增测试覆盖各种来源和"取下中位数" | `test_archive.py` 新增 2 个测试：中位数、继承父节点、换合同为 unknown、缺失回执被跳过、正确性失败、冲突证据 |
| 已有 RSI 测试全部通过 | 108 passed |

新 buffer `pool-schema2-06`（原 buffer 未覆盖）与 `pool-schema2-04` 对比：
- attempt 和 disposition 完全不变（replay_ready 仍为 528）；
- 合格结果从 153 增加到 516，其中 1 个变为 failed，1 个因冲突证据变为 unknown；
- 原来已合格的 attempt 中，有部分因取多次测量的下中位数而分数略有变化。

primary 仍为 0，因为 world 结构还没改，这要靠第 2 步解决。

中间产物 `pool-schema2-05` 是修复"证据缺失"之前的版本：其中 36 个 attempt 被误判为 trace_only。保留它只用于对照，不要使用。

### 第 2 步：按 Expansion 结构重建 buffer（schema 3）· 已完成，2026-10-29

按 [discovery-tree-design-v2.md](discovery-tree-design-v2.md) 实现了阶段 1（trace 核心）和阶段 2（基于树的 replay）。v1 world 暂时并行保留。

**实现：**

- **`trace/`（新包）**
  - `records.py`：7 种只追加记录。
  - `tree.py`：fold、不变式检查，以及派生的 E/A/T、结果、knowledge ledger 和 gap worklist。
  - `gaps.py`：Gap 和 Coverage 从 `reverse_engineering/model.py` 移入。
- **`reverse_engineering/sources/archive_tree.py`**：把 audit 归档转成 trace 记录。
  - **树的范围：** 每个 campaign×kernel 一棵树（你在开发中确认的边界），launch 之间按"同一源码"连接。
  - **新段：** 如果一个 launch 的起点源码是 integrate、validate 或 setup 产出的、没有被重建的源码，就以它实测的 baseline 为 root 开一个新段，不编造边。
  - **跨 launch 信息：** 后续 launch 的 plan 通过 `HISTORY` 输入看到了更早的结果，这部分记入 `informed_by`，所以评为 A 级。
  - **测量范围：** 只取节点所在 launch 内的测量。
- **`harness.py`**：连续性检查改为 `node.continues(source)`，harness 不再依赖 v1 的 `discovery`。
- **`replay_buffer`**
  - `TreeReplay`：选中一个节点后，按记录顺序揭示它的下一个子节点。
  - 树版的 cohort 规则。
  - buffer 增加 `trace.jsonl` 和 `tree_cohorts`。
  - `select(..., structure="tree")`。
- **CLI**：`replay --structure tree` 和 `trace-report`。
- **测试**：新增 15 个，RSI 共 123 passed。

**真实数据**（`pool-schema3-03`，与 `pool-schema2-06` 相比 `worlds`、`attempts`、`excluded` 逐字节相同）：

| 指标 | 结果 |
| :-- | :-- |
| 树 / expansion | 24 / 356 |
| 生成节点等级 | E 34 · A 524 |
| `dispatched > 1` 的 expansion | 150，其中 135 个所有结果已知 |
| 方向级 knowledge use 中 observed 的比例 | 13%（`read` 140 / 共 1,057） |
| primary 树 train / validation / test | **3 / 3 / 3**（v1 world 为 0） |

关于 135 和第 2.6 节的 123：trace 直接测量节点的源码，而 v1 在 attempt 的 context 证据不全时把测量也记成 unknown。这 37 个 attempt 的 context 缺口作为 gap 记录，在 worklist 里可以看到。

A 级远多于 E 级，原因有两个：后续轮次和后续 launch 的 plan 都看到了兄弟或更早的结果（`update_memory` 和 `HISTORY`），而 `informed_by ⊄ ancestors`。这符合第 6 节第 3 项的决定，A 级延续允许进入 primary。

**原通过条件"replay 每条轨迹都 support_complete"不成立，已按证据修订。**

- **现象：** 用当前 KernelWiki 策略在 primary 树上 replay，train 12 条轨迹中只有 7 条 support_complete。
- **原因：** 当前策略没有命中任何规则，全程是 baseline rotation，所以 active 和 neutral 结果完全相同。全部 85 次 unsupported 请求都能解释：
  - 36 次是 root 已揭示完全部子节点后又被选中；
  - 49 次是选中了历史上从未展开过的叶子（21 次 gain>0，28 次 gain≤0）。
- **结论：** replay 的行为是对的，只揭示真实结果，偏离历史就记为 unsupported。但在贪心历史上，任何偏离主干的策略都必然遇到 unsupported；而当前规则语言（按父节点属性 prefer/defer）表达不了"继续 gain 最高的节点"。
- **修订后的通过条件：**
  - train 和 validation 的 primary > 0（已满足）；
  - replay 只揭示记录过的转移，每次偏离都被报告（已满足，85/85 可解释）；
  - active 和 neutral 产出成对轨迹（已满足）。
- **策略的晋级仍被阻塞，** 需要以下两者之一：让策略能在已记录的方向内选择（KernelWiki SDK 改动），或者收集模式（偶尔从第二名继续）。

### 第 3 步：GPU 重测剩余方向（需要授权）

- 56 个源码已归档的方向，用固定合同在 1 张 gfx950 上重测，预计约半小时。
- **通过条件：** 每个结果都与对应 kernel 的根合同一致，并保存完整的进程回执。

### 第 4 步：新数据在 lane 中直接落盘（GEAK 侧）

- 在 `kernel_lane.js` 中，把每个方向的独立测量和 verify 结论写盘，并记录 engineer 实际收到的 knowledge 内容（把 `pointed` 提升为 `read`）。
- 对应 [with_geak_design.md](with_geak_design.md) 第三节第 2 点。
- **通过条件：** 一个真实 run 产出的 Expansion 全部带测量，并且和对同一 run 做逆向的结果一致。

## 6. 已确认的决定（2026-10-29）

| 问题 | 决定 | 落实位置 |
| :-- | :-- | :-- |
| engineer 发起的 `audit_bench` 测量是否算数 | **算数**，多次测量取下中位数 | 第 5 节第 1 步（已完成） |
| 没有改动的方向是否继承父节点分数 | **继承** | 第 5 节第 1 步（由"测量属于源码"自然实现） |
| A 级延续能否参与 primary 评估 | **可以**，证据中保留 A 级标记 | 第 5 节第 2 步 |
| 是否现在用 GPU 重测剩余方向 | **先不做**，第 2 步完成后再决定 | 第 5 节第 3 步 |

## 限制

- 只覆盖 node039 和 node056-0922。node058 缺 knowledge 索引，node001 和 node056 缺源码观测。
- 5 个 plan 无法解析；15 次 integrate 调用没有重建。
- E 级是上限估计；跨 launch 的 memory 继承未验证。
- 采用 engineer 发起的测量后，测量本身仍是独立进程完成的，但"何时测、测几次"由 engineer 决定。多次测量取中位数可以限制挑选偏差，但无法完全消除。

## 复现

目录：`/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_schema3_shape_20261029/`

| 文件 | 内容 |
| :-- | :-- |
| `rebuild.py OUTPUT [POOL]` → `run-01/`（基于 `pool-schema2-04`）、`run-03/`（基于 `pool-schema2-06`） | 按 Expansion 重建，计算等级与决策支持（第 2.3–2.5、3 节；第 5 节第 1 步结果） |
| `pool-schema2-06/`、`ingest-schema2-06.log` | 第 5 节第 1 步之后重新 ingest 的 buffer |
| `pool-schema3-03/`、`pool-schema3-03-trace-report.md` | 第 5 节第 2 步之后的 buffer（含 `trace.jsonl`）和 `trace-report` 输出 |
| `replay-schema3-03/{train,validation,test}-{active,neutral}.json` | 树 replay 的 primary 轨迹；`scripts` 外的分析见本节正文 |
| `pool-schema3-01/`、`pool-schema3-02/` | 中间产物：每个 launch 一棵树的版本，以及修正 gap 记录之前的版本。只用于对照 |
| `scripts/shape.py`、`dig.py`、`greedy.py` | v1 world 形状和源码级真实树（第 2.1 节） |
| `scripts/forks.py`、`plan.py` | 5 个 run 的方向与 knowledge 统计（第 2.2 节） |
| `scripts/extra.py` | knowledge 对照与早停支持（第 3 节） |
| `scripts/remeasure.py`、`failed.py`、`requalify.py` | 结果来源核对与各补齐方式下的覆盖（第 2.6、3 节）；`requalify.py` 需要 KernelWiki 在 `PYTHONPATH` 中 |

所有脚本只读输入；`rebuild.py` 只写入一个尚不存在的新输出目录。

## 决策记录

- **需求和证据**：判断历史 GLM 数据能否作为 discovery tree 用于 replay，并找出 buffer 缩水的原因。证据是对 buffer 和原始索引的只读统计，以及对 `archive.py`、`discovery.py`、`harness.py`、`simulator.py`、`cohorts.py` 的代码核对。
- **删除**：不再把"贪心"当作缩水原因。`parent_attempt_id` 和逐节点 context 相等这两条，按 discovery-tree-design 第 5 节删除。
- **保留**：replay 只揭示真实结果、测量必须通过 `measurement()` 全部检查、E/A/T 分级。
- **最简机制**：先修测量采信规则（只改 ingest，不需要 GPU），再换 Expansion 结构。第 1 步不引入新的 hash 校验，复用 `measurement()` 已有的检查。
- **实际检查**：运行了上述只读脚本，并抽查了原始记录。
  - 第 5 节第 1 步改了 `archive.py` 及其测试：RSI 测试 108 passed；重新 ingest 后，可完整 replay 的多方向 Expansion 为 123/150。
  - 第 2 步新增 `trace/` 等：RSI 测试 123 passed；`pool-schema3-03` 的 trace-report，以及 train/validation/test 上 active 与 neutral 的 replay。
  - 没有跑 GPU。
- **当前关卡**：第 5 节第 1、2 步已通过（第 2 步的通过条件按证据修订过，见上）。策略晋级仍被阻塞，原因是在贪心历史上偏离主干就没有支撑。
- **下一步**：二选一，需要你决定：
  1. 改 KernelWiki SDK，让策略能在已记录的方向内选择；
  2. 在 lane 中加收集模式，产生非主干的展开。

  另外，GPU 重测（第 3 步）能减少 unknown，从而增加 primary 树。
