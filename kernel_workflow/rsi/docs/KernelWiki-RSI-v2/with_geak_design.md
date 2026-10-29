# KernelWiki-RSI v2：把 GEAK 的优化历史变成可 replay 的 discovery tree

> 时间戳：2026-10-27 08:50:26 UTC
>
> 本文是设计分析，没有改任何代码。依据是 `kernel_workflow/rsi` 全部核心代码、`kernel_lane.js` 主循环，以及两份调研：一份是 GEAK pipeline 各阶段产出的文件，一份是 tracing 代码实际在哪里。

## 结论先行

1. **当前 0 个 primary world，主因是搜索形状不匹配，不只是日志缺字段。**
   - `kernel_lane.js` 是贪心爬山：每轮从当前最优版本派出 K 个并行的兄弟方向，验证后取 winner commit（`:1628`、`:1699`），下一轮再从 winner 继续。每轮结束后，tech lead 用**所有兄弟**的结果更新 memory（`:1752`），这份 memory 就是下一轮的输入。
   - v1 的 `DiscoveryWorld` 只接受"一个 root 加几条互不分叉的链"（`discovery.py:183`）。遇到中途分叉就拆成新的 root（`:269`），而跨分支共享的 context 又会被当成一次 context 变化。
   - 结果是每一轮都被切成一个深度为 1 的 world。即使把缺失的来源记录全部补齐，按 v1 schema 也还是这样。
2. **两条数据路径应该产出同一种 transition 记录，只是保真等级不同。** 先定这个契约，再分别做在线 tracing 和离线逆向。
3. **要 upstream 的不是 KernelWiki。** KernelWiki 只提供约 100 行值类型和发布存储。真正要搬进 GEAK 的 tracing 有两部分：
   - GEAK 里已有的 `interface/runtime/engine/rsi_boundary.mjs`；
   - 没有纳入版本管理的外部工具 `/shared_nfs/zepingl/geak_audit/tools/`（`claude_logged.py`、`audit_bench.py`、`build_report.py`）。

## 一、统一契约：什么样的节点可以 replay

Replay 只需要回答一个问题：在状态 S 下做了决策 a，记录里出现了什么结果、花了多少成本。所以每个节点需要这些字段：

| 字段 | 内容 | 为什么需要 |
| --- | --- | --- |
| `parent_ids` | 父节点（integrate 节点有多个） | 定位这次 transition |
| `input_state` | 父节点源码的 fingerprint | 证明子节点确实从这个状态出发 |
| `input_context` | memory、profiling_summary、codebase_context、COMMANDMENT、knowledge refs 各自的 hash | 判断同一父节点下的几个子节点能不能互换 |
| `decision` | 本轮完整方向列表、实际派发的前缀、normal/deep、轮次 | 策略控制的正是这一项 |
| `generation` | model、effort、workflow revision | 冻结生成器 |
| `measurement` | 合同 id、逐 case 原始计时和权重、correctness、是否独立测量 | 可以重算 quality |
| `disposition` | verified、committed 或 rejected，以及原因 | 选择规则 |
| `cost` | 方向单位、agent 调用次数、GPU 时间 | 用于 objective |

prompt、diff、knowledge 投递属于审计证据，给逆向调查和 Developer 看，不是 replay 的输入。

保真等级分三档，门槛不比 v1 低：

- **E（精确）**：parent 的 state 和 context 都精确匹配，context 只依赖祖先路径，并且是同一合同下的独立测量。只有 E 级可以进入 primary。
- **A（近似）**：父子关系精确，但 context 里有共享信息（比如本轮所有兄弟一起写的 memory），或者测量是 engineer 自报的。只能做诊断和离线学习，不能用于晋级。
- **T（只能审计）**：父子关系或 context 无法确认，只用于审计和缺口统计。

## 二、Discovery tree 要改成 GEAK 的真实形状（schema 3）

- **允许任意深度分叉。** 同一组 (state, context, generation) 下的多个子节点构成一个"展开槽"，replay 时按派发顺序揭示。这就是 v1 对 root 的规则推广到任意节点。
- **integrate 是一个多父节点动作。** 只有历史上确实对同一组父节点做过 integrate，才能 replay。
- **策略的动作空间要对齐 lane 的真实决策**：每轮从哪个节点继续、宽度 K、normal 还是 deep、是否 integrate、接受阈值 `min_improve`、是否停止。v1 只有"选 root 还是 leaf"加 1–2 个 worker，这和 lane 不是同一个问题。
- **有利的一点**：lane 本来就执行 `plan.directions.slice(0, remaining)`，"规划 K 个、派发前缀"是现成语义。

按这个形状，贪心历史能评估的是每轮宽度、什么时候用 deep、早停和接受阈值，正好都是和成本相关的决策。但前提是这些子节点的 context 精确匹配，所以生产数据得到的是 A 级，能做诊断和离线学习，还不能进 primary；要拿到 E 级，需要第三节第 5 点的收集模式。

完全不能评估的是"从非最优节点回头继续"，因为历史上从来没有展开过非 winner。这只能在在线收集时刻意制造。

## 三、路径一：深入嵌入 kernel-workflow（在线 tracing）

现状有四个缺口：

- `rsiBoundary` 已经在 lane 的 5 个位置埋点，但只在 GEAK 自带的 Node runtime 下、并且传了 `rsi_capture` 时才生效。用 Claude Code 原生 Workflow 跑时会被直接跳过。
- verify 的结论不写盘，parent id 没有显式记录，memory 的版本没有记录。setup、baseline、integrate、update_memory 这几步都没有埋点。
- 外部 audit 工具会改写 prompt：追加 evidence-contract 说明、注入 `OPERATOR_NOTES.md`，还带了验收关卡。这些不能原样 upstream。
- 现有 1,498 条 attempt 全部来自这个外部格式，GEAK 自己写出的东西目前一条都没有被读取。

建议：

1. **在 GEAK 里建一个自有的 trace 模块。** Node 侧做记录，Python 侧放 schema 和读取器；schema 就是第一节的契约。源码对象按内容寻址，日志只追加。它只观察、不改 prompt：把 audit 工具里"记录"的部分搬过来，验收关卡和 prompt 改写留在外面。
2. **补齐埋点。** 每次 dispatch 都对源码做 snapshot，并记录 parent id 和 context 的 hash；另外要覆盖 verify 结论落盘、integrate、commit、update_memory 和停止决策。这些都在 `agentT` 和 `rsiBoundary` 已有的调用点附近，改动集中在 `kernel_lane.js` 一个文件里。836 条"缺少 private-workspace 源码观测"的缺口，指向的正是这里。
3. **默认开启，不依赖 KernelWiki。** 把 `EvidenceRef`、`canonical`、`digest` 和原子写入这些约 100 行搬进 GEAK，`EvidenceRef` 的 5 个字段保持 wire 兼容，并加一个兼容测试。这样同事跑普通 lane 就会自动产出 trace，不需要 KernelWiki checkout。KernelWiki 只保留 policy 和发布。
4. **运行时的限制。** 原生 Workflow 脚本不能访问文件和时钟，没法可靠地写下 hash。完整 trace 只能在 standalone runtime 下保证；原生 runtime 下只能降到 A/T 级。
5. **收集模式（显式开关，默认关闭）。** 要产生 E 级 primary 支持，有两处行为必须改：
   - memory 只取祖先路径，否则下一轮的 context 会依赖本轮所有兄弟；
   - 偶尔从第二名继续，制造更深的分叉。

## 四、路径二：从已有 log 和 artifacts 逆向

| 数据来源 | 现有读取器 | 能恢复的内容 | 恢复不了的内容 | 预期等级 |
| --- | --- | --- | --- | --- |
| `geak_audit` 归档（5 个 run，1,498 条 attempt） | `archive.py` | prompt、源码对象、两进程独立测量 | 多数 dispatch 时的 context 版本 | 少量 E，多数 A/T |
| 普通 lane 的 `EVAL_DIR`（同事最常见的情况） | 无 | 每轮兄弟、自报速度、`best_patch.diff`、canonical 工作区的 git 链（即历次 commit 构成的主干）、`director_validation.json` | verify 结论、各方向实际收到的 memory、独立测量 | 以 A 为主 |
| 原生 runtime 的 journal 和 `llm_trace` 镜像 | 无 | 每次 agent 调用的 prompt 和返回，可借此还原 context 文本 | 派生关系的边 | A，部分可到 E |
| e2e 运行 | 无 | 嵌套在里面的 lane | — | 同 lane |
| rsi online capture | `capture.py` | 全部 | — | E |

做法：

- 每种格式写一个 adapter，只负责读取，统一输出第一节的事件，并标明 observed 还是 inferred。
- 之后共用一套"重建 → tree → 冻结"流程。等级统一由重建步骤根据契约字段判定，不由 adapter 决定。
- 现有 `investigate` 的逐缺口调查保留，用来把 T 级提升到 A 或 E。

两个现实问题：

- 一个 audit run 在 NFS 上约 30 GB，只能读 index jsonl 和对象存储，不能遍历其中的 workspace。
- 逆向得到的缺口统计，本身就是路径一"该补哪些埋点"的需求清单。

## 五、推进顺序（每一步都有通过条件）

1. **定契约，只用现有 1,498 条数据重组形状。** 不新增任何数据，按 schema 3 重建主干加兄弟。
   - 通过条件：给出 E/A/T 分布，以及宽度和早停动作各自的 replay 支持覆盖率。
   - 这一步最便宜，并且直接检验"形状不匹配"这个判断。
2. **写普通 lane `EVAL_DIR` 的 adapter。** 用 2–3 位同事的真实 run 跑通。
   - 通过条件：重建出的主干和兄弟与 `tech_lead_report.md` 一致。
3. **在 lane 里补埋点，接上 GEAK 自有的记录器。**
   - 通过条件：一个真实 run 能产出 E 级 tree，并且和对同一 run 做逆向的结果对得上。
4. **去掉 KernelWiki 依赖，清理个人路径。** 需要处理的位置：
   - `runtime.py:56` 写死了个人 skill 目录；
   - `replay_buffer/configs/glm7.yaml` 写死了 `/shared_nfs/zepingl` 路径；
   - 测试里也引用了 `/shared_nfs/zepingl`。

   通过条件：在只装了 GEAK 的干净环境里，能完成 ingest、freeze、replay。
5. **加收集模式开关，跑一个小规模真实 pilot。** 之后再进入 v1 已有的内循环、外循环和资格判定。

## 需要你决定

1. **收集模式能不能改变 pipeline 的行为？** 也就是 memory 只取祖先路径、少量从第二名回溯。如果不接受，生产数据最多到 A 级，只能做诊断学习，产生不了 primary。
2. **完整 trace 是否只承诺在 standalone runtime 下提供？**
3. **要不要先做第 1 步？** 它只读现有数据、不需要新实验，可以先验证结论 1，再决定后面投入多少。

## 决策记录

- **需求和证据**：让 GEAK 范围内的其他同事和其他工作也能产出可 replay 的数据。证据是 v1 的结果：0 个 primary world，528 条可 replay 的 branch 深度全部为 1。加上 `kernel_lane.js:1628/1699/1752` 和 `discovery.py:183/269`，说明搜索形状不匹配。
- **删除**：不 upstream audit 工具对 prompt 的改写和验收关卡，tracing 不再依赖 KernelWiki。
- **保留**：v1 的证据不变式、逆向调查循环和资格门禁。
- **最简机制**：一份 transition 契约，在线和离线两条路径共用。
- **实际检查**：读代码、两份调研报告、核对行号。没有运行任何实验。
- **当前关卡**：第 1 步"质疑需求"，即判断形状是否匹配，尚未用数据验证。
- **下一步**：等你对上面三个问题拍板，再执行第 1 步的只读重组。
