# KernelWiki-RSI v2：底层 Node 结构与 Discovery Tree 设计

> 已被第二版取代：[`discovery-tree-design-v2.md`](discovery-tree-design-v2.md)。本文保留作为第一版记录。

我先给出第一版设计，这一轮没有改代码。依据是读过的这些内容：`kernel_lane.js` 的主循环，v1 的 `discovery.py`、`reverse_engineering/model.py`，KernelWiki 的 `knowledge.py`，以及上一份 `with_geak_design.md`。

## 0. 先质疑需求：真正要的是什么

三个使用方关心的其实是同一个问题：**在某个状态下用了哪些知识、做了什么决策，产生了什么代码状态，测出来多少，最后留没留下。**

| 使用方 | 它要从树里拿什么 |
| --- | --- |
| KernelWiki 自迭代（RSI） | replay 宽度、deep 模式、阈值、停止这些决策；评估每条知识对结果的贡献 |
| GEAK 在线兼容 | 主循环按真实形状写入，不改变 pipeline 行为 |
| 逆向工程 | 从历史 log 重建出同一种树，缺的信息记成 gap，再逐个调查补上 |

所以这三者应该共用一套结构，差别只在每个字段的**证据来源**（observed / claim / missing）。不应该分成三套 schema。

## 1. GEAK 的真实形状（对着代码看）

```
setup ─► baseline (root)
          │
          ├─ WarmStart: KB resolve → verify → adopt     (:1352/:1452)  ← KernelWiki kernel_code 进入
          │
          └─ round r:
               plan(memory, kk_refs, learned_refs, skills) (:1584)     ← KernelWiki 知识进入
               dispatch directions.slice(0,K) | 仅 deep    (:1628/:1640)
               engineer × K → verify(独立测量)             (:897)
               integrate(≥2 verified, 多父)               (:1672)
               winner > cumulative·(1+min_improve) ? commit (:1699/:1730)
               update_memory(所有兄弟的结果)               (:1752)    ← 下一轮 context 被兄弟污染
               citations: learned_refs × verified × became_winner (:1798)
          ...
          stop → director validate → curate learned card  ← 知识写回 KernelWiki
```

由此得到两个结构上的要点：

- **决策的单位是"一次展开"，不是一个节点。** context、generation、plan 都是一组兄弟共用的。v1 把这些字段放在每个节点上，造成了重复，还因此必须把分叉拆成新的 root。
- **知识有进有出。** 进来的路径是 plan、engineer、warm-start 和 skill，出去的是 curate card 和 kernel_code put。`:1798` 的 citation join 已经在做"知识→结果"的归因，只是没有落到树里。

## 2. 底层结构：只有 4 个概念

```python
# ---- 引用 KernelWiki：按身份和内容 hash，不 import KernelWiki 代码 ----
@dataclass(frozen=True)
class KnowledgeUse:
    record: str            # KnowledgeRecord.id | geak:kernel_code:... | learned card path
    revision: str          # 实际投递内容的 sha256
    channel: str           # pointed(kk_refs) | read(正文被读) | cited(learned_refs)
                           # | adopted(warm-start patch) | skill
    evidence: tuple[EvidenceRef, ...]   # 沿用 EvidenceRef.kind 表示 observed/claim/review

@dataclass(frozen=True)
class ContextRef:
    revision: str                     # digest(components)
    components: dict[str, str]        # memory/profile/commandment/codebase/brief → sha256
    derived_from: tuple[str, ...]     # 哪些节点的结果写进了这份 context（主要是 memory）

# ---- 决策：从父节点、在给定 context 下、产出一组有序的子节点 ----
@dataclass(frozen=True)
class Expansion:
    id: str
    op: str                           # generate | integrate | adopt
    parents: tuple[str, ...]          # generate/adopt: 1 个；integrate: ≥2 个
    round: int
    context: ContextRef
    generation: GenerationSpec        # model / effort / workflow revision（去掉 worker_limit）
    plan: tuple[Slot, ...]            # 规划的全部方向：id、specialty、mode、knowledge
    dispatched: int                   # 实际派发的前缀长度，也就是 lane 里 slice(0, K) 的 K
    knowledge: tuple[KnowledgeUse, ...]  # 规划时用到的知识

# ---- 状态：某个 slot 实际得到的源码状态 ----
@dataclass(frozen=True)
class Node:
    id: str
    expansion: str | None             # 只有 root 为 None
    slot: int                         # 在 plan 中的序号，replay 按这个顺序揭示
    state: SourceState                # 内容寻址的源码 fingerprint（沿用 v1）
    knowledge: tuple[KnowledgeUse, ...]  # engineer 实际读到或采用的知识

# ---- 结果：单独追加，因为它晚于节点产生，也可能永远缺失 ----
@dataclass(frozen=True)
class Outcome:
    node: str
    measurement: Measurement          # 沿用 v1：contract、逐 case 计时、是否独立测量
    disposition: str                  # failed | incorrect | verified | committed
    cost: Cost                        # 方向单位、agent 调用次数、GPU 秒数
```

**DiscoveryTree** 就是对一份只追加的记录流做 fold：

```python
@dataclass(frozen=True)
class DiscoveryTree:
    task: KernelState                 # operator/backend/gfx/dtype/shape
    contract: str                     # 整棵树统一的测量合同
    params: dict                      # 历史上的 min_improve / budget / deep_cost
    root: str
    expansions: tuple[Expansion, ...] # 按时间排序，也就是策略的决策日志
    nodes: Mapping[str, Node]
    outcomes: Mapping[str, Outcome]
    end: End                          # 停止原因、最终节点、emitted 出的 KnowledgeUse
    gaps: tuple[Gap, ...]             # 直接沿用 reverse_engineering.model.Gap
```

wire 格式是一个 jsonl 文件，只有 5 种记录：`expansion | node | outcome | end | gap`。在线 tracing 和离线逆向都写这同一种格式。

## 3. 不变式（构造时检查）

1. 每个非 root 节点恰好属于一个 expansion；expansion 的 parents 必须在记录流中先出现，所以结构是 DAG，按写入顺序就是拓扑序。
2. 同一个 expansion 的兄弟**按构造就共享** context 和 generation，因为这些字段只存一份，不需要再检查。
3. 所有 qualified 的 measurement 都必须使用 `tree.contract`。
4. `committed` 的节点构成一条从 root 出发的链，也就是 incumbent 轨迹；每次 commit 的 gain 都 ≥ 当时的 `params.min_improve`。
5. 要标 observed，必须有 observed 的 EvidenceRef。这条沿用现有规则：claim 不能升级成 observation。
6. **保真等级 E/A/T 由树计算，adapter 不能写。** 判断规则如下：
   - **E**：parent 的 state 能被观测到，`context.derived_from ⊆ 祖先`，并且是独立测量。
   - **A**：父子关系是 observed，但 context 里混入了兄弟信息，或者测量是自报的。
   - **T**：其余情况。

## 4. 三个使用方怎么接

| 使用方 | 写入 / 读取 | 用到的部分 |
| --- | --- | --- |
| GEAK 在线 | 写入。在现有 `rsiBoundary` 和 `agentT` 调用点附近追加：`:1640` 写 expansion，`:897` 写 node 和 outcome，`:1672` 写 integrate，`:1730` 写 commit，`:1752` 生成新的 ContextRef，`:1452` 写 adopt，validate 阶段写 end.emitted | 全部字段 |
| 逆向工程 | 写入。每种 archive 一个 adapter，缺失字段记成 `Gap`；`investigate` 补上证据后升级字段，等级随之重新计算 | Node、Expansion、Gap |
| KernelWiki RSI | 读取。replay 按 expansion 揭示 `plan[:K]`，评估宽度、模式、阈值和停止决策；同一个 expansion 内兄弟间的 `Node.knowledge` 不同，构成**天然的知识 A/B 对照**；`end.emitted` 加上下游的 `adopted`/`cited` 构成**知识的生命周期闭环** | Expansion、Outcome、KnowledgeUse |

依赖方向：GEAK 拥有 schema；KernelWiki 只读 trace，并通过 `record` 和 `revision` 字符串被引用。这样 trace 不依赖 KernelWiki checkout，和 `with_geak_design.md` 第三节第 3 点一致。

## 5. 相对 v1 删掉的东西

| 删掉 | 原因 |
| --- | --- |
| `DiscoveryWorld` 的"一个 root 加互不分叉的链"限制，以及 `reconstruct` 里"分叉就拆成新 root" | 这正是 0 个 primary world 的根因；用 Expansion 自然表达分叉 |
| `Attempt` 和 `DiscoveryNode` 两套几乎相同的行 | 合并成 Node + Outcome；Attempt 的 `gaps` 移到 tree 级的 Gap |
| 每个节点上重复存放的 `context`、`input_state`、`input_context` | 上移到 Expansion，作为唯一来源 |
| `ContextSnapshot.parent_revision` 链 | 换成 `derived_from`，它直接回答"context 是否只依赖祖先" |
| `GenerationSpec.worker_limit ∈ {1,2}` | 宽度是策略决策，由 `Expansion.dispatched` 表示 |
| Relation 的 `body_delivered`、`direction_forwarded`、`method_adopted` | 合并进 `KnowledgeUse.channel`，只留一套知识投递词汇 |

保留不动的有：EvidenceRef（5 个字段保持 wire 兼容）、SourceState、Measurement、Gap、Coverage，以及只追加加内容寻址的约定。

兼容性：schema 升到 3。v2 的 world 是 schema 3 的特例（每个 expansion 只有 1 个子节点）；按现有规则，legacy 数据要从原始证据重建，不做原地迁移。

## 6. 需要你决定

1. **schema 放在哪里？** 我建议放在 GEAK 侧（比如 `kernel_workflow/rsi/trace/`），KernelWiki 只按 ID 被引用。如果你希望 schema 本身归 KernelWiki 所有，GEAK 跑任何 lane 都要依赖 KernelWiki。
2. **知识粒度到哪一级？** 第一版只记到 record + 内容 hash。要不要记 section 或行范围？EvidenceRef 已经有 start_line/end_line，可以直接用，只是在线埋点会更重。
3. **收集模式**（memory 只取祖先、偶尔从第二名继续）：这仍然是拿到 E 级的前提，和上一份文档的问题 1 是同一个。

## 决策记录

- **需求和证据**：做一套通用的 node/tree，以 KernelWiki 知识的使用为中心。证据是 v1 的形状限制（`discovery.py:183`、`:269`）和 lane 的真实决策点（`kernel_lane.js` 中 `:1628`、`:1672`、`:1699`、`:1752`、`:1798`）。
- **删除**：见第 5 节。
- **最简机制**：4 个概念（KnowledgeUse、Expansion、Node、Outcome），一条 jsonl 记录流，等级由树计算。
- **实际检查**：只读了代码，没有跑任何测试或数据。这个设计还没有被验证。
- **当前关卡**：第 1 步"质疑需求"，即这个形状能否装下真实数据。
- **下一步**：用现有的 1,498 条 attempt 做只读重组，按 schema 3 建树。通过条件是：给出 E/A/T 分布；给出 expansion 宽度大于 1 的覆盖率；给出 `KnowledgeUse` 中 observed 所占比例。

你对第 6 节的三个问题拍板后，我就开始做这一步。
