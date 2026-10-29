# KernelWiki-RSI 策略 v2：完整链路实验报告

> 2026-10-29 · 实验分支：GEAK `rsi-policy-v2-exp`（worktree `/shared_nfs/zepingl/home-storage/work/GEAK-rsi-exp`），
> KernelWiki `kernelwiki-rsi/policy-v2-exp`（worktree `/home/zepingl/work/production/KernelWiki-rsi-exp`）。
> 产物：`/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/`。
> 这两个分支都没有推送，也没有改动主工作区或真实的 KernelWiki。

## 结论

1. **离线部分已经跑通**，依次是：策略语言 v2、沿历史主干回放、用真实 Agent 跑内循环、发布、投影。
2. **真实内循环的结论是 `no_change`**。这个结论本身合理：Developer 提出了一条停止规则，Reviewer 以证据太薄为由驳回了两次（train 只有 1 个 kernel 的 3 棵树）。
3. **瓶颈在训练数据。** 22 个结果未知的方向让 11 棵树进不了 primary，其中包括 train 缺的两个 kernel。这些方向的源码都已归档，用 1 张 GPU 重测大约 10 分钟。
4. **在线一致性、GPU 校准和完整循环都没有执行。** 这三步需要新代码和你的批准。

## 各环节结果

| 环节 | 是否真实执行 | 结果 |
| :-- | :-- | :-- |
| KernelWiki SDK v2（`width`、`select`、`skip`、`stop`） | 真实 | 测试 60 passed；v1 行为和序列化方式都不变 |
| GEAK 解释器、episode replay、内循环 episode 模式 | 真实 | 测试 137 passed |
| 手写策略的离线探针 | 真实 | 见下面"决策 1"和"决策 2" |
| 用真实 Agent 跑内循环（`evolve --episodes step --diagnostic`） | 真实，7 次模型调用，不用 GPU | `no_change`，是诊断性质的结论 |
| 发布到 KernelWiki | 真实执行，但目标是临时克隆 | CAS 发布成功；投影出 `knowledge:policy` 节点，状态和文件内容一致；gate 记录通过 fold 检查 |
| 在线一致性（lane 只出 plan、由策略选方向） | **没做** | 还需要写代码 |
| GPU 校准和完整循环 | **没做** | 需要批准 |

## 数据回答了哪些决策点

**决策 1：replay 用哪种规则？结论是逐步打分。**

截断规则下，就连空策略（也就是历史本身的做法）在 train 上也只有 6/12 条轨迹完整，validation 上 9/12。原因是预算比 plan 窄时，历史上被继续的那个方向会被截掉，结果任何策略都不可能合格。逐步打分规则下，空策略 12/12 完整，可以作为基线。

**决策 2：动作集合要多大？结论是全部保留，但目前只有宽度和停止有信号。**

下表是 validation 上的结果（逐步打分，每个方向成本 0.01）：

| 策略 | 目标值 | 全预算质量 |
| :-- | --: | --: |
| 空策略（派完整个 plan，等同历史） | −0.0112 | 0.0534 |
| 每轮最多派 2 个 | **+0.0003** | 0.0534 |
| 每轮最多派 1 个 | +0.0155 | 0.0566（train 上有 1 条轨迹缺数据，不合格） |
| 跳过 deep 方向 | 不变 | 不变 |
| 停滞 2 轮后停止 | 不变 | 不变 |

按知识选方向的规则在历史数据上无法检验，因为历史读取能解析到 KernelWiki 的是 0/2,475。

**决策 3：要不要把 GEAK 的知识副本原样导入 KernelWiki？结论是现在不需要。**

当前的瓶颈是结果未知的方向，不是知识能否解析。宽度和停止规则也不依赖知识能否解析。

**决策 4：KernelWiki 那套未提交的 SDK 怎么办？结论是必须提交才能真正发布。**

`publish` 和投影都依赖这套 SDK。实验分支里放的是它的一份拷贝，真实发布之前需要把它正式提交。

## 内循环的具体过程

| 机会 | Developer | Reviewer |
| :-- | :-- | :-- |
| 1 | 提出：第 2 轮起，如果 `best_gain ≤ 0.01` 就停止。离线目标值从 −0.0177 提高到约 −0.0124 | 驳回：证据陈述有误；预算 4 和预算 6 的轨迹不独立；只有 3 棵树、1 个 kernel |
| 2 | 修正证据后重新提交 | 驳回：在第 2 轮找到了反例；收益主要来自少派方向，而不是这个判断条件；0.01 的阈值恰好落在三个数据点之间 |
| 3、4 | 放弃修改策略（`evidence_update`），理由是训练数据里已经没有新信号 | — |

Developer 没有尝试宽度规则，可是探针显示宽度规则在 validation 上有改进。原因可能有两个：训练集太小，或者提示还不够引导。数据补齐之后应该重跑一次再看。

## 建议的下一步（按性价比排序）

| 步骤 | 成本 | 预期效果 |
| :-- | :-- | :-- |
| 1. GPU 重测 22 个方向（源码都已归档） | 1 张 gfx950，约 10 分钟，另加准备时间；需要批准 | primary 树最多从 9 棵增加到 20 棵，train 覆盖全部 3 个 kernel，之后可以去掉诊断开关重跑内循环 |
| 2. 重跑内循环（不再是诊断模式） | 约 9 次模型调用 | 得到可以作为晋级依据的离线候选 |
| 3. 在线一致性代码（lane 的 RSI 模式加"只出 plan"步骤） | 中等的开发量 | 在线运行也能执行 v2 决策 |
| 4. 校准，然后跑完整循环 | 校准 1 张 GPU、30 分钟；完整循环约 84 个方向、2 张 GPU；各需一次批准 | 拿到真实的配对证据 |
| 5. 正式发布 | 需要 KernelWiki SDK 先正式提交 | KernelWiki 出现新的策略版本，可以回滚 |

## 复现

```bash
export PYTHONPATH=/home/zepingl/work/production/KernelWiki-rsi-exp:/shared_nfs/zepingl/home-storage/work/GEAK-rsi-exp
E=/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029
POOL=/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_schema3_shape_20261029/pool-schema4-01
python3 $E/scripts/probe_policies.py $POOL                       # 手写策略探针
python3 -m kernel_workflow.rsi evolve --pool $POOL --wiki /home/zepingl/work/production/KernelWiki-rsi-exp \
  --output <新目录> --episodes step --diagnostic --node <node> --claude <claude>   # 内循环
python3 $E/scripts/publish_demo.py <临时克隆> <policy.json> <evidence...>       # 发布和投影
```

## 决策记录

- **需求和证据**：用户要求在独立分支上跑完整链路，再根据结果做决定。证据见上面各表，以及 `evolve-01/`、`probe_policies.txt`、`publish-demo.json`。
- **删除和保留**：截断规则只保留为可选项，不作为默认。v1 行为、v3 的不变式、`release` 的 CAS 发布都保持不变。
- **最简机制**：在现有 `Condition` 和 `PolicyRule` 上扩展字段和效果；episode replay 复用历史主干和 `Plan`；门复用 measure 和 decide。
- **实际检查**：KernelWiki 60 passed，GEAK 137 passed；真实 Agent 7 次调用；在临时克隆上完成了发布和投影。
- **当前关卡**：离线部分已通过。下一步需要你批准 22 个方向的 GPU 重测。
- **下一步**：批准后重测，然后去掉诊断开关重跑内循环。
