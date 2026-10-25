# KernelWiki-RSI

**Knowledge-as-a-Policy = Harness 的探索决策层。**
基于 [Dream-RSI](https://arxiv.org/html/2609.14858v1)，用可追溯的 discovery tree
训练结构化调度策略，再让新策略控制下一轮 GEAK 探索；生成器、模型、评测器和参考知识保持固定。

> 当前格式为 **schema 2**。机制已实现并通过本地回归，尚未取得通过资格门的真实自进化收益。
> 历史 buffer 的 **primary worlds = 0**；诊断 replay 不能代替策略晋级证据。

## 一条闭环

```text
工作策略 θ(t) → 固定 Harness → GEAK 在线探索
                                  ↓
                    源码 / 父节点上下文 / 调用 / 独立测量
                                  ↓
                    reverse trace → 按证据缺口调查
                                  ↓
                    每轮冻结一次 replay buffer B(t+1)
                                  ↓
                    Developer → Reviewer → train replay
                                  ↓
                    validation 选择 θ(t+1) → 下一轮探索
                                  ↓
                    冻结最终候选 → 新鲜配对门禁 → 显式 release
```

Reverse engineering 也可直接调查原始 GEAK run 或在线 capture，不依赖先构建 buffer。
内循环只选择工作策略；只有最终资格判定通过后，`release` 才能 CAS 发布到 KernelWiki。

## 代码地图

| 位置 | 独立职责 | 主要入口 |
| --- | --- | --- |
| [loop/](loop/) | 外循环反馈、内循环演化、在线生成与最终资格判定 | `outer.cycle`、`inner.evolve`、`qualification.release` |
| [reverse_engineering/](reverse_engineering/) | 原始追溯、逐缺口调查、证据覆盖与有界方法分析 | `trace.trace`、`investigate.investigate` |
| [replay_buffer/](replay_buffer/) | 不可变 buffer 构建、只读查询、cohort、模拟与目标函数 | `build.freeze`、`store.open_buffer`、`evaluate.evaluate` |
| [harness.py](harness.py) | 在线与 replay 共用的固定策略解释器和 rollout 状态机 | `Policy`、`Rollout` |
| [discovery.py](discovery.py)、[evidence.py](evidence.py) | 共享领域值、源码/上下文身份与原始证据引用 | `Attempt`、`World`、`EvidenceRef` |
| [runtime.py](runtime.py) | 有界进程与固定实验 Agent；角色 schema 由业务模块持有 | `RuntimeAgent` |
| [__main__.py](__main__.py)、[report.py](report.py) | CLI 组合与报告展示，不另建事实来源 | `python3 -m kernel_workflow.rsi` |

每个机制的 tests、roles、configs 随责任归属放置；不存在的目录不为对称性而补建。
依赖方向为 `loop → replay_buffer → reverse_engineering → 共享领域值`，loop 也直接使用逆向调查。
KernelWiki SDK 是独立依赖，不在 GEAK 内复制。

## 不可突破的边界

- **Replay 只揭示已有结果**：不调用 Agent/GPU，不补造分支或 continuation；缺失转移不能换取更便宜的合格分数。
- **事实与判断分离**：源码、测量、保留与 Director 决定保留原始来源；analyst review 不能改写实验事实，部分搜索不能证明“没有”。
- **Primary 支持先于候选**：分支、continuation、完整 outcome 和最大预算覆盖均须满足；缺少 train/validation 支持时，零 Agent 调用并保留 incumbent。
- **选择与发布分离**：内循环不查看 test、不发布；真实 cycle 需要单独预算授权，最终候选需要新鲜配对 replay/online 非回退门。

实验 Agent 固定为 `claude-opus-5.5 --effort high`；不做模型 fallback、工具/MCP、会话复用或 schema 重试。
逆向调查默认 `--model-calls 0`，只有显式预算才启用方法分析。

## 本地使用

需要 Python ≥ 3.10、PyYAML，以及包含 `KnowledgeClient` 的配套 KernelWiki checkout。
以下命令在 **GEAK 仓库根目录**执行；KernelWiki 使用绝对路径，不假定两个 checkout 相邻。

```bash
export KERNELWIKI_ROOT=/absolute/path/to/KernelWiki
export PYTHONPATH="$KERNELWIKI_ROOT:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export POOL=/absolute/path/to/pool-schema2
export RSI_OUT="$(mktemp -d /tmp/kernelwiki-rsi.XXXXXX)"

python3 -m kernel_workflow.rsi --help
python3 -m kernel_workflow.rsi report --pool "$POOL" --output "$RSI_OUT/coverage.md"
python3 -m kernel_workflow.rsi replay --pool "$POOL" --wiki "$KERNELWIKI_ROOT" \
  --split train --cohort diagnostic --output "$RSI_OUT/replay.json"
```

直接调查原始 run：设置其 `AUDIT_RUN`、原始 `KERNEL`/`CAMPAIGN` 和外部调用 `ATTEMPT`。
报告若要合入已有 buffer，`CAMPAIGN` 必须与该 buffer 的原始记录一致。

```bash
python3 -m kernel_workflow.rsi investigate --run "$AUDIT_RUN" \
  --kernel "$KERNEL" --campaign "$CAMPAIGN" --attempt "$ATTEMPT" \
  --max-steps 4 --model-calls 0 --output "$RSI_OUT/reverse"
```

`trace` / `investigate` 的 `--run`、`--online`、`--pool` 三选一。
输出是新增的不可变工件，不能覆盖已有结果。
`ingest` / `freeze` 构建新 buffer；`evolve` 执行内循环；`calibrate` / `cycle` / `release`
的授权与恢复流程见[完整工程说明](../../docs/kernelwiki-rsi-v1.md#commands-and-authorization)。
示例配置分别位于 [replay_buffer/configs/](replay_buffer/configs/) 和 [loop/configs/](loop/configs/)。

## 当前实验结论

| 证据 | 结论 |
| --- | --- |
| Schema-2 历史 buffer | 1,498 attempts；528 replay-ready；153 qualified outcomes；337 diagnostic worlds；0 primary worlds；3 个逆向报告 |
| 当前内循环门禁 | `insufficient_primary_support`；0 Agent calls；无策略晋级或发布 |
| 归档 schema-1 Opus 实验 | 4 次 Developer + 4 次 Reviewer；候选早停策略提高 Objective，但 held-out full-budget quality 回退，最终 `no_change` |

归档候选的 held-out Objective 从 `-0.0084097556` 提高到 `-0.0080863459`，
但 full-budget quality 从 `0.0055477347` 降至 `0.0054050484`，因此保留 incumbent。
**尚未发现通过门禁的更优 KernelWiki 策略或新 discovery tree。**
逆向恢复历史关系不等于创造新发现，也不证明知识的因果增益。

## 回归与深入阅读

运行测试还需 pytest；runtime 回归需 Node，可用 `GEAK_NODE_BIN` 指定本地路径。

```bash
export KERNEL_EXTRACT_LOCK_DIR="$(mktemp -d /tmp/rsi-test-locks.XXXXXX)"
python3 -m pytest -q kernel_workflow/rsi
"${GEAK_NODE_BIN:-node}" interface/runtime/engine/selftest.mjs
```

2026-10-25 配对回归为 **470 passed / 1 skipped**（其中 RSI 104 项），Node **108 passed / 0 failed**。
远程配对 CI 还需要已发布的匹配 KernelWiki SDK commit，通过 `KERNELWIKI_RSI_REF` 固定其 40 位 SHA。
代码提交不等于实验资格通过或 Knowledge 发布。

- [完整工程说明与重构证据](../../docs/kernelwiki-rsi-v1.md)
- [归档 Opus 实验：候选、指标与拒绝原因](../../docs/kernelwiki-rsi-v1-experiment.md)
