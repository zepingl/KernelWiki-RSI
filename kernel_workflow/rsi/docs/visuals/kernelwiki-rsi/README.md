# KernelWiki-RSI · Loop Atlas

**Knowledge 是探索策略，历史是梦境世界。**

[▶ 播放两轮循环](index.html) · [高清 PNG](overview.png) · [可编辑 SVG](overview.svg)

![KernelWiki-RSI：探索、逆向追溯、冻结 buffer、梦境回放与工作策略反馈](overview.png)

## 一眼读懂

| 图中位置 | 当前代码机制 | 演示重点 |
| --- | --- | --- |
| 固定 GEAK Agent | `loop/online.py`、`harness.py` | 只改探索决策；模型、生成器、评测器和参考知识不变 |
| Discovery tree | `discovery.py` | 策略只看已揭示的 observation prefix，不预知后续分数 |
| Reverse engineering | `reverse_engineering/trace.py`、`investigate.py` | 结果 → 原始证据；按 gap 搜索、更新覆盖，再有界 review |
| Frozen buffer | `replay_buffer/build.py`、`store.py` | 每轮冻结一次；保留源码、父上下文、调用和独立测量 |
| Dream / feedback | `loop/inner.py`、`outer.py` | Developer → Reviewer → train replay；validation 选择工作副本并带回下一轮 |
| 独立发布门 | `loop/qualification.py` | 最终候选另需 fresh replay、paired online 和显式 CAS release |

图的外循环是教学编排，**并未执行真实 online、reverse investigation、freeze 或 outer cycle**。
三条候选是手写策略；实际运行的是现有 Python `replay/evaluate`，不是浏览器另写的 simulator。
第二轮加入另一段构造历史，**保留 θ₂，不强行更新**；跨轮 buffer 变化，不承诺绝对 Objective 单调上升。

## 案例与边界

| 证据层级 | 可看到的结果 | 不能声称的结论 |
| --- | --- | --- |
| 实测数值 + 构造拓扑 | 预算同为 2，T₁ 的 root-relative quality 从 **1.08% → 7.78%**，差 **+6.71 pp** | 不是新 kernel、新 GPU 测量、真实策略晋级或可发布 Knowledge |
| 真实归档 schema-1 实验 | Validation Objective **−0.00931 → −0.00857**，full-budget quality 不变 | Held-out 的 o-proj 质量回退，最终 `no_change`、未发布；局部收益不等于成功自进化 |
| 当前 schema-2 历史池 | **1,498 attempts / 337 diagnostic worlds / 0 primary worlds** | 正式演化仍缺 primary 支持，图示不能解除真实门禁 |

T₁、T₂ 的分数来自归档 `node039/dsa-sparse-mla/`：

| 构造 world | 原始 world ID 后缀 | `worlds.jsonl` 行号 | 弱分支 quality | 强分支 quality |
| --- | --- | --- | --- | --- |
| T₁ | `f8dcf1b80680e86f` | 127 | 1.0771% | 7.7846% |
| T₂ | `d75526e9a709611e` | 167 | 0.6820% | 6.4301% |

**A1、A2 原本是 root 的兄弟节点。** 图中的 A1 → A2 延续、调度顺序是构造；
B0 仅复用 root 测量，不是额外实测 attempt。源码与上下文使用现有 replay 单测 fixture，
不冒充原始 kernel。构造 world 不写入正式 buffer；原始记录与 active Knowledge 均不改变。

本例固定单 worker、预算 `2`，不同于正式 GLM7 的 `1, 2, 4, 6` 预算集。
`Q = best_score / root_score − 1`；`C` 为 charged direction units；
`Objective = Q − 0.01 × C`，沿用当前目标函数。
`case.js` 保留完整 replay trace、原始 attempt / parent ID、行号和 SHA-256；页面节点可打开出处说明。

## 工件

| 文件 | 职责 |
| --- | --- |
| `index.html`、`style.css` | 单一 SVG 图稿（顺时针五站闭环：蓝 = 历史，紫 = 梦境）与响应式页面；窄屏只在图内左右滚动 |
| `player.js` | 播放、暂停、阶段切换、两轮切换、方向键和当前帧 SVG 导出；无计算策略分数的副本 |
| `case.js` | 冻结的实测来源与当前 Python replay 预计算结果 |
| `build_case.py`、`test_case.py` | 只读构造与重放；验证原始数据保留、同预算比较和策略继承 |
| `overview.svg`、`overview.png` | 第一轮反馈阶段的静态导出；PNG 为 2× 高清、无字体依赖 |

直接用浏览器打开 `index.html`，不需要服务器、网络或前端安装。
SVG 使用系统中文字体；PNG 可直接用于 README、报告与幻灯片。
播放器读冻结的 `case.js`，**不是实时实验监控**。更新案例时先重建比对，再重新导出图稿。

## 本地复现

在 GEAK 根目录执行；需要 Python ≥ 3.10、PyYAML、pytest 和配套 KernelWiki SDK。
产物输出到临时目录，不覆盖原始数据或已检查的案例快照。

```bash
export KERNELWIKI_ROOT=/home/zepingl/work/production/KernelWiki
export PYTHONPATH="$KERNELWIKI_ROOT:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RSI_POOL=/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_reconstruction_20261022/pool-schema2-04
export RSI_ARCHIVE=/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_v1_20261022
export RSI_VISUAL_OUT="$(mktemp -d /tmp/rsi-visual.XXXXXX)"

python3 docs/visuals/kernelwiki-rsi/build_case.py \
  --pool "$RSI_POOL" --archive "$RSI_ARCHIVE" > "$RSI_VISUAL_OUT/case.js"
cmp "$RSI_VISUAL_OUT/case.js" docs/visuals/kernelwiki-rsi/case.js
python3 -m pytest -q docs/visuals/kernelwiki-rsi/test_case.py
```

在页面选择 `t = 1 → 反馈`，点击 `SVG ↓`，即可重导出 overview 对应的帧。
详见[代码地图与真实边界](../../../kernel_workflow/rsi/README.md)、
[完整机制](../../kernelwiki-rsi-v1.md)和[归档实验](../../kernelwiki-rsi-v1-experiment.md)。
