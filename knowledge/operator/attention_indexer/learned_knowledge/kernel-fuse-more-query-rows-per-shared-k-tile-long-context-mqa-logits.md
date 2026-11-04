---
origin: kernelwiki-rsi
evidence_scope: kernel
key: paged MQA logits / DSA indexer decode, FP8 K, multi-row draft queries (next_n), gfx950 CDNA MFMA, long max_model_len
type: lever
confidence: ★
effect: +4.5% over its parent on dsa-decode (one recorded step, round 9)
name: fuse-more-query-rows-per-shared-k-tile-long-context-mqa-logits
description: Raise the query-row fusion factor so each loaded K tile feeds MFMAs for more draft rows, and enable it only on long-context shapes through a host-side constexpr gate.
keywords: [QFuse, row fusion, K tile reuse, MQA logits, DSA indexer, next_n, constexpr specialization, shape gating]
kernels: [dsa-decode]
platforms: [gfx950]
kernel_class: paged MQA logits indexer decode
regime: long-context decode, memory-bound K streaming
layer: learned
lifecycle: active
levers: [query-row fusion factor, shared K tile across rows, constexpr specialization, host-side shape gate, SplitKV re-derivation]
origin_kernels: [dsa-decode]
---
# Fuse more query rows per shared K tile in long-context paged MQA logits decode
- lever: Increase the number of query rows fused per workgroup (here from 2 to 3) so one K tile load and layout conversion feeds one MFMA per fused row, which cuts K traffic per output row.
- apply: Add the fusion factor as a constexpr kernel parameter. For the extra row, load its own Q, weights and context length, and widen the shared loop bound to the max context across fused rows. Issue an extra MFMA, ReLU, weight scale and reduce per K stage, plus a store with that row's own mask and -inf tail. On the host, pick the higher factor only when the row count divides evenly and max_model_len is long, and recompute the grid and SplitKV from the fused group count.
- verify: Check correctness and parity per fused row, including tail masking. Inspect the ISA: the VGPR count should rise, occupancy should stay unchanged, and there should be no scratch or spills, with MFMA count scaling with rows. Then confirm per-bucket timings improve on long contexts and stay neutral on short ones that keep the old body.
- pitfall: Ungated, the wider body was slower on shorter contexts. The gate also keys on pool-wide max_model_len rather than each call's real context, so serving traffic may get the wrong variant. It assumes fused row groups never straddle sequences, and extra rows can push VGPRs past the occupancy budget.
- source: recorded direction `node039/dsa-decode/20260923T201521_35b8ae55a70842f6b726b5a1427752ae/0` (dsa-decode, round 9, algorithm): +4.5% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/b89ef3e88c666ab702c5b1629575a4d519308dfd5187f330da85296ab8eb5921__f68e94bfef833f77f0b335aeab8157560652f0fdd4f7b7ac1f04a48af55a71e3.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-04 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
