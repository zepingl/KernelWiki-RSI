---
origin: kernelwiki-rsi
evidence_scope: kernel
key: fused MoE (sort + gemm1 + gemm2), mxfp4 a4w4, gfx950, graph-captured small-M decode where M*topk <= num_experts and block_m covers one pair
type: lever
confidence: ★
effect: +1.9% over its parent on mxfp4-moe (one recorded step, round 20)
name: identity-pair-routing-drops-moe-sort-node-small-m-decode
description: At small decode M, give each routed (token, expert) pair its own m-block using host-built routing metadata cached per M; this deletes the routing sort kernel, and gemm1's prologue takes over the output zero-fill and the top-k weight scatter.
keywords: [moe sort, identity routing, decode, graph node count, zero-fill fusion, small M]
kernels: [mxfp4-moe]
platforms: [gfx950]
kernel_class: fused MoE grouped GEMM pipeline
regime: graph-captured small-M decode
layer: learned
lifecycle: active
levers: [delete routing sort kernel, host-cached routing metadata, fuse zero-fill into gemm1 prologue, fuse weight scatter into gemm1 prologue, gate on M]
origin_kernels: [mxfp4-moe]
---
# Replace the MoE sort node with cached identity pair routing at small decode M
- lever: Remove the sort/align kernel from the decode graph with identity routing (m-block p holds pair p; expert ids are topk_ids flattened), and fold the sort kernel's moe_buf zero-fill and sorted_weights scatter into the gemm1 prologue.
- apply: Build sorted_token_ids, m_indices and num_valid_ids on the host for each gated M, using the same padding value the sort kernel uses. Cache them per (M, topk, BM, max_sorted, device), never build them during stream capture, and fall back to the sort path when they are not yet cached. Pass topk_ids.view(-1) as the expert ids only when the ids are int32 and contiguous. Add a compile flag with a distinct kernel name suffix that makes gemm1 threads zero-fill the output buffer in 16-byte chunks and copy each pair's weight into its sorted_weights row. Keep the original sort path for all other M values and for prefill.
- verify: Count the nodes of the captured decode graph (for example with hipGraphGetNodes) and confirm one fewer node on the gated M values. Check output parity against the baseline, expecting a relative norm slightly higher than the sort path's rather than an identical band: on the one kernel tested the identity path read 0.0039-0.0044 against 0.0022-0.0040 for the sort path and still passed every parity check. Assert on the host that grid*threads covers both the zero-fill and the pair count. Confirm with interleaved A/B runs against a control that the gated decode cases improve and no other case regresses.
- pitfall: Identity routing gives every pair its own block, so it can lose to sorted grouping as M grows. The run measured this only at M=18, where the identity path was slower (64.4 us vs 58.4 us in a warm-cache graph microbench), and the engineer attributed it to the shared expert being re-read by each token; gate the lever to the smallest M only. Padding values must match what the downstream kernels expect. Uninitialised padding rows are safe only if the epilogue never writes rows with token_id >= M. gemm2 must stay stream-ordered after gemm1 so the zero-fill lands before the atomic accumulation.
- source: recorded direction `node039/mxfp4-moe/20260923T213721_2c6613ed2b534443b74d7e33a7113dc2/20260924T053219_8e83ffe88ee147d9a5e1a24312f3526d/0` (mxfp4-moe, round 20, host_runtime): +1.9% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/549c13acf5f60638706ed5a76a2a535bbd6a85f9d6f5e4f4d93ea8ad23c00a6d__fe1a9985d2f357aa8598427c0c4c8c4f38a4460f72cfa8d30aa4cd51bd2e7a09.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-05 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
