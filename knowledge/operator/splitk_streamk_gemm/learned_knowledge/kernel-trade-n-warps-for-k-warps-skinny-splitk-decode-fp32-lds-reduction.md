---
origin: kernelwiki-rsi
evidence_scope: kernel
key: bf16 split-K GEMM, decode with small padded M (tile_m 32, tile_n 64), gfx950, memory-bound skinny activation against a wide weight
type: lever
confidence: ★
effect: +3.8% over its parent on qkv-a-proj (one recorded step, round 1)
name: trade-n-warps-for-k-warps-skinny-splitk-decode-fp32-lds-reduction
description: For a small-M split-K decode tile, re-split the CTA's warps from all-N to half N and half K (for example W1x4x1 to W1x2x2), and keep the k-warp partials in fp32 in LDS so the epilogue rounds to bf16 only once.
keywords: [slice-K, split-K, block_k_warps, warp layout, fp32 LDS reduction, decode, bf16 atomic, parity]
kernels: [qkv-a-proj]
platforms: [gfx950]
kernel_class: skinny bf16 split-K GEMM
regime: decode small-M memory-bound
layer: learned
lifecycle: active
levers: [block_n_warps to block_k_warps re-split, fp32 C-LDS for slice-K partials, single bf16 rounding before atomic, LDS size estimator update]
origin_kernels: [qkv-a-proj]
---
# Trade N-warps for K-warps in skinny split-K decode GEMM, reducing partials in fp32 LDS
- lever: Launch the small-M split-K tile with block_k_warps above 1 (fewer n-warps, more k-warps), so each CTA has several warps streaming different K sub-ranges, and reduce their partials in fp32 in LDS before a single bf16 rounding and atomic or store.
- apply: Gate a config override on the exact tile and shape row. Make the C-LDS scratch dtype fp32 when block_k_warps is above 1. Keep the slice-K partial in fp32 when storing it, sum the k-warp partials in fp32, and truncate to bf16 only after the sum, just before the bias add or the atomic or store. Count 4 bytes per C element in every LDS-size estimator used for this case.
- verify: Check that the cache mints the new W1xNxK variant name. Run a parity soak with alternating candidate and seed runs in fresh processes, and compare per-bucket decode speedup against each bucket's noise floor. Confirm that buckets not routed to this tile stay within noise.
- pitfall: fp32 C-LDS doubles the epilogue scratch, which can exceed the pipeline LDS budget or cut occupancy if the estimators are not updated. Rounding each k-warp partial to bf16 before summing adds extra rounding error and parity noise. VGPR use and occupancy of the new warp layout were not checked by ISA dump, so the gain may not carry over to other tile rows or larger M.
- source: recorded direction `node039/qkv-a-proj/20260923T174848_ff1b01d34055458cbd455114b39ad154/0` (qkv-a-proj, round 1, compute): +3.8% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/d2a593c3d0aebc21753735fa9afd69206c6fc80aa11e82c829367f42bfffe2ad__0ac956133617ffcea9d8d361e81f97790090ca6da43b1ddf377030b845298cf1.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-05 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
