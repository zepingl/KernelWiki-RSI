---
origin: kernelwiki-rsi
evidence_scope: kernel
key: fused MoE gate-up GEMM (gemm1), mxfp4 A4W4 with inline quant, gfx950, decode with a tiny token count where the grid has fewer CTAs than CUs
type: lever
confidence: ★
effect: +1.7% over its parent on mxfp4-moe (one recorded step, round 4)
name: intra-cta-wave-k-split-fill-limited-decode-moe-gemm
description: When a decode MoE GEMM launches too few CTAs to keep loads in flight, make each CTA two 4-wave groups that each stream half of K for the same output tile, then sum the partials in LDS before the shared epilogue.
keywords: [k-split, intra-cta, wave groups, decode, occupancy, fill-limited, mxfp4, fused moe, lds reduction]
kernels: [mxfp4-moe]
platforms: [gfx950]
kernel_class: fused MoE grouped GEMM
regime: decode, fill-limited small batch
layer: learned
lifecycle: active
levers: [intra-CTA K split, larger CTA (more waves), private LDS region per wave group, LDS partial-sum reduction, per-grid-bucket dispatch gate]
origin_kernels: [mxfp4-moe]
---
# Split K across wave groups inside each CTA when decode MoE GEMM underfills the GPU
- lever: Double the threads per CTA so two wave groups each stream a disjoint half of the K tiles for the same n-block, which doubles outstanding weight loads per CU in a grid that cannot fill the GPU.
- apply: Derive the group id from the wave index (wave >> 2) and remap the wave to wave & 3. Offset every A, B and scale K index by the group's first K tile. Give each group its own LDS staging region and its own accumulator slice. After a barrier, have group 0 load and sum all slices and run the unchanged epilogue alone. Size LDS for both regions and enable the split only for listed grid-size buckets.
- verify: Run correctness on every case. A/B per batch bucket with fresh processes against the seed, and confirm the gated bucket's gemm1 kernel time drops in rocprof. Confirm the ungated decode and prefill buckets stay at parity.
- pitfall: The split hurt at a slightly larger batch (bs2), so gate it per grid bucket rather than enabling it broadly. Each group must keep more K tiles than the pipeline stage count and K tiles must divide evenly. The doubled LDS footprint and larger CTA can cut occupancy, and the epilogue must run in only one group to avoid duplicate stores.
- source: recorded direction `node039/mxfp4-moe/20260923T184138_57e5897319ce4f7b8aecd9a5323b3b49/0` (mxfp4-moe, round 4, compute): +1.7% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/b177993e47e9cf25e7b313a96610d3cdc6e74b4ab86c10241d5e99037b781cd6__53962292322b6d32fe615fbe120496b14bdc5e036c2dba504a0e9853c7711ca4.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-05 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
