---
origin: kernelwiki-rsi
evidence_scope: kernel
key: bf16 A16W16 linear (y = x W^T), decode with very small M (M<=8 routed), memory-bound, AMD MFMA gfx950, Triton
type: lever
confidence: ★
effect: +1.6% over its parent on o-proj (one recorded step, round 29)
name: fold-k-into-mfma-n-axis-for-tiny-m-decode-gemm
description: Give each CTA only 16/FOLD output columns but keep a full 16-wide MFMA B operand by stacking FOLD contiguous K-chunks of W (and of X) along the tile's N and M axes, then sum the diagonal accumulator blocks, which raises CTA count without split-K reduction or extra weight traffic.
keywords: [k-fold, decode, skinny gemm, mfma tile fill, cta parallelism, bf16, triton, memory-bound]
kernels: [o-proj]
platforms: [gfx950]
kernel_class: skinny decode GEMM
regime: tiny-M memory-bound decode
layer: learned
lifecycle: active
levers: [output columns per CTA, FOLD factor, diagonal-block accumulator reduction, grid size, MFMA operand fill]
origin_kernels: [o-proj]
---
# Fold K into the MFMA N axis to multiply CTAs for tiny-M decode GEMM
- lever: Replace a 16-column-per-CTA tile with a 16/FOLD-column tile whose B operand is W reshaped as W'[h*BNH+j, k'] = W[n0+j, h*K/FOLD+k'], with X folded the same way, so the grid grows FOLD-fold while each W byte is still read once.
- apply: Build the A and B pointers with plain pointer arithmetic (row = j or m, offset = h*K/FOLD + k', contiguous in k'), loop K/FOLD in BLOCK_K steps with tl.dot into a (FOLD*BMH, FOLD*BNH) fp32 accumulator, zero off-diagonal blocks with tl.where(a_h == b_h), reshape to (FOLD, BMH, FOLD, BNH), sum over both h axes and store BMH x BNH; mask padded X rows (m >= M) on load and store; route only small-M buckets to it, behind a shape check that falls back loudly to the previous kernel.
- verify: Check correctness against the existing path, then run paired control/candidate timing per M bucket and keep the route only for buckets that gain beyond their spread; confirm via ISA dump that VGPR use stays modest with no spills, and check that untouched buckets did not move.
- pitfall: The off-diagonal MFMA work is wasted, so this only pays when the shape is memory-bound with tiny M; a larger slab (BMH=16, 32-row folded X tile, M12) lost in every screened arm. Requires K divisible by FOLD*BLOCK_K, N divisible by 16/FOLD and unit k-stride. Triton inserts an LDS layout round-trip in the small epilogue.
- source: recorded direction `node039/o-proj/20260924T030927_fe4aa4422e054690b4f4379ab2ae32f0/0` (o-proj, round 29, algorithm): +1.6% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/c90268d1897e2245c2b683028d47f01bba9645b9fe4e1ec97c95d33f3dc38281__8bf2acc78743005871798dceb3c37882e7f21d35878b494c3e254937a01411b7.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-05 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
