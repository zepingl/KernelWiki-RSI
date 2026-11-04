---
origin: kernelwiki-rsi
evidence_scope: kernel
key: fp8 e4m3 QK-style dot products with reduction dim a multiple of 128 (e.g. HiddenDim=128) on gfx950 / CDNA4, Gluon kernels, decode indexer logits
type: lever
confidence: ★
effect: +12.4% over its parent on dsa-decode (one recorded step, round 11)
name: cdna4-scaled-mfma-unit-scales-fp8-k128
description: On gfx950, issue one v_mfma_scale_f32_16x16x128_f8f6f4 at unit scales via gl.amd.cdna4.mfma_scaled instead of four v_mfma_f32_16x16x32_fp8 per 128-wide reduction.
keywords: [gfx950, cdna4, mfma_scaled, fp8 e4m3, 16x16x128, k_width, gluon, mfma layout]
kernels: [dsa-decode]
platforms: [gfx950]
kernel_class: fp8 MFMA dot-product / attention indexer
regime: gfx950 fp8 decode, compute-issue bound inner loop
layer: learned
lifecycle: active
levers: [wider-K MFMA instruction, scaled MFMA with None scales, operand k_width 32, arch-gated MFMA helper]
origin_kernels: [dsa-decode]
---
# Use CDNA4 scaled MFMA with unit scales for fp8 dots with K=128
- lever: Replace the CDNA3 fp8 MFMA (16x16x32, issued four times per K=128) with the CDNA4 scaled MFMA 16x16x128 f8f6f4, passing None scales so it acts as a plain fp8 dot.
- apply: Add a constexpr-gated helper: if CDNA_VERSION==4, return gl.amd.cdna4.mfma_scaled(a, None, 'e4m3', b, None, 'e4m3', acc); otherwise call gl.amd.cdna3.mfma. Build the MFMA parent as AMDMFMALayout(version=4, instr_shape=[16,16,128], warps_per_cta, tiles_per_warp) and set k_width=32 on both DotOperandLayouts for CDNA4. Route every MFMA call site, including fused extra rows and tail paths, through the helper. Keep the per-tensor fp8 dequant scales applied outside the MFMA as before.
- verify: Dump the ISA and confirm v_mfma_scale_f32_16x16x128_f8f6f4 appears and the v_mfma_f32_16x16x32_fp8 count drops. Check VGPR count and occupancy. Run correctness/parity checks. Then A/B against the seed in the same time slot with interleaved runs.
- pitfall: This only applies on gfx950, and the reduction dim must be a multiple of 128. A mismatched k_width or layout can insert costly convert_layout shuffles or break compilation. Real microscaling factors must not be passed as unit scales by mistake. The new layout can raise register pressure and lower occupancy. A missed call site such as a tail or fused-row path leaves mixed layouts.
- source: recorded direction `node039/dsa-decode/20260923T205931_330e6e2291b74946b03b0ccea26904eb/0` (dsa-decode, round 11, compute): +12.4% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/f68e94bfef833f77f0b335aeab8157560652f0fdd4f7b7ac1f04a48af55a71e3__60c823cbded91aae40a8852718b908f7e54420f8a1e697ebc97295516e4e120a.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-04 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
