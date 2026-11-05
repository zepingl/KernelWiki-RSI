---
origin: kernelwiki-rsi
evidence_scope: kernel
key: Any operator (here the DSA/MQA indexer logits kernel) applying ReLU or max(x, 0) to fp32 MFMA accumulators on AMD CDNA (gfx950) in a VALU-heavy epilogue.
type: lever
confidence: ★
effect: +3.2% over its parent on dsa-decode (one recorded step, round 3)
name: relu-on-mfma-output-as-int-max-on-f32-bits
description: Replace gl.maximum(acc, 0.0) on an MFMA result with bitcast-to-i32, integer max with 0, bitcast back, which removes the extra canonicalize v_max_f32 LLVM inserts because MFMA output is not known-canonical.
keywords: [relu, mfma, canonicalize, v_max_f32, v_max_i32, bitcast, valu, epilogue, gfx950]
kernels: [dsa-decode]
platforms: [gfx950]
kernel_class: MFMA logits/score kernel with ReLU epilogue
regime: VALU-bound epilogue, decode, fp32 accumulator
layer: learned
lifecycle: active
levers: [integer max on float bits, remove canonicalize, epilogue VALU reduction]
origin_kernels: [dsa-decode]
---
# Do ReLU on MFMA output as an integer max on the f32 bits
- lever: Compute ReLU on the MFMA fp32 accumulator as max(bitcast_i32(x), 0) then bitcast back to f32, so it lowers to one v_max_i32 instead of a canonicalizing v_max_f32 plus the max.
- apply: At every ReLU site fed by an MFMA result (prologue, main loop, tail and every fused row), write o = gl.maximum(o.to(tl.int32, bitcast=True), 0).to(tl.float32, bitcast=True) in place of gl.maximum(o, 0.0).
- verify: Dump the ISA and count per-iteration VALU and v_max_f32 vs v_max_i32 in the main loop: the canonicalize max should disappear, while MFMA count, VGPRs, occupancy and spills stay unchanged. Outputs should match bit for bit on finite data, then A/B time against the seed.
- pitfall: It differs from the float max only for NaN inputs (and -0.0 maps to +0.0), so it is unsafe if NaNs must propagate. First check the ISA, because if the input is already known-canonical there is nothing to remove. Algebraic alternatives such as sign(w)*max(x*|w|, 0) can de-pack the following multiplies and raise VALU and VGPR use.
- source: recorded direction `node039/dsa-decode/20260923T183521_7f4eb820d8b448998ebdb1d7b5ff8ffa/1` (dsa-decode, round 3, compute): +3.2% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/8566425ba0dfbade8881c6feae433bbd13d4ee58c3f03f97ac9ba3f0f4964b11__b89ef3e88c666ab702c5b1629575a4d519308dfd5187f330da85296ab8eb5921.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-05 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
