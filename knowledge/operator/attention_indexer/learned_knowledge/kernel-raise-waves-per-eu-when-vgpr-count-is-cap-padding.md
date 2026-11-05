---
origin: kernelwiki-rsi
evidence_scope: kernel
key: attention_indexer / paged MQA logits decode, fp8, gfx950 (CDNA), Triton/Gluon kernel with a fixed CTAs-per-CU split grid
type: lever
confidence: ★
effect: +2.5% over its parent on dsa-decode (one recorded step, round 2)
name: raise-waves-per-eu-when-vgpr-count-is-cap-padding
description: When the ISA shows an inflated next_free_vgpr but low real VGPR use and no AGPRs, the register count is only enforcing the waves_per_eu cap, so raising waves_per_eu can make the whole grid resident in one wave.
keywords: [waves_per_eu, occupancy, VGPR, next_free_vgpr, AGPR, gfx950, one-wave grid, launch config]
kernels: [dsa-decode]
platforms: [gfx950]
kernel_class: decode attention indexer (paged MQA logits)
regime: occupancy-limited decode with grid sized as CTAs-per-CU x CU count
layer: learned
lifecycle: active
levers: [waves_per_eu, occupancy, split CTA per CU]
origin_kernels: [dsa-decode]
---
# Raise waves_per_eu when VGPR count is cap padding, not real pressure
- lever: Increase the waves_per_eu compile hint so the occupancy matches the number of CTAs per CU the grid launches, letting all CTAs be resident at once.
- apply: Dump the ISA and compare real VGPR/AGPR use with next_free_vgpr. If the gap is padding that enforces the current waves_per_eu cap, raise waves_per_eu to the grid's CTAs-per-CU value and change every launch/compile default that carries it.
- verify: Recompile with a fresh TRITON_CACHE_DIR. Check that the ISA reports the higher Occupancy with no spills or scratch, then run a same-slot interleaved A/B against the seed and confirm correctness and graph replay pass.
- pitfall: Going above the occupancy needed to fit the grid in one wave added nothing here (it measured slightly lower), and if the VGPR count is real pressure, raising the cap forces spills.
- source: recorded direction `node039/dsa-decode/20260923T181405_8765d480a1e34c2dab51c065cd3221a2/0` (dsa-decode, round 2, compute): +2.5% over its parent, continued by later rounds; source diff `/shared_nfs/zepingl/geak_audit/runs/20260923T165210Z_node039_opus55_d6dc79bd/index/source_diffs/86a2ede1dbfb0d64a45bf5ab3a5ede1a3a6a7a7e3fb6e23db82eeed60288d8c1__8566425ba0dfbade8881c6feae433bbd13d4ee58c3f03f97ac9ba3f0f4964b11.diff`

## Provenance

Distilled by KernelWiki-RSI on 2026-11-05 from one recorded step that followed no KernelWiki page. Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. The effect is that step's recorded measurement on one kernel, not a new benchmark, and the card was not validated online.
