---
title: sparse_attention_nsa — tuning
kind: technique
operator: sparse_attention_nsa
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/native-sparse-attention
  - https://arxiv.org/html/2508.18224v1
  - https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
---
# sparse_attention_nsa — tuning

## The three cost centers (tune in this order)
1. **Indexer / top-k selection** — scoring KV blocks and picking top-k. On DeepSeek sparse MLA this is
   the **fp8 MQA logits** kernel (`pa_mqa_logits` / `fp8_mqa_logits`); it dominates because it touches
   all (or compressed) KV. Online top-k (FLA) avoids materializing the score matrix.
2. **Gather + ragged attention** — the selected branch reads a **variable, non-contiguous** set of KV
   blocks per query. Flash Sparse Attention (FSA) shows the key wins: an **early-return** on empty index
   tensors, **decoupling** online-softmax statistics into a pre-compute kernel, and **decoupling**
   attention accumulation into a separate kernel — to avoid loop-order stalls from non-contiguous query
   access.
3. **The matmul** — once gathered, it's a normal FA block; least of your worries. (On gfx950 fp8 paged
   MQA logits decode, the matrix-core instruction choice did matter — see below.)

## AMD Triton knobs (selected/compressed/sliding branches — `triton_amd`)
- `BLOCK_M / BLOCK_N` aligned to the NSA `block_size` (64). Keep `BLOCK_N == block_size` so a selected
  block is one KV tile (clean gather). Qualifier: in the gfx950 dsa-sparse-mla split-K kernel, walking keys
  in 32-row tiles instead of 64 brought the kernel to occupancy 2 (204 VGPR vs 254 + 16 at occupancy 1).
  That alone was slower; the report attributes this to a grid of about one workgroup per CU. It helped only
  together with a larger split count so that 2 workgroups per CU were filled. bs1 kept the 64-row launch,
  because 32-row tiles hurt it.
- `matrix_instr_nonkdim=16`, `num_warps=4` (avoid 8 spill), `num_stages=1` (fused FA body).
- `waves_per_eu=2–3`, `schedule_hint=attention`.
- `knobs.amd.use_buffer_ops=ON` — selected-block gather is a masked/indexed load; bounds-checked
  `buffer_load` matters.
- fp8 indexer: **fnuz** (`fp8e4b8`/`fp8e5b16`) on gfx942.

## Capture-safe metadata (the production lever on MI300X)
The DeepSeek-V4 bring-up's biggest single win was making the sparse-MLA-decode metadata **static,
capture-safe tensors** — no dynamic ragged allocations or host→device scalar writes under HIP-graph
capture. Rebuilding ragged index tensors each step (host sync) was the bottleneck; static buffers gave
**+8.6%** (2485→2699 tok/s/GPU). Tune the bookkeeping, not just the kernel. The recorded gfx950 dsa-decode
wins kept to this rule. SplitKV stayed context-independent and shape choices were host ints frozen per
captured graph; graph replay passed with these choices.

## gfx942 dispatch routing
On CDNA3, route to aiter where a tuned path exists and **refuse dispatch (fall to Triton)** where the
AITER path is broken on gfx942 (AITER prefill MQA logits / sparse prefill logits were broken on gfx942).
The missing-entirely paths (paged MQA logits, sparse MLA prefill/decode) needed a ROCm helper →
Triton fallback. Generic Triton is "several × slower than a tuned kernel" — so verify which path runs.

## Autotune sketch
Sweep per branch keyed on `(seq, block_size, top_k, head_dim)`. Selected branch: `(BLOCK_M)∈{64,128}`,
`num_warps∈{4,8}`, `waves_per_eu∈{2,3}`. Indexer: tune the reduction tile + fp8 scale. Re-tune per
`top_k` and per context length.

Qualifiers for the gfx950 Gluon DSA indexer:
- **Wider row tiles in dsa-prefill:** raising the split-arm BLOCK_M from 4 to 8 kept VGPR at 256 and
  occupancy at 2 waves/SIMD, so the occupancy gate passed. Spills still rose from 19 to 351 and scratch
  from 80 B to 952 B. The kernel was bit-exact and about −59.4% on T. Gate on spills and scratch, not only
  on occupancy.
- **Head-reduction chains in dsa-prefill:** NUM_CHAINS 2→4 on the split arm returned max_rel_err=0.0 and
  priced NULL over 3 invocations, which suggests the reassociation did not reach the ISA. Single-process
  probes at a stacked-operand geometry also showed 4 no better than 2 and 1 worse.
- **Split-KV grid in dsa-decode:** the split-KV grid should be swept as a launch-shape knob.
  `_SPLIT_CTA_PER_CU` 6→3 was the kept setting; 5, 4 and 2 were weaker. Fractional 2.5 and 3.5 were
  associated with a large regression of unknown cause.

## What recorded gfx950 runs showed (dsa-decode / dsa-prefill / dsa-sparse-mla)
- **Read the compiled ISA before trusting a profile summary.** In dsa-decode, the seed compiled to 120 VGPRs
  at 2 waves per SIMD, not the 88 VGPRs and 5 waves that a profiling summary implied. The direction built on
  the summary's premise did not hold.
- **Make row fusion shape-selective.** In dsa-decode, fusing 3 draft rows per CTA on every bucket lost at
  short contexts (104k, 173k) and was killed. Using 3-row only when max_model_len ≥ 200000 kept 768 CTAs and
  beat the certified best, with no bucket regressing. The selection is a host int, so it is frozen per graph.
- **Use the gfx950 scaled MFMA for fp8 logits.** In the dsa-decode preshuffle kernel, replacing four
  `v_mfma_f32_16x16x32_fp8` per 16x16 tile with one `v_mfma_scale_f32_16x16x128_f8f6f4` coincided with a gain
  on all six buckets. That instruction was called with unit scales via `gl.amd.cdna4.mfma_scaled`. Matrix
  instructions in the 3-row loop went from 96 to 24, and occupancy stayed 3. A plain `cdna3.mfma` with a
  128-wide K did not compile.
- **Expect little from epilogue VALU cuts here.** Packed fp32 FMA and permlane swaps were already present in
  the seed. A joint 2-row reduce made the ISA worse.
- **Stacking query rows into one MFMA A operand did not pay in dsa-prefill.** A stacked [64,128] A operand on
  the split arm had no spills but measured slower on the split buckets.

## Verify the sparsity is real
Prefill time should scale with `top_k·block_size`, not `seq`. If it scales with `seq`, the selected
branch is reading all KV (gather/skip broken) or the indexer is the bottleneck — profile both.

## Sources
- FLA NSA (online top-k, block_size=64, fused selected+sliding): https://github.com/fla-org/native-sparse-attention
- FSA kernel optimizations (early-return, decoupled softmax/accumulate): https://arxiv.org/html/2508.18224v1
- MI300X capture-safe metadata +8.6%, gfx942 routing: https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
- aiter sparse MLA / mqa logits: `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/{unified_attention_sparse_mla,pa_mqa_logits,fp8_mqa_logits}.py` (on-box).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/sparse_attention_nsa/tuning.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.

<!-- kernelwiki-rsi:evidence -->
## Evidence (KernelWiki-RSI)

Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. Recorded GEAK runs; associations, not causes; not validated online.

- 48 recorded directions used this page; 26% beat their parent by more than 0.5%; 40% were continued by later rounds.
- 23 paired contrasts with sibling directions that did not use it: mean -0.4%, range -16.1% to +3.9%.
- Cited directions:
  - `node039/dsa-prefill/20260924T020511_be9d88523d4e4759a817fca9b3cc9ffb/1`: dsa-prefill, round 17, compute; -59.6% over its parent
  - `node039/dsa-prefill/20260923T195214_9c017fd339f5445e8a6f7adb0463be0b/0`: dsa-prefill, round 7, algorithm; -12.7% over its parent
  - `node056-0922/dsa-prefill/20260923T072549_ecb76fe2bba24d7a9ec727acea5ba205/2`: dsa-prefill, round 9, algorithm; -1.1% over its parent
  - `node039/dsa-decode/20260923T181405_8765d480a1e34c2dab51c065cd3221a2/1`: dsa-decode, round 2, algorithm; -1.0% over its parent
  - `node039/dsa-decode/20260923T173936_c7d991dd2e6f46c89084706ec690ee87/0`: dsa-decode, round 1, host_runtime; +4.1% over its parent; continued
  - `node039/dsa-decode/20260923T195104_e4fdc52df3e749f4ad909f2dd7bc3aac/0`: dsa-decode, round 7, algorithm; +4.5% over its parent
  - `node039/dsa-sparse-mla/20260923T182107_aeb48f69ed7040108ed4d1df87ce52fd/0`: dsa-sparse-mla, round 3, compute; +7.7% over its parent; continued
  - `node039/dsa-decode/20260923T204322_6f6be2fda6244585a7c9eff4adacc6fc/0`: dsa-decode, round 10, compute; +12.4% over its parent
<!-- /kernelwiki-rsi:evidence -->
