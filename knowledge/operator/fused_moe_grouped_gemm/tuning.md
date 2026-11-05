---
title: fused_moe_grouped_gemm — tuning
kind: technique
operator: fused_moe_grouped_gemm
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-06-09
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/tuned_fmoe.csv
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
  - https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
  - https://www.lmsys.org/blog/2026-05-28-mori/
  - https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
  - https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
---
# fused_moe_grouped_gemm — tuning

## What you actually tune
Three things: the **quant path** (biggest perf knob), the **per-shape DB entry** (stage-1/stage-2 kernel +
block_m + ksplit), and the **align&sort BLOCK_M** (how tightly per-expert tiles pack the 304 CUs). On
serving stacks the live lever is **aiter's `tuned_fmoe.csv`**, not hand-written kernels.

## 1. Quant path (the dominant lever)
`fused_moe` routes by `quant_type` / `q_dtype_a` / `q_dtype_w`:
| quant | path | when |
|---|---|---|
| `QuantType.No` (bf16) | bf16 asm fused MoE | no quant; baseline |
| `per_Token` / `per_Tensor` fp8 (E4M3FNUZ gfx942) | CK block-scale + asm stage-1 | the MI300X MoE workhorse, **up to 3× on DeepSeek-V3** |
| int8 | CK/asm int8 path | int8 weights |
| A4W4 (fp4) | FlyDSL when available, else CK | gfx950 (FP4 HW); **1.6× latency @ concurrency 512 (MI355X)**; gfx942 falls back |

fp8 block-scale is the headline win on MI300X (up to 3× on DeepSeek-V3); on MI355X the **FlyDSL A4W4/MXFP4**
path is the lever (1.6× latency cut @ concurrency 512), and MXFP4 GEMMs are ≈62% of Llama2-70B e2e. The MoE
**align&sort redesign cut the sort step 10×** (so don't re-tune the sort — it's already collapsed). **MoRI**
fuses cross-GPU EP dispatch/combine into the FusedMoE kernel. **A4W4/fp4 needs CDNA4** — on gfx942 it falls
back (and `FP4BMM` crashes vLLM, set `VLLM_ROCM_USE_AITER_FP4BMM=0`).

## 2. The tuned_fmoe DB
Key columns: `cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w,
q_type, use_g1u1, doweight_stage1, block_m, ksplit` → results `us1, kernelName1, err1, us2, kernelName2,
err2, us, run_1stage, tflops, bw`.
- **Capture**: run the model warm to append untuned shapes to `untuned_fmoe.csv`.
- **Tune**: aiter MoE tuner (analogous to gradlib; `--errRatio` default 0.05) races stage-1/stage-2
  kernels per shape, picks `block_m`/`ksplit`.
- **Deploy**: `AITER_CONFIG_FMOE=/abs/tuned_fmoe.csv` (`:`-mergeable). SGLang block-MoE:
  `SGLANG_ROCM_AITER_BLOCK_MOE=1`, `CK_BLOCK_GEMM=1`.
- ⚠ The key includes `cu_num` and `token` (the M-bucket) + full quant signature — a capture/live mismatch
  misses → the tuned CSV is inert (same failure mode as [dense_gemm](../dense_gemm/README.md)).
- ⚠ **`token` is a padded tier, not the real M.**
  - Where it comes from: `get_2stage_cfgs` keys on `get_padded_M(M) = nextPow2(M)`.
  - Effect in recorded GEAK runs (mxfp4-moe, gfx950, decode): bs1/bs2 land on tiers 8/16 and bs3–bs7 on tiers 32/64.
  - Consequence: bs3, bs4 and bs5 all share tier 32, so a CSV/tier-keyed route cannot treat them differently. Splitting them needs the real token count passed into the config lookup.
  - Rows on neighbouring tiers can carry the same `kernelName1`, `block_m` and `ksplit` and differ only in `kernelName2`.
- ⚠ **`kernelName2` is a dispatch descriptor, not just a label.**
  - On the FlyDSL MXFP4 path, `parse_g2_kname_any` derives a `v2` flag from the name string through an anchored regex (`_FLYDSL_V2_GEMM2_RE`). That flag decides which stage-2 kernel family runs.
  - Appending any suffix to a `flydsl_moe2_layout_*` name makes the match fail and silently re-routes to a different kernel family.
  - In the same runs, a guarded host-side substitution of the tier 32/64 `kernelName2` was associated with a large decode gain. The substituted name was the one the CSV already used for tiers 8/16; no kernel body changed and no new module compiled.
  - Lesson: when a decode bucket range looks like a performance cliff, diff the CSV rows on either side of the padded-tier boundary before suspecting LDS or the kernel itself.

## 3. Tile / grid knobs (per-shape, in the DB)
- **`block_m`**: the grouped-GEMM M-tile = the align&sort BLOCK_M. Bigger reduces launch overhead but wastes
  more padding when `m_e < block_m` (decode). Tune per regime.
- **`ksplit`**: K-split for skinny per-expert GEMMs (decode) to reach ≥1024 tiles across 304 CUs.
  - ⚠ The benefit was bucket-specific, not monotonic in fill. Recorded on mxfp4-moe (gfx950, FlyDSL gemm1 intra-CTA wave K-split, ks=2 via a table keyed on the `ctas_256` bucket id, decode):
    - bs1 (108 CTAs at BN=256): associated with a gain.
    - bs4–bs6: small, repeatable gains.
    - bs2 (216 CTAs): measured slower, both on the default route and on the identity route.
    - bs3: no setting beat the current one.
  - Measure each bucket and keep a per-bucket table rather than a global fill rule.
- **gemm1 n-block width (BN)**: in the same runs, BN is per-bucket too.
  - BN=128 at decode bs3–bs7 was associated with gains at bs4/bs5; bs3 stayed flat.
  - Routing bs3–bs7 back to BN=256 lost at bs4/bs5.
  - BN=64 at bs1 was slower.
  - Treat the decode BN axis there as bracketed on both sides.
  - A host-side switch keyed on the BN=256 CTA count (`ctas_256`) did not reach prefill: prefill starts at 2660 CTAs.
- **`use_g1u1`**: fuse gate+up into one stage-1 GEMM (almost always on).
- **`doweight_stage1`**: where the routed weight multiply lands (stage-1 vs stage-2 epilogue).
- **CU fill**: per-expert tiles total should give `≈ k·304` blocks. Decode (few tokens/expert) under-fills
  → consider a **masked grouped GEMM** (DeepGEMM-style) that skips empty experts instead of padding.
  - ⚠ Fewer, fatter CTAs did not pay off in a recorded mxfp4-moe gemm2 decode attempt (gfx950, bm16 non-temporal atomic port, K = 256 so one K pass per CTA).
    - Packing 2 or 4 n-tiles of the same expert into one CTA was slower at bs1, which ran half the CTAs.
    - A kernel-level gain at bs2 disappeared end to end.
    - The engineer associated this with lost CTA-level parallelism and serialized per-tile epilogues/atomics.
- **gemm2 kernel choice at decode (mxfp4-moe, gfx950)**: swapping the incumbent non-temporal-load gemm2 port lost at every changed tier. The replacements were:
  - the same port without NT loads;
  - the baseline's `bhoist_spart4x2` v2 kernel;
  - v2 256x256 or 128x128 tiles.
  
  A rocprofv3 trace associated the gap with the NT loads. Keep the cache-hint choice fixed when A/B-ing gemm2 routes.

## 4. Triton fused-MoE tuning (the editable path — and the memory-free win for int4)
vLLM/SGLang ship **per-shape JSON** configs (`E=…,N=…,device_name=MI300X.json`) generated by a
`tuning_*.py` sweep, loaded at startup (no runtime autotune). Knobs: `BLOCK_M/N/K`, `GROUP_SIZE_M`
(multiple of XCD=8), `num_warps` (wave64, 4 — not 8), `num_stages` (1–2), `matrix_instr_nonkdim=16`,
`waves_per_eu`. `SGLANG_MOE_CONFIG_DIR` / `SGLANG_MOE_PADDING` to point at / pad tuned configs.

> **vLLM int4_w4a16 — the highest-ROI, ZERO-HBM lever (don't skip it).** vLLM ships **no** tuned
> config for an unseen `E,N,int4_w4a16` shape → the int4 expert GEMM runs on a slow default fallback
> (the "Using default MoE config" warning). A per-M-bucket micro-sweep → `VLLM_TUNED_CONFIG_FOLDER`
> (vLLM's env, NOT the sglang one above) gives **~1.5–1.6x** on the prefill-dominant buckets and only
> changes Triton tile/scheduling, so it adds **no memory**. Proven: +16.4% e2e on Kimi-K2.6 (TP=8,
> int4) with `--max-num-batched-tokens 16384`. Recipe + driver: `e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md`.
>
> ⚠️ **Memory caveat for the quant path (§1) at e2e.** The fp8/CK block-scale path is the dominant
> *latency* lever in isolation, but on an already-int4 model it caches a **second (fp8) weight copy** →
> at memory parity it **OOMs at KV-cache init** (op-level 1.5x, e2e-undeployable). Before deploying any
> quant rewrite, check the extra weight bytes fit at the run's `gpu_memory_utilization` without shrinking
> the measured KV-cache budget. For int4 models, take the zero-HBM Triton config tune above FIRST.

## 5. CDNA3 vs CDNA4 deltas
- gfx942: fp8 = **fnuz**; LDS 64 KB; no FP4 HW (A4W4 falls back).
- gfx950: OCP fp8 + **FP4/MXFP4 block-scaled MFMA**; LDS 160 KB (bigger tiles); the A4W4 fast path lives
  here. The fp4 MoE GEMM is the CDNA4 headline.

## How to verify a tuning win
- `AITER_LOG_MORE=1`: confirm `fmoe_stage1_*` + `moe_ck2stages_*` fire (not a Triton MoE fallback).
- Isolated per-shape grouped-GEMM timing before/after the tuned CSV; then e2e MoE tok/s (A/B,
  non-overlapping, engagement proven).
- Quant paths: greedy parity + a small eval (the DB `err` is kernel tolerance, not task accuracy).
- **Prove the route resolves before timing.**
  - Check on the host that every bucket resolves to the intended kernel name, BN, ksplit and CTA count.
  - Confirm with a rocprofv3 trace that the new kernel is the one dispatched.
- **Use untouched buckets as a null control.**
  - In the recorded mxfp4-moe runs, buckets whose dispatch did not change still moved between process pairs.
  - Reading the changed buckets against that offset exposed one real per-bucket regression that an aggregate null had masked.
- **Measure in the right harness.**
  - Use alternating same-slot candidate/control runs in fresh processes.
  - Gate on the full harness, not a seam microbench: in one run BN=256 at bs7 won the microbench but lost in the UT.

## Sources
- DB schema, kernel names, deploy env: `ROCm/aiter@a6bb49937:aiter/configs/tuned_fmoe.csv`, `aiter/fused_moe.py`, `aiter/jit/core.py`.
- up to 3× fused MoE / block-scale fp8 (DeepSeek-V3, MI300X): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- 10× align & sort (sort step): https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- FlyDSL A4W4 1.6× latency @ concurrency 512 (MI355X) + MoRI in-kernel EP fusion: https://www.lmsys.org/blog/2026-05-28-mori/
- Kimi-K2.5 FlyDSL FusedMoE (−65% TTFT / −69% TPOT / +162% tput): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- MXFP4 GEMMs ≈62% of Llama2-70B e2e: https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
- masked/contiguous grouped GEMM concept: https://github.com/deepseek-ai/DeepGEMM
- Triton per-shape JSON tuning: https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/fused_moe_grouped_gemm/tuning.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.

<!-- kernelwiki-rsi:evidence -->
## Evidence (KernelWiki-RSI)

Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. Recorded GEAK runs; associations, not causes; not validated online.

- 32 recorded directions used this page; 43% beat their parent by more than 0.5%; 34% were continued by later rounds.
- 12 paired contrasts with sibling directions that did not use it: mean -1.2%, range -16.0% to +1.7%.
- Cited directions:
  - `node056-0922/mxfp4-moe/20260922T190015_21fb7a1dd62244acaeb65b3cc43ef98a/0`: mxfp4-moe, round 2, algorithm; +11.4% over its parent
  - `node039/mxfp4-moe/20260923T173539_2aa5f9a6dd0643e79f1e7252fb247f30/0`: mxfp4-moe, round 1, memory; -17.0% over its parent
  - `node056-0922/mxfp4-moe/20260922T202731_28cb2f43de7d41588702341730a85df8/20260923T061338_dfffd99ee8c94ab6817f45b3f366947a/1`: mxfp4-moe, round 6, host_runtime; -2.1% over its parent
  - `node056-0922/mxfp4-moe/20260922T190015_21fb7a1dd62244acaeb65b3cc43ef98a/1`: mxfp4-moe, round 2, compute; +1.1% over its parent
  - `node039/mxfp4-moe/20260923T181344_115cc2d4991141a5b17ecd1a1557a3ec/0`: mxfp4-moe, round 3, compute; +1.3% over its parent
  - `node039/mxfp4-moe/20260924T072508_4dee5cd258f04506ad87bd79f27192b8/20260924T083320_aec5056e554247f6a3fa90c0ba6b89aa/1`: mxfp4-moe, round 2, compute; -1.5% over its parent; continued
  - `node039/mxfp4-moe/20260924T072508_4dee5cd258f04506ad87bd79f27192b8/20260924T073754_4fcccda71c894489a489b23726a7932a/0`: mxfp4-moe, round 1, compute; +1.8% over its parent; continued
  - `node039/mxfp4-moe/20260923T191237_e75a7810cd7c4d05bae5220ce60fd104/1`: mxfp4-moe, round 5, memory; -1.9% over its parent
<!-- /kernelwiki-rsi:evidence -->
