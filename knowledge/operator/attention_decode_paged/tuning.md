---
title: attention_decode_paged — tuning
kind: operator_overview
operator: attention_decode_paged
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---
# attention_decode_paged — tuning

Paged decode is **memory-bandwidth + launch-latency bound**, not FLOP-bound (the query is one token).
So tuning is about (1) filling 304/256 CUs when batch is small, (2) coalescing the KV-cache reads, and
(3) cutting launch overhead. There is no big GEMM to tile.

## Pick the backend first (vLLM ROCm, Feb 2026)
Ranking: **ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN > TRITON_ATTN > ROCM_ATTN**. ROCM_AITER_FA vs legacy
ROCM_ATTN TPS (64/128 req): **MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×**; TPOT
**2.8–4.6×** faster. The single biggest decode "tune" is not landing on the ROCM_ATTN/Triton fallback.

## Lever 1 — splitKV / flash-decoding (the main lever)
With `sq=1` and small batch, a naive one-CU-per-(batch,head) grid starves the GPU. Split the KV history
into chunks across CUs, compute partial `(O, m, ℓ)` per chunk, then a **reduce kernel** combines them.
- aiter custom paged-attn: `_PARTITION_SIZE_ROCM = 256` (the per-split KV chunk);
  `num_partitions = ceil((max_seq_len + 256 - 1) / 256)`. `paged_attention_v2` is the split path,
  `paged_attention_v1` the single-pass path.
- aiter MLA decode: `num_kv_splits` is auto-picked by `get_meta_param` from batch/total-KV/heads — leave
  it `None` (auto) unless you measured a better value.
- Triton flash-decoding: `NUM_KV_SPLITS` constexpr; combine in a stage-2 reduce kernel.
- **Rule**: more splits when batch×heads is small (fill CUs); fewer when batch is already large (avoid
  reduce overhead). The auto heuristics encode this — trust them first, but measure when the split
  count is a hand-set knob (see below).
- **Recorded GEAK runs (gfx950, dsa-decode paged MQA-logits Gluon kernel, fp8, bs2/bs3, 104k–418k
  context, graph-replayed):** there, *fewer* splits were associated with a win. `_SPLIT_CTA_PER_CU` 6 → 3
  measured about +3.5% weighted with all six buckets improving. 5 and 4 each dropped one bucket, and
  non-integer 2.5/3.5 regressed heavily (cause unknown). Sweep the factor at the served shapes, two
  fresh processes each, rather than assuming more splits fill better. A later split-2 arm lost 2.75% on
  the 261k bucket, which the engineer attributed to rounding whole 128-token units per split. When
  changing the factor, check units-per-split per bucket.

## Lever 2 — KV-cache layout & coalescing
- **NHD vs HND**: pick the layout the kernel reads coalesced. vLLM's custom path reshapes to
  `[2, num_blocks, block_size*num_kv_heads*head_size]` with `x = 16/elt_size` inner split on K so each
  lane reads a 128-bit chunk. `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` reshuffles for the AITER FA path at
  concurrency ≥32.
- **block_size / page_size**: 16 common; larger pages reduce block-table indirection but coarsen memory.
  Tune `--page-size N` per model.
- Verify coalescing in ISA: want `global_load_dwordx4` / `buffer_load_dwordx4` on the KV reads.

## Lever 3 — MFMA path vs small-head path
vLLM custom paged-attn has two kernels:
- `paged_attention_ll4mi_QKV_mfma16_kernel` — MFMA-16 main path (use matrix core for the q·Kᵀ).
- `paged_attention_ll4mi_QKV_mfma4_kernel` — MFMA-4 small-head path.
Plus `paged_attention_ll4mi_reduce_kernel` for the cross-split softmax reduce. Templated on `BLOCK_SIZE`,
KV dtype, fp8 KV. Use `matrix_instr_nonkdim=16`, `waves_per_eu` to trim VGPRs (decode is memory-bound →
favor occupancy 3-4).
- **Gate body changes on VGPR/occupancy and ISA first (recorded, gfx950 dsa-decode, 256-thread CTAs):**
  - Widening row fusion `_QFUSE` 2 → 3 raised VGPR 120 → 154 and blocks/CU 4 → 3, and it measured
    below the seed.
  - Collapsing K address generation was an exact codegen no-op (VGPR 120 → 120). An offline ISA census
    had already shown the premise false.
  - A scalar guard skipping an all-pad prologue stage helped the 104k bucket but lost on longer contexts.
    The engineer attributed this to the branch splitting the prologue→loop schedule.

## Lever 4 — launch overhead (small batch)
Decode at small batch is launch-bound. Levers: HIP-graph capture (CUDA-graph equiv), persistent kernels
(`SGLANG_AITER_MLA_PERSIST=1`), and **unified attention** (one kernel for chunked-prefill+decode, avoids
two launches — see [../chunked_prefill/overview.md](../chunked_prefill/README.md)). `HSA_NO_SCRATCH_RECLAIM=1` is near-mandatory on
MI300X (stops scratch-reclaim idle gaps).
- **Folding the split-KV reduce into the main kernel via an arrival counter did not pay in the recorded
  run (gfx950 dsa-sparse-mla, 192–252 CTAs, 32→6 splits).** The handshake alone (counter +
  `scope="gpu"` release/acquire, no combine work) cost 5458–7725 ns/call against the 1790 ns dispatch it
  would delete. The engineer attributed the flat cost to per-CTA L2 writeback on release. Only that one
  handshake form was measured. Keep the graph-captured two-dispatch split+reduce unless a cheaper
  handshake measures below the dispatch cost.

## Head-dim / GQA specifics
- Supported head sizes (aiter custom paged-attn): **64, 80, 96, 112, 120, 128, 192, 256** — 192/256 are
  CDNA3/4-relevant. An unsupported head size falls back to Triton decode (`ROCM_ATTN` cliff: 2.7–4.4×
  slower).
- `gqa_ratio` (q-heads per kv-head) must be **1..32** for the custom path; the kernel broadcasts each KV
  head across its query-head group in-register (no KV replication in HBM) — see
  [../gqa_mqa_attention/overview.md](../gqa_mqa_attention/README.md).
- `max_seq_len ≤ 65536` for the custom paged-attn fast path.

## fp8 KV-cache
Store K/V as fp8 (half the KV bandwidth = directly faster decode), dequant on read. FNUZ on gfx942, OCP
on gfx950. `--kv-cache-dtype fp8_e4m3` (fnuz on MI300X). This is the **highest-leverage** decode tune for
long context — but it is an accuracy gate (see [numerics.md](numerics.md)). Note: aiter MLA decode does
**not** support fp8 KV-cache in vLLM upstream.

## How to verify a tune helped
Isolated decode bench at the served `(batch, heads, kv_heads, head_dim, context, block_size, dtype)`,
median ≥3 warm reps. rocprofv3 → confirm `paged_attention_ll4mi_*` (not a Triton fallback) ran. e2e:
TPOT at served concurrency + greedy temp=0 parity. For multi-bucket workloads, compare per bucket against
a seed measured in the same slot: in the recorded runs, wins on some buckets often came with a loss on one.

## Sources
- aiter paged-attn split size 256 / v1-v2 / supported head sizes / gqa 1-32 / max_seq_len 65536: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/paged_attn.py`.
- vLLM custom paged-attn kernel names / KV layout / shuffle env: https://github.com/vllm-project/vllm/tree/main/csrc/rocm ; `backends/vllm_kernels/rocm_kernels.md`.
- HSA_NO_SCRATCH_RECLAIM / ≥1024 grid / mfma_16x16: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; `backends/sglang_kernels/overview.md`.
- Backend ranking + ROCM_AITER_FA vs ROCM_ATTN TPS (MI300X 3.82×/2.65×, MI325X 4.36×/3.12×, MI355X 3.61×/2.88×; TPOT 2.8–4.6×) (vendor, Feb 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend.

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/attention_decode_paged/tuning.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.

<!-- kernelwiki-rsi:evidence -->
## Evidence (KernelWiki-RSI)

Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. Recorded GEAK runs; associations, not causes; not validated online.

- 31 recorded directions used this page; 27% beat their parent by more than 0.5%; 35% were continued by later rounds.
- 11 paired contrasts with sibling directions that did not use it: mean -2.4%, range -15.2% to +4.1%.
- Cited directions:
  - `node039/dsa-decode/20260923T173936_c7d991dd2e6f46c89084706ec690ee87/0`: dsa-decode, round 1, host_runtime; +4.1% over its parent; continued
  - `node039/dsa-decode/20260924T074446_280e15ea5839438cb107b790e4facb5e/0`: dsa-decode, round 41, memory; +1.6% over its parent
  - `node056-0922/dsa-sparse-mla/20260922T170231_4879546d7c464a708f321f4606c8934f/0`: dsa-sparse-mla, round 1, host_runtime; -30.3% over its parent
  - `node056-0922/dsa-decode/20260922T192647_692ceef8572b4a04ab6f81d861d4f03e/20260922T220616_d160a51bd33a4170af953f621f649e21/0`: dsa-decode, round 2, algorithm; -1.1% over its parent
  - `node056-0922/dsa-decode/20260922T192647_692ceef8572b4a04ab6f81d861d4f03e/20260923T015843_3661e5b7667645378d730b01072e5c95/2`: dsa-decode, round 4, memory; -1.2% over its parent; continued
  - `node056-0922/dsa-decode/20260922T192647_692ceef8572b4a04ab6f81d861d4f03e/20260923T000053_34c9d868c95e4ad984f2031bce6db7be/0`: dsa-decode, round 3, algorithm; -1.8% over its parent
<!-- /kernelwiki-rsi:evidence -->
