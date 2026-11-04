---
title: "FlyDSL — GEMM authoring levers (tiling / LDS / MFMA-loop / epilogue)"
kind: language
gens: [gfx942, gfx950]
updated: 2026-06-01
source_commit: AMD-AGI/GEAK@c0a1f937
sources:
  - AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_gemm_optimization.md
---
> **Reference (how-to), not a verdict.** Ingested from the FlyDSL authoring skill. GEMM-specific
> follow-on to [`authoring_optimization.md`](authoring_optimization.md). For the SOTA backend decision
> for dense GEMM (which path to pick, library vs author) see the operator card
> [`../../operators/dense_gemm/backends/flydsl.md`](../../../operator/dense_gemm/backend/flydsl.md).
>
> **These levers are written for a dense GEMM, where the caller's shape fixes the tile.** They do
> not carry over unchanged to the grouped/MoE arm: per-expert M is ragged and padded to the sort
> block size, so `tile_m` is set by the sorting stage rather than by the caller, and the parity gate
> is a magnitude-weighted `logits_diff < 0.01` rather than an elementwise `allclose`. For that arm
> start from the operator card
> [`../../operators/fused_moe_grouped_gemm/backends/flydsl.md`](../../../operator/fused_moe_grouped_gemm/backend/flydsl.md).
>
> **Recorded runs (gfx950).** On the dense decode GEMM `qkv-a-proj` (bf16), the tile, wave-layout
> and K-loop levers below were associated with measured gains. On the mxfp4 grouped-GEMM
> `mxfp4-moe` at BM=16 decode, several kernel-body levers were net-flat or closed by a source read.
> The largest recorded gain on that kernel came from host-side config routing: a tuned-CSV row
> boundary selected a different stage-2 kernel family. No kernel body was touched. For grouped/MoE,
> check what the host dispatch selects before pulling a lever here.

# FlyDSL GEMM optimization

## Overview

Use this document when a FlyDSL kernel is already clearly **GEMM-like** and the
next question is no longer "what category is this kernel?" but "which GEMM
optimization lever should I pull first?".

This is a specialized follow-on to the generic [`authoring_optimization.md`](authoring_optimization.md)
workflow. Start there when the bottleneck is still broad or ambiguous. Switch
here when the optimization work centers on GEMM structure, MFMA-loop
efficiency, LDS layout, or epilogue/store strategy.

---

## When to use

Use this guide when most of the optimization discussion is about:

- `tile_m` / `tile_n` / `tile_k` selection
- MFMA instruction shape and repeat layout
- LDS ping-pong or multi-stage buffering
- XOR swizzle or other bank-conflict avoidance
- global-load / LDS-load / MFMA overlap
- `sched_mfma` / `sched_vmem` / `sched_dsrd` / `sched_dswr` tuning
- direct-store vs shuffle/reorder epilogue
- arch VGPR pressure, occupancy, or spill risk
- ATT trace or ISA evidence from the GEMM hot loop

Do **not** start here for:

- first-time kernel authoring from scratch (see [`authoring_tile_programming.md`](authoring_tile_programming.md))
- general correctness bugs (see [`debugging.md`](debugging.md))
- non-GEMM kernels where the bottleneck is still unclear
- a regression or cliff between nearby batch sizes that lines up with a host config/dispatch boundary. In the recorded `mxfp4-moe` run, such a cliff was a tuned-CSV row boundary keyed on padded M, not a kernel limit.

---

## Workflow

### 1. Confirm the kernel is really GEMM-like

Read the full device kernel and host launcher. Verify that the hot path is a
tiled matmul or fused matmul-style loop rather than a generic reduction or
copy-heavy kernel with only incidental MFMA use.

Questions to answer first:

- What are the logical `M`, `N`, and `K` dimensions?
- How are blocks and waves partitioned across output tiles?
- Which operands are read from global memory each iteration?
- Which data is reused enough to justify LDS staging?
- Is the epilogue simple direct store, or does it reorder output fragments?
- Which config or kernel variant does the host actually dispatch for each benchmarked M? The recorded runs confirmed this with a dispatch probe or a rocprof kernel name.
- How many bytes does each operand really move? On `mxfp4-moe` the A operand is read through a buffer resource sized to the real token count. Padded rows are therefore hardware out-of-bounds and issue no memory request. From the buffer construction the engineer inferred, without bandwidth counters, that the real A stream was a small fraction of the per-CTA traffic. Separately, a measured variant removed the duplicated A load and inline quant, but it also added one dispatch. That variant was net-flat against the control. No arm isolated the dispatch cost, so whether the duplication itself is small stayed unproven.

### 2. Classify the bottleneck before rewriting

Use the strongest evidence you have:

- runtime measurements: kernel time, shape sensitivity, achieved throughput
- ATT trace if available: `vmcnt`, `lgkmcnt`, `s_barrier`, `ds_*`,
  `buffer_load_*`, `v_mfma_*`
- ISA dump if available: MFMA density, barrier count, memory-instruction mix
- a static read of the hot loop when no timed discriminator is affordable. It is weaker evidence. On `mxfp4-moe` it showed that the weight stream was already software-pipelined, and the engineer did not fund pipeline depth.

Rule of thumb:

- High `s_waitcnt vmcnt(0)` before MFMA -> global-load latency exposed
- High `s_waitcnt lgkmcnt(0)` or `ds_*` stall -> LDS latency or bank conflicts
- High `s_barrier` stall -> synchronization overhead
- Low MFMA density / many bubbles -> schedule or loop-shape problem
- Good MFMA density but poor overall speed -> tile shape, occupancy, or store path

### 3. Prioritize high-impact structure first

Tune in this order:

1. **Tile strategy**
2. **LDS staging and overlap**
3. **MFMA-loop scheduling**
4. **Epilogue/store strategy**
5. **Final parameter tuning**

Do not start by micro-tweaking constants if the kernel still has obvious
pipeline, tiling, or memory-layout problems.

Do not trade a removed in-kernel cost for an added kernel dispatch without pricing both. On `mxfp4-moe` decode, A was quantized once in a separate kernel launched by the host, which added one dispatch. The result was net-flat against the control, and the engineer could not isolate the dispatch cost.

---

## Core patterns

### Tiling

Check these constraints first:

- `tile_m` is a multiple of the MFMA M dimension
- `tile_n` is large enough to keep waves busy and maps cleanly to wave/workgroup partitioning
- `tile_k * elem_bytes` aligns with the kernel's load and MFMA packing strategy
- total per-stage LDS fits the target arch budget
- threads per CTA times the output vector width divides the output tile. Otherwise the kernel's output-write asserts reject the layout.

Use larger `tile_k` when compute can hide memory latency and LDS budget allows.
Reduce `tile_k` when LDS usage, register pressure, or occupancy becomes the
real limit.

When shapes are irregular, prefer a tile choice that remains robust across the
benchmarked range instead of overfitting to one hotspot shape.

Recorded on `qkv-a-proj` (bf16 decode, gfx950):

- Going from 4 to 8 waves per CTA on the tier-32 decode kernel was associated with a gain on the weighted metric. The N-split W1x4x2 layout was adopted over the slice-K W1x2x4 layout.
- A 32x32 SPK3 tile with 512 threads was associated with a decode gain. Its LDS was 64 KB, against 96 KB for the old tile. SPK4 was slower.
- Both 512-thread layouts became legal only after the output vector width was lowered: to 4 for the 32x64 tile and to 2 for the 32x32 tile. The change was made in both the kernel and the host-side validator.
- At tile_m=48, W1x2x2 was illegal and W1x1x2 was the only legal slice-K layout.
- With that tier-48 slice-K change, dec_bs6 (M=36, padded to 48 rows) gained +0.9%. The engineer attributed the shortfall to the kernel staying latency-bound. dec_bs7 (M=42) gained +5.1%.

### LDS staging

For GEMM kernels, ask whether one of these is true:

- an operand tile is reread many times by MFMA -> stage it through LDS
- global-load latency is exposed -> prefetch earlier
- one LDS buffer is idle while compute runs -> consider ping-pong staging

If the kernel already uses LDS, inspect whether the problem is:

- **capacity**: too much LDS per workgroup
- **layout**: bank conflicts from stride or access pattern
- **timing**: `ds_write` too close to dependent `ds_read` / wait

Multi-stage depth depends on how many K iterations each split has. On `qkv-a-proj` tier-32 decode, each split had 12 K iterations, and 4 stages was the best setting tried:

- STAGES=5 and 6 were 4–6% slower than the seed, over 4 processes each. The engineer explained this as more stages leaving fewer main-loop iterations and a longer drain with no new loads.
- STAGES=3 was 6–8% slower, over 2 processes.

In the same pipeline, fully unrolling the K loop and its drain made every LDS stage index a compile-time constant. That change was associated with a weighted gain, a small VGPR increase and no spills.

### Swizzle and bank conflicts

Prefer XOR-style swizzle when:

- the access pattern is regular
- read/write transforms can stay consistent
- LDS headroom is tight

Prefer padding when:

- swizzle would overcomplicate address math
- LDS has enough headroom
- a small stride change removes the conflict cleanly

Always keep the read path and write path consistent. A swizzled store with an
unswizzled load is a correctness bug, not a performance optimization.

### Prefetch and scheduling

Prefetch helps only when there is enough independent work to hide the latency.
Good candidates:

- next-tile global loads
- address computation
- independent MFMA groups
- epilogue preparation that does not consume the not-yet-ready data

If prefetch adds substantial carried state, re-check VGPR pressure and
occupancy. Prefetch that causes spills often loses more than it wins.

Before adding prefetch depth, check whether loads are already hoisted. On `mxfp4-moe` gemm1, the weight stream was already a 2-deep register pipeline. It used 128-bit loads, and each refill was issued two k-tiles ahead under scheduling fences. A static source read therefore ruled out too few outstanding loads. The engineer's leading remaining candidate was the inline-quant dequant-to-MFMA dependency chain, which was not measured.

For scheduler hints, match them to the real loop body instead of copying fixed
numbers from another kernel. The MFMA count, LDS-read count, and VMEM count per
iteration should come from the current kernel's loop structure.

### Epilogue choice

Here, "epilogue" means the final stage that maps accumulator fragments to
output-memory stores.

Use direct store when it is already reasonably coalesced and simple.

Consider a shuffle/reorder epilogue when:

- output stores are poorly coalesced
- tile shape creates fragmented writes
- the extra LDS/barrier cost is smaller than the store inefficiency

---

## Quick reference

| Symptom | Likely issue | First action |
|---|---|---|
| High `s_waitcnt vmcnt(0)` before MFMA | global-load latency exposed | move next-tile loads earlier; revisit prefetch distance |
| High `s_waitcnt lgkmcnt(0)` / `ds_*` stall | LDS latency or bank conflicts | inspect LDS layout, swizzle, padding, write-read distance |
| High `s_barrier` stall | too many sync points | reduce stage boundaries or merge dependent phases |
| Low MFMA ratio in hot loop | schedule overhead or loop shape | count MFMA vs memory ops and simplify loop body |
| Speed good on one shape, bad on nearby shapes | brittle tile choice | re-check tile divisibility, occupancy, and edge handling; also check host config/dispatch boundaries |
| Throughput drops after adding prefetch | register pressure too high | reduce carried state or use a lighter staging strategy |
| New wave layout rejected by asserts | thread count x output vector width does not divide the output tile | lower the output vector width in both kernel and host validator |

---

## Correctness constraints

Performance changes must preserve these constraints:

- respect arch-specific LDS limits
- keep tile packing and vector widths aligned with operand layout
- verify accumulator/output type conversions do not introduce unexpected overflow
- apply any swizzle or padding consistently to both producer and consumer paths
- confirm edge masking still works for non-divisible shapes

---

## Common mistakes

- starting from scheduler constants before proving the bottleneck
- copying tile sizes from another kernel without checking work decomposition
- adding multi-stage LDS buffering that destroys occupancy
- treating every LDS issue as a swizzle problem instead of checking wait distance
- comparing only one benchmark shape and overfitting to it
- assuming a trace or ISA pattern from another repository matches the current kernel
- removing "duplicated" operand traffic without measuring how much traffic it really is. Hardware-OOB padded rows may already cost nothing (`mxfp4-moe` at BM=16).
- tuning kernel bodies when the host is routing the shape to a different kernel family than intended

---

## Verification

After each meaningful change:

1. Re-run correctness checks first
2. Re-measure the same GEMM shapes as the baseline
3. If trace or ISA evidence is available, confirm the targeted stall actually moved
4. Check that speedup is not coming from one shape while others regress
5. Confirm the new variant is the one dispatched, by probe or by kernel name in the profile
6. Weigh per-case gains by each case's weight in the headline metric. On `qkv-a-proj`, a tier-48 change sped up dec_bs7, but the weighted headline did not measurably move because those cases carry little weight.

## Sources
- AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_gemm_optimization.md — origin (ingested as reference).
- AMD CDNA3 ISA (MFMA shapes, LDS banks): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- Matrix Cores on CDNA (MFMA + LDS staging for GEMM): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- CDNA4 GEMM kernels (tiling / epilogue): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- Cross-refs: [`authoring_optimization.md`](authoring_optimization.md) · operator card [`../../operators/dense_gemm/backends/flydsl.md`](../../../operator/dense_gemm/backend/flydsl.md) · library knobs [`knobs.md`](knobs.md)

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/languages/flydsl/authoring_gemm_levers.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification. Updated from [AMD-AGI/GEAK `baf0b2e46abf`](https://github.com/AMD-AGI/GEAK/blob/baf0b2e46abfb4ad796b9c591392a03b3d1d061e/perf_knowledge/languages/flydsl/authoring_gemm_levers.md) on 2026-10-22. The dense-versus-grouped scope note below the title was written for this import and has no counterpart at that commit.

<!-- kernelwiki-rsi:evidence -->
## Evidence (KernelWiki-RSI)

Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. Recorded GEAK runs; associations, not causes; not validated online.

- 28 recorded directions used this page; 41% beat their parent by more than 0.5%; 36% were continued by later rounds.
- 16 paired contrasts with sibling directions that did not use it: mean +0.5%, range -1.1% to +6.9%.
- Cited directions:
  - `node056-0922/mxfp4-moe/20260922T190015_21fb7a1dd62244acaeb65b3cc43ef98a/0`: mxfp4-moe, round 2, algorithm; +11.4% over its parent
  - `node056-0922/mxfp4-moe/20260922T202731_28cb2f43de7d41588702341730a85df8/20260922T230509_34ca61921b5f48258dc9cab975dc7781/0`: mxfp4-moe, round 2, algorithm; -1.0% over its parent
  - `node056-0922/mxfp4-moe/20260922T202731_28cb2f43de7d41588702341730a85df8/20260923T010154_4dfdbfc07daa4e18b6b302e0deb86a73/2`: mxfp4-moe, round 3, memory; -0.4% over its parent; continued
  - `node039/qkv-a-proj/20260923T191932_e102e8cf359b4c8c9ceee6b81713d3d3/0`: qkv-a-proj, round 4, compute; +3.2% over its parent; continued
  - `node039/qkv-a-proj/20260923T212046_b3724cf93020478991f2a0da5e14b0f4/1`: qkv-a-proj, round 6, compute; +2.4% over its parent; continued
  - `node039/qkv-a-proj/20260924T001411_0f14e9580e9849d2b01d52511e1f0391/0`: qkv-a-proj, round 12, memory; +4.1% over its parent; continued
  - `node039/qkv-a-proj/20260923T182548_859c22d5a867488a8503e24c613c8d71/1`: qkv-a-proj, round 2, algorithm; -0.3% over its parent
<!-- /kernelwiki-rsi:evidence -->
