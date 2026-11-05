---
title: Gluon programming model (block-level, manual scheduling) — reference
kind: language
gens: [gfx942, gfx950]
dtypes: [fp16, bf16, fp8_e5m2, fp8_e4m3, fp4_e2m1]
regimes: [prefill, training, both]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
---
# Gluon programming model

## TL;DR
Gluon keeps Triton's **tile-level** (not thread-level) authoring but hands the programmer the four things
Triton's compiler normally owns: **explicit layouts**, **explicit pipeline stages**, **explicit register
budgeting**, and **direct MFMA intrinsics** (incl. CDNA4 scaled-MFMA). You write the wave schedule
(ping-pong / interleave) yourself instead of getting `num_stages`-driven autoscheduling.

## The abstraction
- **Tile-level, wave/wave-aware.** Like Triton, a Gluon kernel operates on block-shaped tiles. Unlike
  Triton, the **layout** of each tile (registers, LDS, MFMA operand placement) is an explicit object —
  including **swizzled** and **padded** shared (LDS) layouts authored to avoid bank conflicts. CDNA
  wavefront = **64 lanes**; LDS = 64 KB/CU (CDNA3) / 160 KB/CU (CDNA4); 512 VGPR/EU.
- **MFMA intrinsics are first-class.** `tl.dot` in Triton becomes a hand-issued matrix-core op in Gluon.
  The CDNA4 *scaled* MFMA is exposed as `gl.amd.cdna4.mfma_scaled`, lowering to the hardware op
  **`v_mfma_scale_f32_16x16x128_f8f6f4`** (one instruction consumes FP8/FP6/FP4 operands + a block scale).
- **LDS movement is explicit**, including `ds_read_tr` — the transpose variant of `ds_read` — to feed MFMA
  operands in the right layout without an extra transpose.

## Manual pipelining (vs Triton autoscheduling)
| Concern | Triton (autoscheduled) | Gluon (manual) |
|---|---|---|
| Pipeline depth | `num_stages` knob; stream-pipeliner picks placement | **explicit** 2-/3-stage pipeline you author |
| Operand layout | compiler-assigned blocked/MFMA layouts | **explicit** swizzled/padded LDS + register layouts |
| Register pressure | compiler allocates (can spill) | **you** budget live values vs 512-VGPR limit (per wave at your occupancy: 256 seen at 2 waves/SIMD) |
| Matrix op | `tl.dot` lowered to MFMA | **you** issue the MFMA intrinsic (incl. `mfma_scaled`) |
| Wave schedule | implicit | **ping-pong / interleave** authored by hand |
| Instruction interleave | compiler | `llirSched` (Triton-IR) + `amdgcnas` (post-asm peephole) |

The interleaving passes are toggled at runtime:
```bash
# enable Gluon's IR-level interleave + post-assembly peephole scheduler
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 9 --K 8192 --dtype fp16
```

## Wave scheduling patterns (why manual)
NVIDIA's producer/consumer **wave specialization underperforms on CDNA3/CDNA4** — AMD's static register
allocation starves the producer waves (only ~80% of peak BF16 GEMM). So Gluon exposes the two patterns that
*do* reach peak on CDNA, both originating in HipKittens:
- **8-wave ping-pong** — two groups of waves alternate MFMA vs memory phases so the matrix core never idles.
- **4-wave interleave** — one wave per SIMD, full 512-VGPR budget, 128×128 tile; the robustness/perf
  successor (no `#pragma unroll` tuning, consistent across ROCm releases).

See [optimization/mfma_scheduling](../../optimization/mfma_scheduling.md) and [languages/hipkittens](../hipkittens/overview.md) for the scheduling theory; Gluon is the
Python authoring surface for both.

## MXFP4 numerics (scaled-MFMA path)
MXFP4 packs **two 4-bit values per byte** and uses a **per-group 8-bit scale factor for every 32 elements**
(microscaling block scale). The scaled MFMA consumes the packed FP4 operands and the block scale in one
instruction (`v_mfma_scale_f32_16x16x128_f8f6f4`), accumulating in FP32. The a4w4 kernel adds a dedicated
**scale pipeline (GR → LW → LR)** — global-read → LDS-write → LDS-read of the scales — alongside the data
pipeline; this is the extra complexity that caps MXFP4 at 92.41% (vs ~99% for BF8) due to LDS port
contention. (The tutorial uses the term "8-bit per-group scale"; it does not spell out "E8M0".)

## What recorded GEAK runs showed (gfx950, DSA kernels)
Most recorded directions that followed this page did not beat their parent. Lessons the reports support:
- **Budget registers at the occupancy you run, not at 512.** The dsa-prefill split arm ran at 2 waves/SIMD and 256 VGPR. Raising `BLOCK_M` from 4 to 8 kept occupancy, LDS and bit-exactness, but spills went from 19 to 351, scratch from 80 B to 952 B, and the kernel was about 2.8x slower. The engineer attributes this to 8 rows of Q and weights not fitting in registers; keeping Q in LDS was left untested.
- **ISA counts did not rank candidates; interleaved timing did.** In dsa-prefill a reduce-scatter epilogue halved `v_permlane32_swap` (56 to 28) and cut spills from 19 to 14, yet was 16.5% slower (cause unprofiled). Splitting the body loop into masked boundary runs plus an unmasked interior run raised split-arm spills from 1 to 29 and scratch from 8 to 104 bytes, yet was 4.41% faster; a zero-scratch single-loop variant was 2.4% slower. A stacked-row MFMA A operand with a matching `DistributedLinearLayout` epilogue reached 0 spill / 0 scratch and was still 13.5% slower.
- **What held:** a `@gluon.jit` twin of the dsa-sparse-mla ONE_TRIP body (bs1/bs2) staged the gathered KV tile once into one swizzled LDS buffer, read as the QK B operand (kWidth16, scaled MFMA 16x16x128) and as the PV B operand (kWidth8, MFMA 16x16x32): +3.9% call-weighted, bs1 +8.7%, bs2 +10.8%. A variant loading KV directly global-to-LDS failed the ISA gate.

## Sources
- Gluon GEMM tutorial (layouts, pipeline stages, register budget, `mfma_scaled`, `ds_read_tr`, `llirSched`/`amdgcnas`, scale pipeline GR→LW→LR): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Tutorial code (a16w16 256×256×64, a8w8 256×256×128, a4w4 256×256×256): https://github.com/ROCm/gfx950-gluon-tutorials
- Wave-specialization-fails-on-CDNA / ping-pong + interleave origin: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/languages/gluon/programming_model.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.

<!-- kernelwiki-rsi:evidence -->
## Evidence (KernelWiki-RSI)

Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. Recorded GEAK runs; associations, not causes; not validated online.

- 54 recorded directions used this page; 22% beat their parent by more than 0.5%; 39% were continued by later rounds.
- 12 paired contrasts with sibling directions that did not use it: mean -5.1%, range -59.5% to +1.0%.
- Cited directions:
  - `node039/dsa-prefill/20260924T020511_be9d88523d4e4759a817fca9b3cc9ffb/1`: dsa-prefill, round 17, compute; -59.6% over its parent
  - `node039/dsa-prefill/20260924T030150_8d4dc6278ef84091ac7bbab033f3cf26/1`: dsa-prefill, round 19, memory; -15.5% over its parent
  - `node039/dsa-prefill/20260923T195214_9c017fd339f5445e8a6f7adb0463be0b/0`: dsa-prefill, round 7, algorithm; -12.7% over its parent
  - `node039/dsa-prefill/20260923T195214_9c017fd339f5445e8a6f7adb0463be0b/1`: dsa-prefill, round 7, compute; +4.2% over its parent; continued
  - `node039/dsa-sparse-mla/20260924T035158_4e17408b207845d89158f7c4122f074a/20260924T084933_5e87494d629e4f928d1572a1091ff501/0`: dsa-sparse-mla, round 17, memory; +3.8% over its parent; continued
<!-- /kernelwiki-rsi:evidence -->
