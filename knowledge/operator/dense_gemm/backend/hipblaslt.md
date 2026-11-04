---
title: dense_gemm on hipBLASLt — SOTA card
kind: sota_card
operator: dense_gemm
backend: hipblaslt
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
  - ROCm/aiter@a6bb4993:aiter/tuned_gemm.py
---
# dense_gemm × hipBLASLt

## TL;DR
hipBLASLt (Tensile-generated `Cijk_*` kernels) is the **default BLAS** and the kernel actually executed for
dense bf16/fp16/fp8 GEMM on Instinct — directly under PyTorch, and indirectly **under aiter**
(`hipb_gemm → hipb_mm`). For raw-torch paths, tune it **offline** (dump shapes → `hipblaslt-bench` search →
override file). On sglang/vllm it runs under aiter's dispatcher, so the override file is **not consulted** —
deploy the win through aiter's DB instead ([operators/dense_gemm/backends/aiter](aiter.md)).
In recorded gfx950 bf16 runs, per-M solution-index pins helped only at some prefill sizes. A pin counted only
if it also won under the harness's own timer (see *Recorded runs* below).

## SOTA implementation
aiter wraps hipBLASLt directly; the chosen solution index travels in the tuned CSV (`solidx`). From
`/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`):

```python
def hipb_gemm(inp, weights, solidx, bias=None, otype=None,
              scale_a=None, scale_b=None, scale_c=None, bpreshuffle=False):
    if not extensions_created:
        hipb_create_extension(); ...
    return hipb_mm(inp, weights.t(), solidx, bias, otype,
                   scale_a, scale_b, scale_c, bpreshuffle)
```

`solidx = -1` means "let hipBLASLt heuristically pick"; a tuned positive index pins a specific Tensile
solution. Offline (raw-torch) the same indices are discovered by `hipblaslt-bench`.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| hipBLASLt Tensile `Cijk_*` (tuned solution index) | ROCm/hipBLASLt | gfx908–950; bf16/fp16/fp8 | bf16 **708 TFLOPS** (M4096,N4864,K32896) ≈54% of 1307 theo. peak; fp8 **1223.6 TFLOPS**; fp16 654 TFLOPS @ MI300X, ROCm 6.3, AMD Labs Feb-2025 | default dense GEMM; any BLAS-dispatch framework |
| hipBLASLt FP8 (CDNA4 no-tune bar) | ROCm/hipBLASLt | gfx950; fp8 | **~2750 TFLOPS** @ M=N=K=4096; **~3130** @ 8192 (MI355X, ROCm 7.1) — the bar HIP/C++ kernels target | the default FP8 GEMM bar on MI355X |
| TensileLite custom logic (generate new instances) | ROCm/hipBLASLt TensileLite | gfx942/950 | shape-specific; closes gaps where pooled solutions are weak | a hot shape with no good pooled solution |

Note: ~80% *software* efficiency (vs clock-adjusted peak) but only **~45%** of *theoretical* peak sustained
on MI300X across fp8/bf16/fp16 — the gap is power/clock throttle under the 750 W cap (third-party,
narrowing), not just kernel quality (Ambati & Diep 2025). On **CDNA4 (MI355X)** the picture is far
healthier: the FP8 hipBLASLt bar is ~2750@4096 / ~3130@8192 TFLOPS, and it is now **beaten without
assembly** by a HIP/C++ 8-wave ping-pong kernel (3204 @ 8192) — see
[operators/dense_gemm/backends/asm](asm.md) / [operators/scaled_quant_gemm/tuning](../../scaled_quant_gemm/tuning.md) and the ping-pong
scheduling prior in [optimization/mfma_scheduling](../../../general/optimization/mfma_scheduling.md).

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `TORCH_BLAS_PREFER_HIPBLASLT` | 0 / 1 | route torch `addmm`/`mm` to hipBLASLt vs rocBLAS | 1 (recent ROCm) |
| `HIPBLASLT_LOG_MASK` | bitmask (32 = bench) | dump GEMM shapes for offline tuning | 0 |
| `hipblaslt-bench --algo_method` | all / index / heuristic | search all solutions vs replay a chosen index | heuristic |
| `HIPBLASLT_TUNING_FILE` | path | write/read tuned solution map (raw-torch only) | unset |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | path | force the tuned solution at runtime (raw-torch only) | unset |
| `OPTIMIZE_EPILOGUE` | 0 / 1 | fuse epilogue (bias/act) into the GEMM store | 0 |
| `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` | bytes | bound per-solution workspace | impl-default |
| solution tiling | `mfma_16x16` vs `32x32` | prefer 16×16 on MI300X for most shapes | per-solution |

MI300X levers: prefer `mfma_16x16` solutions, set `OPTIMIZE_EPILOGUE=1`, avoid 512-B leading-dim strides
(Tagram bank hotspot — pad to non-512-multiple).

## Numerics / parity
Same bf16/fp8 math, different tiling/solution → parity-safe up to accumulation order (cross-solution argmax
flips possible but rare). fp8 paths need a downstream accuracy gate ([../numerics.md](../numerics.md)).

## Integration (rebind seam)
- **raw torch path**: `TORCH_BLAS_PREFER_HIPBLASLT=1` + `HIPBLASLT_TUNING_OVERRIDE_FILE` — engages directly.
- **sglang/vllm**: hipBLASLt runs *under aiter*; the override file is NOT read on the live aiter path. The
  rebind seam is aiter's CSV (`solidx` per shape) — deploy via [operators/dense_gemm/backends/aiter](aiter.md).
- **own dispatch shim** (recorded on gfx950): a per-M `solidx` table keyed on the exact signature (arch, CU
  count, N, K, dtypes, no bias, no preshuffle) routes only those M values to `hipb_gemm`. Every other shape
  stays on the default `F.linear` path. A cached `hipb_findallsols` membership check falls back to the
  default path when a pinned index is missing from the installed hipBLASLt.

## Recorded runs (gfx950, bf16, qkv-a-proj N=2624 K=6144 and o-proj)
- **Prefill pins that held:** on qkv-a-proj, pinning solution indices for M=5916, 16955 and 30519 was
  associated with prefill gains. These held across several rounds of re-landing and alternating seed/candidate
  re-measurement, and decode was left on the unchanged path. rocprofv3 kernel traces showed the pinned sizes
  switching macro-tile (e.g. MT192x192 → MT224x320 at M5916). Unpinned sizes kept the seed's kernels. Use the
  kernel name in the trace as the engagement proof.
- **Microbench winners that did not hold:** on qkv-a-proj, winners picked outside the harness timer did not
  hold at two sizes, so neither size was pinned.
  - M=1936: a hot-cache full-search winner was about 10% faster back-to-back but 13% slower in the harness
    (0.0896 → 0.1026 ms). In a later run, the L2-flushed full sweep of all 2,084 `hipb_findallsols`
    candidates picked 439561 (+3.7%), which scored 0.92–0.94 in `audit_bench`.
  - M=32768: the L2-flushed full-sweep winner 439397 (+3.0%) scored 0.97–1.01 in `audit_bench`. An earlier
    13-candidate hot-cache race found only about 2.9%, below a 3% pin bar.
  - o-proj M1936: no index beat `F.linear` under the unit test's real timer (flushed, eager, one call,
    median). Indices edged ahead only in a warm race and under a flushed 20-call graph replay (45.6 vs
    47.9 us), and neither of those is how that case is timed.
- **Heuristic default can be the best route:** at qkv-a-proj M=1936, the hipBLASLt heuristic path beat the
  best FlyDSL hgemm and vendored Triton `gemm_a16w16` configs in L2-flushed isolated timing. FlyDSL was limited
  to tile_n=64 because N=2624 is not a multiple of 128, and its split-K was blocked by the semaphore limit.
- **Search coverage matters:** one run fully searched only two sizes and timed just 13 candidates at the larger
  sizes, so better indices may exist where the search was partial.
- **Open lead (not tested):** on o-proj, `F.linear` ran faster cold than every raced index. The engineer
  suspected torch calls hipBLASLt with M and N swapped; that swapped solution set was not raced.
- **Lesson:** race solution indices under the same timer and cache state the harness uses. Confirm a winner in
  the harness itself before pinning, and leave sizes on the heuristic path when the harness does not confirm.

## Pitfalls & anti-patterns
- Solution indices are **ROCm/hipBLASLt-version-locked** — re-tune on every upgrade; a stale index can map
  to a different or removed kernel. Guard pins with a solution-pool membership check that falls back to the
  default path.
- `hipblaslt-bench` CLI is **absent in some images** — then aiter/gradlib (which races the same solution
  pool) is your only tuner.
- On sglang, tuning the override file alone does nothing (aiter dispatch) — a known dead-end; verify
  engagement through aiter's `is tuned on cu_num` log, not hipBLASLt markers.
- Don't read theoretical 1307 TFLOPS as achievable — real bf16 lands ~708 (≈54%); plan headroom from there.
- Don't pin from a hot-cache, warm or graph-replay race when the harness times flushed eager calls. In the
  recorded gfx950 runs, such winners at M=1936 reversed or vanished in the harness.

## How to verify (worked example)
```bash
# raw-torch offline tune of one shape
HIPBLASLT_LOG_MASK=32 python my_gemm.py 2> shapes.log          # dump shapes
hipblaslt-bench --algo_method all -m 4096 -n 4864 -k 32896 \
    --a_type bf16_r --b_type bf16_r --compute_type f32_r       # search solutions
TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TUNING_OVERRIDE_FILE=tuned.json python my_gemm.py
# serving path: prove via aiter engagement instead
grep -c 'is tuned on cu_num' server.log
# own-shim pins: prove engagement by the macro-tile name in a rocprofv3 kernel trace
```

## Alternatives / cross-links
[operators/dense_gemm/backends/aiter](aiter.md) (deploy seam on serving) ·
[operators/dense_gemm/backends/ck](ck.md) (custom epilogue / int8) ·
[operators/dense_gemm/backends/asm](asm.md) (peak) · [operators/dense_gemm/backends/triton](triton.md) ·
[operators/dense_gemm/overview](../README.md).

## Sources
- hipBLASLt FP8 bar ~2750@4096 / ~3130@8192 (MI355X, ROCm 7.1); HIP/C++ 8-wave ping-pong beats it (3204@8192, no asm): AMD CDNA4 GEMM blog (https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html).
- hipBLASLt offline tuning: https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
- TensileLite custom tuning: https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
- MI300X GEMM levers (mfma_16x16, epilogue): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- bf16 708 / fp8 1223.6 TFLOPS, ~45% sustained: https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html ; https://arxiv.org/pdf/2510.27583
- aiter wraps hipBLASLt (`hipb_mm`): `/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/dense_gemm/backends/hipblaslt.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.

<!-- kernelwiki-rsi:evidence -->
## Evidence (KernelWiki-RSI)

Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. Recorded GEAK runs; associations, not causes; not validated online.

- 21 recorded directions used this page; 60% beat their parent by more than 0.5%; 48% were continued by later rounds.
- 17 paired contrasts with sibling directions that did not use it: mean -0.4%, range -5.3% to +1.1%.
- Cited directions:
  - `node039/qkv-a-proj/20260923T174848_ff1b01d34055458cbd455114b39ad154/1`: qkv-a-proj, round 1, host_runtime; +1.2% over its parent
  - `node039/qkv-a-proj/20260923T231055_379ccc2f91664f9c96a9a703fa9a824d/0`: qkv-a-proj, round 9, host_runtime; +1.3% over its parent
  - `node039/qkv-a-proj/20260923T233441_4ed29e1beeb9447c99e28b872c6c418a/0`: qkv-a-proj, round 10, host_runtime; +1.3% over its parent
  - `node039/qkv-a-proj/20260923T202153_a5e22410e0c644768461d5478b02db3b/2`: qkv-a-proj, round 5, host_runtime; +1.0% over its parent
  - `node039/o-proj/20260923T205717_118e9400cbc2444ba52a265275425246/1`: o-proj, round 9, host_runtime; -0.2% over its parent; continued
  - `node039/qkv-a-proj/20260924T034903_71663cf09cd242448da7a5e7fb8bff7e/0`: qkv-a-proj, round 19, algorithm; -0.5% over its parent; continued
<!-- /kernelwiki-rsi:evidence -->
