---
title: dense_gemm on Triton — SOTA card
kind: sota_card
operator: dense_gemm
backend: triton
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html
  - ROCm/aiter@a6bb4993:aiter/tuned_gemm.py
  - https://arxiv.org/html/2511.08083v1
---
# dense_gemm × Triton

## TL;DR
Triton is the fastest backend to **write/iterate** a GEMM and the one PyTorch-Inductor max-autotune emits —
but on a *plain* dense bf16 GEMM it typically **loses to tuned hipBLASLt/aiter** on MI300X (AMD Triton still
under-lowers buffer loads / register lifetimes). Use it when you need a **fused** epilogue (bias+act+residual)
the library can't do, or as an authored Tier-C candidate to e2e-gate. On sglang the dense path is aiter, not
torch dispatch, so an authored Triton GEMM must be wired via the aiter `triton` libtype seam.

## SOTA implementation
aiter exposes Triton as a dispatch target, but the in-tree `triton_gemm` is a thin shim with hard limits.
From `/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`):

```python
def triton_gemm(inp, weights, solidx, bias=None, otype=None,
                scale_a=None, scale_b=None, scale_c=None, bpreshuffle=False):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    assert scale_a is None and scale_b is None and scale_c is None, \
        "Triton gemm_a16w16 does not support scaling yet"
    assert not bpreshuffle, "Triton gemm_a16w16 does not support bpreshuffle yet."
    return gemm_a16w16(inp, weights, bias=bias, dtype=otype)
```

So Triton on the aiter path is bf16/fp16, no scale, no preshuffle. To win you author your own kernel and
register it (or call-site rebind), then e2e-gate.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Inductor/Triton mm template (max-autotune) | `triton`, PyTorch Inductor | gfx90a–950; bf16/fp16/fp8 | ~0.7–0.95× of tuned hipBLASLt on plain bf16 GEMM @ MI300X, 2025 | fused-epilogue GEMM; `torch.compile` path |
| Hand-written Triton GEMM (split-K) | author via kernel layer | gfx942/950 | shape-specific; can win decode skinny GEMM with SPLIT_K filling all 304 CUs (but regressed on gfx950 o-proj decode, see below) | skinny/decode or custom fusion |

perf_knowledge e2e validation: an authored Triton GEMM measured **0.99–1.47× isolated** on target shapes but **did
not beat the aiter env at e2e** (2026-06) — the win must survive e2e gating, not just microbench.

## Config space / knobs
| param | range / typical | effect | default |
|---|---|---|---|
| `BLOCK_SIZE_M/N/K` | 128×128×64 / 256×128×64 | output + K tile | 128×128×64 |
| `GROUP_SIZE_M` | 4–8 | L2 re-use via grouped scheduling | 8 |
| `matrix_instr_nonkdim` | 16 / 32 | MFMA shape — **16 (mfma_16x16) > 32 (32x32)** on CDNA | 16 |
| `kpack` | 1 / 2 | K-packing into one MFMA issue | 2 |
| `waves_per_eu` | 1–4 | occupancy hint | 2 |
| `num_warps` | 2 / 4 / 8 | warps per block; 2 beat 4 on a 16×16 decode tile (below) | 4 |
| `num_stages` | 0 / 2 | **0 for a single GEMM on CDNA** (no async pipeline win); the recorded gfx950 o-proj M≤12 tier (BM16/BN16/BK256) ships 3 and was not A/B'd against 0 (an nw2 + num_stages 2 arm was mixed, about 1% either way per bucket) | 0 |
| `SPLIT_K` | 2–16 | K split across CUs for skinny/decode; regressed on gfx950 o-proj decode (below) | 1 |
| `OPTIMIZE_EPILOGUE` | 0 / 1 | fuse epilogue store | 1 |

Autotune key on `(M,N,K)`. ISA target: want `global_load_dwordx4` (buffer loads) and LDS `_b128` stores.

## Recorded gfx950 decode runs (o-proj bf16, M 3–42, N=6144, K=2048)
- **Look at runtime integer/index work, not just tile knobs.** `ONE_PID_M` (constexpr pid decomposition when `num_pid_m==1`), `I32_C` (i32 C offsets) and a constexpr `NUM_K_ITER` trip count, with no tile/occupancy knob moved, went with decode fold 1.18881 → 1.35120. The three edits were bundled and not ablated; the engineer suspects K-loop unroll but did not verify it. Separately, splitting a coupled `even_mn` flag into `EVEN_M`/`EVEN_N` went with +2.06%; decode M never divides BM, so under the coupled flag N paid a runtime `% N`. That change also replaced `% M` with `tl.minimum(offs, M-1)`. The engineers' reading is that gfx950 has no idiv instruction; no ISA dump was taken.
- **Split-K regressed here.** SPLIT_K=2 (BN=16) / 4 (BN=32) brought the grid to 768 CTAs on 256 CUs. It lost with a `tl.atomic_add` fp32 workspace (geomean 0.70587) and with a deterministic workspace plus a reduce kernel (0.89582). In the deterministic variant only M=3/6 gained; M=36 read 0.56886.
- **M≤12 BM16/BN16/BK256 tier: `num_warps` 4→2** was about 10% faster at M=3/6/12. It was not profiled.
- **Cache modifiers:** dropping the shipped `.cg` B-load cost 2.5%, and `.wt` on the C store also lost. `eviction_policy` was not expressible on the AMD toolchain used.

## Numerics / parity
bf16/fp16 in / **fp32 accumulate** → parity with the library up to tiling rounding. Triton block-scaled fp8/
fp4 is a separate path → [operators/scaled_quant_gemm/backends/triton](../../scaled_quant_gemm/backend/triton.md).

## Integration (rebind seam)
On sglang the dense path is aiter, not torch dispatch. An authored Triton GEMM is engaged either via the
aiter `triton` libtype (a CSV row → `triton_gemm`) or a direct call-site rebind in the model's `LinearMethod`,
then e2e-gated through [operators/dense_gemm/backends/aiter](aiter.md)'s verification flow.

## Pitfalls & anti-patterns
- **AMD Triton buffer loads not default**: may fail to reclaim registers or to lower `global_load_dwordx4`
  — always verify with `AMDGCN_ENABLE_DUMP=1` (want `global_load_dwordx4`, LDS `_b128`).
- The **in-aiter Triton GEMM is a shim** (asserts no scale/preshuffle) — treat "author needed" for anything
  beyond plain bf16/fp16.
- **Don't expect to beat tuned hipBLASLt on plain GEMM** — the win is fusion or skinny split-K, and it must
  hold at e2e, not just isolated (isolated 1.47× still lost e2e here). On gfx950 qkv-a-proj prefill (M=1936, N=2624,
  K=6144) the best vendored `gemm_a16w16` config (128x128x64, 4 warps, 2 stages) read 0.1320 ms vs hipBLASLt
  0.0818–0.0825 ms; the default 256x256x64 config read 0.189 ms.
- `num_stages>0` on a single CDNA GEMM usually *hurts* (no overlap to hide); start at 0.

## How to verify (worked example)
```bash
AMDGCN_ENABLE_DUMP=1 python my_triton_gemm.py 2>&1 | grep -E 'global_load_dwordx4|ds_write_b128'
# isolated bench vs hipBLASLt default on the target shape, then:
# wire via aiter triton libtype CSV row and run the same A/B + parity gate as the aiter card
```

## Alternatives / cross-links
[operators/dense_gemm/backends/aiter](aiter.md) (dispatch + gate) · [operators/dense_gemm/backends/flydsl](flydsl.md)
(higher ceiling, more knobs) · [operators/dense_gemm/backends/hipblaslt](hipblaslt.md) (beats plain GEMM) ·
[operators/scaled_quant_gemm/backends/triton](../../scaled_quant_gemm/backend/triton.md) (block-scaled) · [optimization/mfma_scheduling](../../../general/optimization/mfma_scheduling.md) ·
language deep-dive `languages/triton_amd/` (P1).

## Sources
- Optimizing Triton kernels (knobs, ISA verify): https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- Triton kernel dev notebook: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html
- AMD Triton GEMM under-performance / buffer loads (HipKittens): https://arxiv.org/html/2511.08083v1
- In-aiter Triton shim: `/sgl-workspace/aiter/aiter/tuned_gemm.py` (`triton_gemm`, `ROCm/aiter@a6bb4993`).
- perf_knowledge e2e validation: authored Triton GEMM 0.99–1.47× isolated, did not beat aiter env at e2e (2026-06).

## Migration provenance

Imported from [AMD-AGI/GEAK `02bfbb0b7274`](https://github.com/AMD-AGI/GEAK/blob/02bfbb0b72746941be3ab3ad48fb21a284ae2fde/perf_knowledge/operators/dense_gemm/backends/triton.md) on 2026-10-21. KernelWiki reorganized this document and adapted local links. Performance claims retain their original dates, environments and evidence; this import is not a new benchmark or a current-best certification.

<!-- kernelwiki-rsi:evidence -->
## Evidence (KernelWiki-RSI)

Replay buffer: `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_policy_v2_exp_20261029/pool-schema5-02`. Recorded GEAK runs; associations, not causes; not validated online.

- 29 recorded directions used this page; 35% beat their parent by more than 0.5%; 38% were continued by later rounds.
- 11 paired contrasts with sibling directions that did not use it: mean +0.6%, range -1.6% to +5.5%.
- Cited directions:
  - `node056-0922/o-proj/20260923T085311_c7cde6fd042240baaa77c68ba89ebc3d/0`: o-proj, round 10, compute; +11.0% over its parent
  - `node056-0922/o-proj/20260923T064155_711a11cfd48f4ee5bd29a241ec311e8b/2`: o-proj, round 9, algorithm; +1.9% over its parent; continued
  - `node056-0922/o-proj/20260923T051343_b78cdb13cd1045b9a099d3817ba34d4a/0`: o-proj, round 8, algorithm; -20.8% over its parent
  - `node056-0922/o-proj/20260922T234916_ec516c583d184566ac5c5c5fdecd3d35/0`: o-proj, round 5, algorithm; -0.4% over its parent
  - `node039/o-proj/20260923T182224_1eca4b3276d742ac967388f2f066e610/0`: o-proj, round 2, algorithm; +4.1% over its parent
  - `node056-0922/o-proj/20260923T051343_b78cdb13cd1045b9a099d3817ba34d4a/1`: o-proj, round 8, memory; -0.8% over its parent; continued
  - `node039/qkv-a-proj/20260924T034903_71663cf09cd242448da7a5e7fb8bff7e/0`: qkv-a-proj, round 19, algorithm; -0.5% over its parent; continued
<!-- /kernelwiki-rsi:evidence -->
