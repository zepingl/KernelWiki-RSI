# KernelWiki-RSI: archived schema-1 Opus experiment

> Historical record, not current reconstruction acceptance. The old pool and
> frozen code below remain unchanged. Schema 2 changes source/context semantics,
> batch legality, missing-transition handling, primary cohorts and score
> normalization. The old test set has already been inspected and is now regression
> data. Do not re-label these results as proof of the new outer loop, primary
> replay support, reverse-investigation feedback or fresh final qualification.

Current implementation, reconstruction evidence and calibration disposition:
[KernelWiki-RSI schema 2](kernelwiki-rsi-v1.md#reconstruction-evidence-2026-10-22).

Date: 2026-10-22. GEAK development branch: `KernelWiki-RSI`.

## Frozen input and coverage

Artifact root:
`/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_v1_20261022/`.
The final historical pool is `pool-05`, not earlier investigative pools.
It contains 1,498 external optimization attempts, 528 replay-ready attempts,
153 qualified node outcomes and 337 verified worlds. All worlds currently have
depth-one recorded branches. Train/validation/held-out have 137/47/153 worlds.
Unsupported histories remain trace-only with their exact provenance gaps.

`round-opus-01/manifest.json`, `frozen_code`, `frozen_knowledge` and `frozen_pool`
retain the actual experiment implementation and inputs. The incumbent knowledge
revision is `d166eedd01c6ceadacfa8f9549d551e617ff6be62efa994e2efec3c7f880a8da`.
Later ordinary-reference retrieval compatibility improvements do not change
this experiment's empty incumbent policy or its fixed scheduling interpreter.

## Agent calls and training feedback

The live experiment used `claude-opus-5.5 --effort high`, zero schema retries,
no tools/MCP/session reuse, four Developer calls and four independent Reviewer
calls after connectivity preflight. Full arguments, prompts, results and metrics
are retained under `round-opus-01/agents`.

| Opportunity | Disposition | Train Objective | Train full-budget quality |
| --- | --- | ---: | ---: |
| Incumbent | Unchanged baseline | -0.0045126013 | 0.0101601711 |
| 1 | Reviewed and replayed: global stop after stagnant batch | -0.0043414576 | 0.0093692384 |
| 2 | Rejected: applicability broader than the stated evidence | Not replayed | Not replayed |
| 3 | Reviewed and replayed: narrowed operator/backend/gfx predicates | -0.0042893349 | 0.0100568171 |
| 4 | Reviewed and replayed: scoped alternative | -0.0042893349 | 0.0100568171 |

The rejected proposal consumed one opportunity. Reviewer feedback was returned
to Developer without validation/test results. Training gains are hypothesis
feedback, not a publication decision or causal kernel-speedup claim.

## Validation and held-out decision

Candidate 1 improved validation Objective from -0.0093096814 to -0.0085650006
while preserving validation full-budget quality at 0.0057945626. Candidates 3
and 4 tied the incumbent and did not qualify. Candidate 1 alone was selected
for the final held-out comparison.

Its held-out Objective improved from -0.0084097556 to -0.0080863459, but full-budget
quality fell from 0.0055477347 to 0.0054050484. The `o-proj` kernel regressed
from 0.0067211577 to 0.0062930989; the other two held-out kernels were unchanged.
Therefore the final result is **`no_change`**, `published=false`, with the incumbent
retained. No candidate was reselected using held-out feedback, and no weak policy
was forced into KernelWiki to manufacture an update.

This is a completed negative experiment under the archived implementation. It
does **not** prove that the reconstructed Dream-RSI outer loop or reverse
investigation ran, and does not establish successful policy self-improvement,
online GPU benefit, or isolated knowledge causality.

## Reproduction and evidence

- `REPORT-01.md`: generated coverage, gaps and evolution report.
- `round-opus-01/result.json`: authoritative no-change verdict.
- `round-opus-01/opportunity-*.json`: proposal/review/training receipts.
- `round-opus-01/*-validation.json`: paired validation replay trajectories.
- `round-opus-01/incumbent-test.json`, `candidate-1-test.json`: sole held-out pair.
- `trace-one-trip-01.json`: two exact-body deliveries verified against raw events;
  installed candidate observed, final Director status remains unknown in this trace.
- `trace-moe-01.json`, `trace-absorb-01.json`: raw direction/review evidence and
  explicit incomplete provenance, not fabricated replay outcomes.
- `baseline-neutral-train-01.json`: empty-policy neutral baseline matches incumbent.

The old portable effect-summary fixtures were insufficient mechanism evidence
and have been removed from the current tests. Current regressions read original
ONE_TRIP/MoE/Absorb archives. Absorb's saved candidate is the later r2_d0 call
`20260930T034949_9d15c81467ab45fab8f1080de98b615a`, not the older diagnostic call.

Local CPU validation includes the real paired KernelWiki SDK, immutable publication
and rollback, fake-worker online generation and host subprocess scoring, source
guards, KB offline compatibility, trace regressions and runtime selftests. The
paired CI workflow is added but cannot have remotely run or pin uncommitted SDK
code; maintainers must publish a matching KernelWiki commit and configure its SHA.
No new real GPU optimization run is claimed.

Archived schema-1 final checks (not current reconstruction acceptance):
**526 passed, 1 skipped** in the paired Python suite; **108 checks
passed, 0 failed** in runtime selftests. Ruff on new RSI/knowledge modules and
tests, and both repositories' `git diff --check`, pass. Related existing KB,
candidate-floor, dispatch and architecture-policy checks also pass. The source
guard tests use an isolated `KERNEL_EXTRACT_LOCK_DIR` because the host's shared
default `/tmp/team_gpu_locks` is owned by another user; no lock protection was
removed to make tests pass.

The archived frozen implementation was loaded separately after the compatibility
fixes. Its code hashes, knowledge revision, pool evidence and all five archived
incumbent/candidate paired summaries reproduce exactly. This reproducibility
check performs no new model call, proposal or held-out reselection.

## Historical disposition (superseded for schema-2 acceptance)

| Requirement | Evidence and disposition |
| --- | --- |
| Knowledge-as-a-Policy = Harness decision layer | Structured policy, fixed interpreter, explicit source/state boundaries; no arbitrary executable policy |
| Dream-RSI offline loop | Prefix-only batch replay; no Agent/GPU execution in replay; failures charged and hidden branches inaccessible |
| Traceable GLM7 replay buffer | 1,498 attempts imported; 337 verified worlds; trace-only gaps retained |
| Reverse engineering | Real ONE_TRIP/MoE/Absorb traces and golden mechanics tests; observed delivery separated from review/claim |
| Fixed Opus experiment | Four isolated Developer/Reviewer opportunities, live CLI arguments and results retained |
| Replay objective and independent qualification | Macro kernel Objective, validation selection and one final held-out comparison executed |
| Publication/rollback integrity | Real KernelWiki temporary-checkout tests cover successful publication, pinned read, CAS, locking, recovery and rollback |
| Future feedback | Fake-worker online run → host measurements → freeze → replay pool → trace tested end to end |
| Initial real-policy promotion | **Rejected**, not achieved: selected policy regresses held-out o-proj quality; incumbent remains active |
| Remote CI and new GPU benefit | Not claimed; matching SDK SHA publication and a separate real online experiment are still required |
