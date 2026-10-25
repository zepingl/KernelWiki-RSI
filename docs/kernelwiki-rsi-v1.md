# KernelWiki-RSI v1 — structured Dream-RSI implementation

The current discovery/replay format is **schema 2**. The earlier schema-1
[Opus experiment](kernelwiki-rsi-v1-experiment.md) remains an immutable historical
result, not evidence that this reconstructed outer loop ran.

**Mechanisms are implemented and CPU-tested; successful real end-to-end policy
self-improvement is not yet qualified.** The current evidence and remaining
authorization boundary are recorded at the end of this document.

## Outcome and boundary

Knowledge-as-a-Policy is the **exploration decision layer of the Harness**:
`H(t) = Policy(K(t))`. The fixed interpreter makes structured knowledge executable;
changing the selected policy changes the next round's scheduling Harness, not the
kernel generator or evaluator.
Following [Dream-RSI](https://arxiv.org/html/2609.14858v1), an isolated Developer
improves scheduling against recorded discoveries; it never imagines missing
kernel outcomes. The runtime, generation prompts, reference knowledge, tools,
model, evaluator, objective and split remain fixed.

This implements the technical direction in `doc/KernelWiki-RSI/roadmap.md`,
independently of its schedule. Reverse investigation follows the local
“Knowledge 逆向工程追溯” design: installed result → source → direct parent →
invocation/plan → knowledge version/body → independent measurement.
Delivery, method adoption, measured increment, threshold, retention and Director
acceptance are separate facts.

Five-Step Engineering and its structural/closed-loop references govern changes
and are supplied verbatim to experimental roles. There is no new workflow
service, database, generic Agent framework, executable policy language, automatic
commit, external harvester or duplicate KernelWiki SDK. Dependency: **GEAK →
KernelWiki**, never the reverse.

## Executable pipeline

```text
raw GLM7 invocation inventory → attempts + source/context discovery trees
                                     ↓
working policy θ(t), separate from active KernelWiki
  → bounded online discovery on train/validation kernels
  → captured source + branch context + independent host measurements
  → in-memory TraceSubject: original Attempt + optional direct parent + sources
  → reverse trace → per-gap local search / method review → updated evidence and gaps
  → ONE freeze into B(t+1): attempts, worlds, reverse.jsonl, fixed cohorts
  → inner: full train feedback → Developer → Reviewer → train replay
           ≤4 opportunities; editing base advances on supported improvement
  → validation selects working θ(t+1), or retains θ(t)
  → θ(t+1) actually controls the next online round (at least two rounds)
                                     ↓
freeze ONE final candidate; no subsequent Developer or test reselection
  → NEW incumbent/candidate online runs on the original test kernels
  → equal source/context/reference/evaluator/physical-placement checks
  → replay both policies on the same fresh incumbent discovery worlds
  → fresh replay AND paired online nonregression, or explicit rejection
  → explicit release: KernelWiki CAS publication as replay_validated
```

Reverse investigation also runs directly on a raw archive or a completed online
capture, without any replay pool. The outer round does not create a
`round-*-normalized` pool: it investigates verified histories first and commits
its final buffer once. Completed stages are reused on resume.

`no_change`, `insufficient_primary_support` and final rejection are meaningful
outcomes, not reasons to manufacture an update. Inner selection changes only
`selected-policy.json`; it never changes the active knowledge pointer.

## Code ownership

All three mechanisms are sibling packages under `GEAK/kernel_workflow/rsi/`:

```text
rsi/
├── discovery.py, evidence.py, harness.py, runtime.py
├── loop/
│   ├── outer.py, inner.py
│   ├── online.py, evaluator.py
│   ├── run.py, qualification.py
│   └── roles/, configs/, tests/
├── reverse_engineering/
│   ├── model.py, trace.py, investigate.py, analyst.py
│   ├── sources/
│   │   └── archive.py, capture.py
│   └── roles/, tests/
├── replay_buffer/
│   ├── build.py, store.py, cohorts.py
│   ├── simulator.py, objective.py, evaluate.py
│   └── configs/, tests/
└── __main__.py, report.py, agent_workflow.js, glm7_evaluator.py
```

| Owner | Responsibility and boundary |
| --- | --- |
| `discovery.py` | Shared Attempt, source/context/measurement identities, generation contract and discovery-tree reconstruction; no duplicate replay or reverse schema |
| `evidence.py` | Immutable, line-addressable references, source snapshots and artifact I/O; not method or policy inference |
| `harness.py` | Fixed Policy interpreter and shared online/replay Rollout state machine |
| `runtime.py` | Bounded processes and the fixed experimental Agent; role schemas/instructions are supplied by their owners |
| `loop/outer.py`, `inner.py` | Outer history/report/buffer/policy feedback and bounded inner Developer/Reviewer evolution; role schemas and prompts live with the inner loop |
| `loop/online.py`, `evaluator.py` | Restricted GEAK generation, calibration and independent fixed host evaluation |
| `loop/run.py`, `qualification.py` | Frozen inputs, explicit authorization/budgets, safe recovery, fresh paired final gates and separate CAS release |
| `reverse_engineering/model.py` | Immutable TraceSubject, ReverseReport, Relation, Gap and Coverage values |
| `reverse_engineering/sources/` | GEAK-specific archive/capture parsing, verified histories, evidence search and original measurement/source checks; no dependency on loop or replay |
| `reverse_engineering/trace.py`, `investigate.py` | Relation/effect interpretation and deterministic per-gap evidence actions with explicit coverage |
| `reverse_engineering/analyst.py` | Its own role schema and evidence-bounded method judgments; no tool selection or performance promotion |
| `replay_buffer/build.py`, `store.py` | Initial ingestion/new-buffer commit and read-only BufferSnapshot access; callers do not traverse pool JSONL |
| `replay_buffer/cohorts.py`, `simulator.py` | Policy-independent frozen support/splits and finite recorded transitions |
| `replay_buffer/objective.py`, `evaluate.py` | Objective units/macro aggregation and paired evaluation/qualification |
| `report.py`, `__main__.py` | Presentation and explicit command composition, not alternative truth sources |

Tests, roles and configuration move with their owners. Package initializers do
not eagerly compose the loop. The old flat mixed modules are retired rather than
kept as forwarding facades.

Dependency direction is `loop → replay_buffer → reverse_engineering → shared values`;
loop also uses reverse investigation directly. The reverse package can operate
without constructing a buffer. The runtime does not import any of these three
mechanisms.

`KernelWiki/kernelwiki/knowledge.py` remains the owner of legal knowledge/state/
policy values and retrieval. `knowledge_store.py` owns immutable revisions,
locking, recovery, rollback and CAS publication. No SDK implementation is copied.

Existing `kernel_lane.js` keeps the normal path and shares its Optimize block
with the restricted one-generation RSI path.
`interface/runtime/engine/rsi_boundary.mjs` captures existing plan/dispatch,
candidate/Verify and Commit/Director boundaries. `agent_workflow.js` and
`glm7_evaluator.py` retain their stable integration locations.

## Replay-buffer boundary

`store.open_buffer(path)` verifies schema-2 artifacts/evidence and returns a
read-only `BufferSnapshot`. Its metadata is a detached copy; worlds and attempts
are immutable domain values. Consumers use:

- `select(split, cohort="primary" | "diagnostic")`, with support fixed before any candidate;
- `attempt(id)`, `parent(id)` and `subject(id)`, with the original direct-parent relation;
- `reverse_reports(kernels)`, bound to inventoried Attempts;
- frozen `budgets` and `objective`.

`build.freeze(base, online_histories, output, investigations=...)` verifies the
original histories and completed reports, then writes one new immutable buffer.
An existing output is never overwritten. Unprovided online traces are derived
once; a completed investigation is not redundantly retraced during the merge.
The manifest is written last; a failed build is not a completed buffer.

Schema-2 `attempts.jsonl`, `worlds.jsonl` and `reverse.jsonl` projections remain
compatible with the existing GLM7 archive. The builder reads both new standalone
`report_version: 1` investigations and existing pool-bound reports. Source
recovery only updates reverse findings: it cannot relabel the original Attempt,
invent context/continuations or promote trace-only data into replay.

## Discovery and replay invariants

- A policy changes root-vs-leaf selection, batch size, stopping and budget use.
  It is structured KnowledgePolicy/PolicyRule data, not generated Python/JS.
- Conditions can inspect task state and the revealed observation prefix only.
  Kernel/world identity, source hashes, future branches, hidden quality and
  retrospective reviews are not policy observation fields.
- A batch is a **set**, including no duplicate root. All parents are chosen before
  any result is revealed. Ordinary generation costs one direction unit; deep
  generation costs two. Worker limit is pinned at one or two; deep uses one.
- Only nonempty batches count as decision rounds. The round limit is
  `budget // direction_cost`. Empty policy stop is recorded, not charged.
- Root consumes recorded branches in dispatch order; a leaf consumes its unique
  recorded continuation. A missing transition returns no outcome, charges no
  attempt cost, **retains the leaf**, and still consumes a nonempty round.
  Unsupported requests cannot earn a cheaper qualifying objective.
- Recorded failures and unknown outcomes consume full attempt cost. Internal
  edits/tests remain evidence inside an external attempt, not extra free nodes.
- SourceState and ContextSnapshot are independent of measurement. A failed
  candidate retains its observed source and diagnostics. A child must name the
  exact retained parent source AND context; failure is not silent root rollback.
- Changed shared context/merges start separate historical worlds or audit
  relations. Missing/ambiguous dispatches block their round instead of being
  compressed into a better-supported tree.

## Objective and qualification

```text
quality   = max(0, qualified_score / root_score - 1)
objective = best_quality - 0.01 * charged_direction_units
```

Budgets are frozen at `1, 2, 4, 6` in the GLM7 configuration. Optional
`beta_parallel` defaults to zero: batch counts are not invented wall-clock
savings. Aggregate budgets within world, worlds within kernel, then macro-average
kernels. Full-budget quality and support coverage remain explicit diagnostics.

The **primary cohort is fixed before policy evaluation**: at least two branches,
at least one continuation, enough recorded cost for the largest budget, and no
unknown outcomes. All attempts and diagnostic worlds remain available. Unsupported
world/budget pairs are never dropped separately for a candidate. Missing primary
train/validation coverage stops evolution before any model call.

| Split | Kernels |
| --- | --- |
| Train | dsa-decode, dsa-prefill, dsa-sparse-mla |
| Validation | mxfp4-moe |
| Historical regression / kernels for fresh final runs | mla-bmm, qkv-a-proj, o-proj |

All campaigns for one kernel stay in one split. Developer/Reviewer see **all**
primary training trajectories at all frozen budgets, train-only reverse evidence,
the original incumbent, the current editing base, and every complete preceding
proposal with matched review/replay feedback. There is no hidden first-N sample.
Malformed/rejected/needs-online/evidence-only proposals consume opportunities.
Validation selects a supported Objective improvement without macro full-budget
quality regression. Ties retain the incumbent.

The inner loop never evaluates test or publishes. After the outer rounds, freeze
one candidate before fresh collection. Existing GLM7 test runs are already
inspected regression data, not a fresh holdout. New runs on those original test
kernels must pass paired source/context/reference/evaluator/runtime/physical-GPU
checks. Fresh replay Objective and **each test kernel's** full-budget quality must
not regress; the paired real online comparison must also pass. No test-driven
reselection follows failure. Fixture execution can at most be
`fixture_validated`, never publication.

## Reverse-engineering mechanism

Ingest enumerates original `index/agent_calls.jsonl`, including failed/interrupted
Optimize calls, and recovers missing entries from archived
`recording/records/tasks/*/launches/*/agents/*/invocation.json`, rather than
enumerating chart points. Source objects, original prompts, context versions and
independently linked measurements are verified. Explicitly excluded non-GLM7
tasks remain in `excluded.jsonl`.

Historical scores are `1 / median(sum(calls * optimized_ms))` across independent
processes. This preserves root-relative latency gain without candidate baseline
drift. Online evaluation pins root normalization and runtime once, measures each
candidate twice, verifies full case coverage and immutable source/contracts, and
ignores Agent-reported speedups. The report must bind the actual physical device.

`trace` returns original inputs/plans, source changes/diffs, internal vs independent
measurements, knowledge delivery, review-only method roles, installation scope,
typed relations, effects and explicit gaps. Exact-body delivery requires the raw
returned text, not a prompt asking to read a card. Empty citations/cold directions
are not proven knowledge-off controls; a later installation is not Director approval.

The [REA reference](https://github.com/morluto/rea) informs evidence ownership,
explicit unknowns and evidence-producing investigation steps, not a new framework.
The actual adapters remain specific to GEAK invocation indexes, original tool
returns, private source boundaries, independent Verify receipts and captured
plan/Commit/Director boundaries.

`trace(subject)` returns a `ReverseReport`; the subject is the authoritative
Attempt, an optional supported direct parent, and original observed references.
It is not identified by a replay-pool path. Its version-1 wire form carries
`report_version`, `subject`, `sources` and the compatible `trace` projection.
Findings are detached when read; reviews cannot mutate the subject.

`investigate(subject, output)` is a rules-controlled loop:

1. Derive concrete gaps from verified facts, not from model speculation.
2. Choose one authorized gap-specific archive action, or a method-role review.
3. Record its question, action, new evidence, inspected scope and unread/
   unavailable paths; retrace from the accumulated evidence.
4. Reconsider the remaining gaps within explicit step, document and model-call
   budgets. A later gap still gets its own action; there are no global
   `local_searched`/`method_reviewed` switches.

Gap states are:
- `resolvable_local`: bounded related-original-evidence lookup, with partial coverage retained.
- `needs_method_review`: optional isolated analyst restricted to supplied evidence.
- `needs_online_control`: requires a separately authorized real control.
- `unrecoverable_history`: absent within the fully inspected named archive scope, or an explicitly uncaptured boundary; retain unknown.

A partial/read-budget-limited search cannot establish absence. Unread batches can
be inspected in later steps; unavailable evidence is reported rather than
silently classified as searched. A complete local search is not proof that an
event never happened outside its recorded scope.

The analyst can only label method role and cite supplied evidence. It cannot
change identities, measurements, retention, Director or causal gain. An
inconclusive review does not itself justify another identical review: ordinary
comparison of the relevant original method evidence controls reconsideration.
New source/plan/diff/receipt evidence can make a later review legal. Calls default
to zero; enabling them still uses the fixed Opus/high runtime with no tools.

Initial report, per-step receipts, review output, coverage and final gaps are
durable artifacts. The final result is `resolved` or `audited_with_limits`, not
a claim that knowledge caused a speedup. The buffer keeps its projection in
`reverse.jsonl`; the next Developer receives only the training subset.

Raw regression tests use actual local ONE_TRIP/MoE/Absorb archives, not pre-filled
effect summaries. Absorb's saved candidate is call
`20260930T034949_9d15c81467ab45fab8f1080de98b615a` (`r2_d0`), not its preceding
no-code-change diagnostic call. A recovered save does not recover missing direct
parent measurement or Director acceptance. Missing local archives skip explicitly.

## Online and resource boundary

Restricted generation runs one fixed TechLead plan and one ordinary/deep Engineer
block, returning before Setup/WarmStart/shared memory/Commit/writeback. Developer,
Reviewer and reverse analyst use exactly `claude-opus-5.5 --effort high`, no tools,
no MCP, no persistent sessions, no fallback and zero schema retries. Generation
retains its fixed Bash/Read/Write tools.

Policy records and `policy.yaml` are excluded from injected reference knowledge.
Continuation contains full prior plans, results, diagnostics, measured outcomes and
retention evidence. All batch generators finish before serial host evaluation,
so an evaluator does not overlap another unfinished generation on its GPU.
Inputs/code/reference copies are archived; live frozen inputs are checked during
execution. Archiving starts before root evaluation, so a failed baseline also
retains its exact code and configuration. Candidate artifacts are not
installations into the user's source:
`director_status=not_run`.

Real discovery requires existing NRC allocation metadata, consistent logical and
physical GPU order, and read-only source/GEAK/KernelWiki/context mounts. Reports
need `device_identity: {node, pci}` from the actual runtime device. Do not add a
second HIP/ROCR/CUDA visibility filter. The NRC frontend runs on the login host;
use existing RUNNING allocations and existing images only. An NRC run ID must be
inspected to terminal status before claiming completion.

`gpu_lock.sh` preserves NRC's original visibility filter and checks the selected
PCI device instead of guessing a render-node index. Its command receives the
selected logical `GEAK_GPU_ID`; a coordinated multi-GPU command must explicitly
select that device (for GLM7, pass `--gpu-id "$GEAK_GPU_ID"` inside the locked
command). It does not reap processes owned by the NRC supervisor. Ordinary
non-NRC callers keep the existing visibility-selection behavior.

For original schema-2 GLM7 workspaces, use
`[python3, -m, kernel_workflow.rsi.glm7_evaluator, --gpu-id, '{gpu_id}']` and
`result_file: ut_result.json`. This adapter executes the original `unittest.py`,
keeps its timing and correctness logic unchanged, and adds only observed HIP PCI
identity. Freeze the original UT, harness/weight/meta files and every
`baseline_src` file in `contract_files`; `case_weights` must cover the exact
original cases. Stale reports, legacy reports and changed runtime devices cannot
qualify. No replacement timing harness is introduced.

NRC `--container-isolation` disables network: an in-container live model call is
**not** assumed to work with that option. Verify credential/network isolation and
the real evaluator deployment before claiming a calibration. CPU fake-worker
tests prove code contracts, not that production deployment has passed.

## Commands and authorization

Use matching editable repositories; no SDK mock replaces the real dependency.

```bash
cd /home/zepingl/work/production
export PYTHONPATH="$PWD/KernelWiki:$PWD/GEAK"
export GEAK_NODE_BIN=/shared_nfs/zepingl/geak_audit/tools/node
export GEAK_CLAUDE_BIN=/home/zepingl/.local/bin/claude
python3 -m kernel_workflow.rsi ingest --output /absolute/path/new-schema2-pool
python3 -m kernel_workflow.rsi replay --pool "$POOL" --wiki "$PWD/KernelWiki" --cohort diagnostic --output "$BASELINE"
python3 -m kernel_workflow.rsi trace --run "$RAW_RUN" --kernel "$KERNEL" --campaign "$CAMPAIGN" --attempt "$CALL_ID" --output "$TRACE"
python3 -m kernel_workflow.rsi investigate --run "$RAW_RUN" --kernel "$KERNEL" --campaign "$CAMPAIGN" --attempt "$CALL_ID" --max-steps 4 --read-limit 64 --model-calls 0 --output "$INVESTIGATION"
python3 -m kernel_workflow.rsi investigate --online "$ONLINE" --attempt "$ONLINE_CALL" --model-calls 0 --output "$ONLINE_INVESTIGATION"
python3 -m kernel_workflow.rsi evolve --pool "$POOL" --wiki "$PWD/KernelWiki" --output "$ROUND"
python3 -m kernel_workflow.rsi freeze --pool "$POOL" --online "$ONLINE" --investigation "$INVESTIGATION" --output "$NEXT_POOL"
```

For `trace`/`investigate`, choose exactly one of `--run`, `--online` or the
compatible `--pool` input. A raw run additionally needs its `--kernel` assignment.
No replay pool is created by tracing or investigation. When merging a standalone
archival report, use the same campaign namespace as ingestion (`--campaign`);
original Attempt binding is deliberately strict.

Output directories are new and immutable. `--policy` reads a working policy
without changing active knowledge; `--revision` pins stored knowledge.
`evolve --publish` is explicitly rejected. Schema-1 pools are rejected by the
new reader: reconstruct to a new output, never silently recertify old archives.

`kernel_workflow/rsi/loop/configs/online.example.yaml` illustrates the evaluator/runtime contract.
`kernel_workflow/rsi/loop/configs/cycle.example.yaml` deliberately leaves all full-run limits and approval
unset. The initial GLM7 corpus configuration is
`kernel_workflow/rsi/replay_buffer/configs/glm7.yaml`.
No full-run budget is inferred from calibration authorization.

1. Run `calibrate` with normal mode, `budget: 1`, one GPU and an explicit limit
   of at most 1800 seconds. It records actual cost and does not start a cycle.
2. Set `calibration` to that result and propose `limits`: total direction units,
   total Agent calls, peak GPUs and wall-clock seconds. Run `cycle` to produce
   `budget-request.json`. Without separate approval, it makes no model/GPU calls.
3. Obtain approval JSON containing `decision: approve`,
   `scope: kernelwiki-rsi-cycle`, `approved_by`, an exact EvidenceRef in
   `request` to that budget request, and identical `limits`. Set `approval`
   to this path. Changed code/pool/knowledge/plan requires a new request.
4. Start the authorized cycle in a new output. `--resume` verifies completed
   result/artifact hashes and reuses them. A started stage without a valid done
   receipt blocks automatic Agent/GPU retries, even if a partial result exists.
   Inspect/cancel that original NRC run; never resubmit an ambiguous operation.
5. After real final gates pass, `release` reproduces qualification and performs
   CAS from the writable control checkout. Discovery mounts remain read-only.

```bash
python3 -m kernel_workflow.rsi calibrate --config calibration.yaml --wiki "$PWD/KernelWiki" --output "$CALIBRATION"
python3 -m kernel_workflow.rsi cycle --config cycle.yaml --wiki "$PWD/KernelWiki" --output "$REQUEST"
python3 -m kernel_workflow.rsi cycle --config approved-cycle.yaml --wiki "$PWD/KernelWiki" --output "$CYCLE"
python3 -m kernel_workflow.rsi cycle --config approved-cycle.yaml --wiki "$PWD/KernelWiki" --output "$CYCLE" --resume
python3 -m kernel_workflow.rsi release --cycle "$CYCLE" --wiki "$PWD/KernelWiki"
```

## Validation and limits

KernelWiki retains its nonblocking writer lock, immutable snapshots, expected
revision, user-edit protection, evidence verification and recoverable CAS/rollback.
Publication is `replay_validated`, never `online_validated`. A paired online gate
does not establish reference-knowledge causality or general GPU benefit.

```bash
export KERNEL_EXTRACT_LOCK_DIR="$(mktemp -d /tmp/rsi-test-locks.XXXXXX)"
python3 -m pytest -q KernelWiki/tests KernelWiki/kernelwiki/_store/tests GEAK/kernel_workflow/scripts/tests GEAK/kernel_workflow/rsi
"$GEAK_NODE_BIN" GEAK/interface/runtime/engine/selftest.mjs
```

Paired CI includes the real SDK, raw-evidence tests when archives are mounted,
fake-worker online execution and two complete outer rounds. It requires a real
immutable matching KernelWiki SHA; uncommitted work cannot be remotely pinned.

Finite replay cannot predict unrecorded outcomes. Shallow historical worlds are
diagnostic evidence, not primary RSI support. True knowledge causality, prompt or
reference evolution, out-of-distribution gains and real-model/GPU improvement
require additional explicitly authorized evidence.

## Reconstruction evidence (2026-10-22)

Artifact root:
`/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_reconstruction_20261022/`.
The current pool is **`pool-schema2-04`**, not the earlier investigative pools or
the archived schema-1 `pool-05`.

| Evidence | Observed result |
| --- | --- |
| Raw Optimize inventory | 1,498 GLM7 attempts; 259 missing index entries recovered from archived invocations; 329 out-of-scope attempts explicitly retained separately |
| Replay coverage | 337 diagnostic worlds, 528 replay-ready attempts, 153 qualified outcomes, **0 primary worlds** |
| Historical evolution gate | `historical-primary-gate/result.json`: `insufficient_primary_support`, zero Agent calls, no publication |
| Reverse ONE_TRIP | Direct-parent gain 0.04279167579889798 and retained candidate; Director acceptance unknown; causal knowledge gain not established |
| Reverse MoE | Parent restored; rollback observed; no retained candidate and no invented parent gain |
| Reverse Absorb | Correct r2_d0 saved candidate recovered; direct-parent measurement and Director acceptance still unknown |
| Reverse investigations | Three persisted investigations merged into `reverse.jsonl`; zero analyst model calls |
| Outer mechanism tests | Two real fake-worker rounds feed the selected policy and reverse evidence into the next round; fresh paired gates, regression rejection and nonduplicating resume exercised |

Pre-refactor paired local validation (2026-10-22): **451 passed, 1 skipped** across both KernelWiki
test directories and GEAK's complete workflow-script test directory; **108 passed,
0 failed** in Node runtime selftests. Ruff, shell syntax, paired CI YAML parsing
and both repositories' `git diff --check` pass. The tests use isolated source-lock
directories; no source or qualification guard is disabled to obtain these results.
Remote CI is not claimed: it still requires a published matching KernelWiki SHA.

`REPORT-schema2.md` summarizes historical coverage and gaps. Missing historical
continuations are not reconstructed from chart adjacency. Diagnostic replay is
available, but these shallow histories cannot authorize Developer calls or
policy promotion through the primary gate.

## Structural refactor evidence (2026-10-25)

The refactor retains the authoritative `pool-schema2-04` archive unchanged.
Read-only replay still reports 1,498 attempts, 337 diagnostic worlds and zero
primary worlds. SHA-256 of the **complete canonical evaluate result**, including
trajectories, support and qualification, matches the pre-refactor baseline:

| Check | SHA-256 |
| --- | --- |
| Frozen cohorts | `f9893ab07e2f2f67a5c5db2f3169f25d4578a64b97ac61121acca180de9d66f3` |
| One worker | `65e5d8223d91cf72c09efc3f619fe74fd47c8497d6a161b1b3221c6102a38534` |
| Two workers | `651f4d277392f0981bbc9cda257a997ab21c5674ed1f5f1fc2337ee8473f4882` |

The actual ONE_TRIP, MoE rollback and Absorb saved-source tests now trace directly
from raw archives, without constructing test pools. Additional checks cover
per-gap multi-step progress, partial coverage, unchanged-review suppression,
review after relevant new evidence, immutable report/metadata boundaries,
original-attempt binding, old pool-bound report compatibility, standalone raw/
online CLI, one final buffer per outer round, import direction and absence of
architectural cycles. The two-round fake-worker tests still exercise fresh
paired qualification, rejection and nonduplicating recovery.

Current paired local acceptance: **470 passed, 1 skipped**, including all **104
RSI checks**, both KernelWiki test directories and GEAK's complete workflow-script
test directory. The real ONE_TRIP/MoE/Absorb archives are exercised, not skipped.
Ruff, 47-module Python parsing, shell syntax, paired CI collection and both
repositories' `git diff --check` pass. Node runtime selftests report **108 passed,
0 failed**. No additional real model/GPU experiment or policy publication ran
during this refactor.

Decision receipt: the user requirement is independently understandable loop,
reverse and replay ownership, supported by the actual archive/capture workflows.
Delete the flat mixed modules, global gap-skip flags, runtime-owned role schemas,
ad hoc caller JSONL traversal and intermediate normalized pools; retain verified
source/context/measurement semantics and immutable legacy archives. The minimum
mechanism is typed immutable provenance, deterministic per-gap actions, one
shared scheduling Harness and one buffer commit per round. Local paired acceptance
passes; the next legal action is code handoff. An additional real calibration
requires explicit authorization. The separate fresh-holdout/publication gate
remains unresolved and is not bypassed by CPU fixture results.

### Deployment trial and remaining gate

CPU-only NRC preflight `nrc-651a041b891d40238999cd40cc9e38e5` completed with
exit code 0 on the existing image `rocm/atom-dev:nightly_202610221638`.
It verified read-only source/code/knowledge/context mounts, a private HOME,
original contract files and weights, Torch 2.10.0 / Triton 3.8.0 / AITER imports,
Node v22.23.2, Claude CLI 2.1.306 and gateway TCP connectivity. This made no model
call and used no GPU. Credentials were mounted as specific read-only files, not
copied into configuration or command arguments. AITER uses its supported private
`AITER_JIT_DIR`; the image and original UT are not patched.

The single authorized real calibration was attempted with normal mode, budget 1,
one GPU and a 1740-second internal limit. NRC run
`nrc-196e6ae1b9544b5787f5a82e4717da49` on `crsuse2-m2m-061`, physical PCI
`0002:00:01.0`, terminated with exit code 1 at 18:09:57 UTC.
`calibration-work/calibration-01/failure.json` records **6.695 seconds,
zero generation requests and zero model calls**. Root qualification failed before
any usable GPU measurement: the adapter imported Torch while the workspace's
`unittest.py` shadowed the standard library. This was an adapter defect, not a
kernel performance failure. The failed result and original logs are retained;
it is not a completed cost calibration or a new replay world.

The adapter now removes the workspace from dependency search before importing
Torch and restores interpreter state afterward, without renaming or changing the
original UT. The bug is reproduced by a fresh CPU subprocess regression. The
real-image CPU regression `nrc-5165e982dad34a4d8549963df5795a2a` also completed
with exit code 0: the unchanged workspace now imports correctly and reaches the
expected no-GPU boundary. This does not recertify the failed GPU trial.

No GPU retry is automatic. A second one-direction calibration needs explicit
authorization and a new output directory; a full cycle still needs a separate
post-calibration budget approval. No current real-model self-improvement, fresh
holdout qualification or policy publication is claimed. The active KernelWiki
revision remains
`d166eedd01c6ceadacfa8f9549d551e617ff6be62efa994e2efec3c7f880a8da`.
GEAK remains on `KernelWiki-RSI`, KernelWiki remains on `main`; neither repository
has been committed or pushed.
