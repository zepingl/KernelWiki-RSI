# KernelWiki-RSI

**Knowledge-as-a-Policy: KernelWiki becomes the exploration policy of the GEAK harness.**

Built on [Dream-RSI](https://arxiv.org/html/2609.14858v1), KernelWiki-RSI turns GEAK's auditable discovery
trees into replayable "dream worlds", evolves a structured scheduling policy against them, and lets the
qualified policy steer the next round of real exploration. The generator, model, evaluator and reference
knowledge stay fixed. Only the exploration decisions change.

> **Status: schema 2.** The mechanism is implemented and covered by local regression tests.
> No self-improvement gain has passed the qualification gate yet. The historical buffer has
> **0 primary worlds**, and diagnostic replay is not promotion evidence.

![KernelWiki-RSI loop: explore, reverse-trace, freeze buffer, dream replay, policy feedback](docs/visuals/kernelwiki-rsi/overview.png)

[▶ Play two iterations](docs/visuals/kernelwiki-rsi/index.html) ·
[SVG](docs/visuals/kernelwiki-rsi/overview.svg) ·
[Case data & reproduction](docs/visuals/kernelwiki-rsi/README.md)

*The figure's mechanisms match the code and its numbers come from real archives. Its topology is a
teaching construction: the same-budget replay gain of 1.08% → 7.78% is illustrative, not a promoted policy.*

## The loop

```
online explore ──► reverse engineering ──► frozen replay buffer ──► inner evolution ──► qualification ──► release
 (fixed agent)      (results → evidence)     (immutable worlds)      (train/validation)   (paired gates)    (KernelWiki CAS)
      ▲                                                                                                       │
      └──────────────────────────────────────── next policy ◄────────────────────────────────────────────────┘
```

- **Outer loop** (`loop/outer.py`) runs online discovery, reverse-traces new evidence, freezes a new buffer
  and invokes the inner loop.
- **Inner loop** (`loop/inner.py`) only selects a working policy. It never sees the test split and never publishes.
- **Release** (`loop/qualification.py`) publishes to KernelWiki via compare-and-swap, and only after fresh
  paired replay and online non-regression gates pass.

## Code structure

```
kernel_workflow/rsi/
├── __main__.py            CLI: ingest · freeze · trace · investigate · replay · evolve · report
│                               explore · calibrate · cycle · release
├── harness.py             Policy + Rollout: one scheduling state machine shared by online and replay
├── discovery.py           Domain values: SourceState, Measurement, Attempt, DiscoveryNode/World
├── evidence.py            Immutable, line-addressable EvidenceRef; JSON/JSONL artifact I/O and hashing
├── runtime.py             RuntimeAgent: fixed experimental agent, bounded processes, frozen sources
├── report.py              Derived coverage and experiment reports (never a source of truth)
├── glm7_evaluator.py      Runs the unchanged GLM7 UT and binds its report to the real HIP device
├── agent_workflow.js      One isolated agent call (developer / reviewer / analyst) for runtime.py
│
├── loop/                  Bounded discovery, policy evolution, final qualification
│   ├── outer.py           cycle(): online → reverse → new buffer → evolve
│   ├── inner.py           evolve(): Developer/Reviewer loop over frozen replay worlds
│   ├── online.py          explore() / calibrate(): live discovery with the fixed GEAK worker
│   ├── evaluator.py       FixedEvaluator: independent fixed-contract host measurements
│   ├── qualification.py   qualify() / release(): paired gates, then KernelWiki publication
│   ├── run.py             Frozen cycle inputs, resource envelopes, stage receipts
│   ├── roles/             developer.md, reviewer.md
│   └── configs/           cycle.example.yaml, online.example.yaml
│
├── replay_buffer/         Immutable buffers, fixed cohorts, finite harness replay
│   ├── build.py           ingest() raw history · freeze() online history + investigations
│   ├── store.py           open_buffer(): read-only verified snapshot
│   ├── cohorts.py         Policy-independent primary/diagnostic cohorts and kernel-level splits
│   ├── simulator.py       Deterministic recorded transitions (no agent, no GPU)
│   ├── objective.py       ReplayObjective with explicit units and macro aggregation
│   ├── evaluate.py        evaluate() / qualifies(): paired fixed-buffer evaluation
│   └── configs/           glm7.yaml
│
├── reverse_engineering/   Evidence-bound provenance of GEAK optimization attempts
│   ├── model.py           Relation, TraceSubject, Coverage, Gap, ReverseReport
│   ├── trace.py           trace(): verified evidence → observations, method reviews, effects
│   ├── investigate.py     investigate(): per-gap search with explicit coverage
│   ├── analyst.py         Optional bounded method review that cannot rewrite facts
│   ├── sources/           archive.py (GEAK audit runs) · capture.py (fixed-harness captures)
│   └── roles/             analyst.md
│
└── docs/                  Visual atlas, roadmap, objective analysis
```

**Dependency direction:** `loop → replay_buffer → reverse_engineering → harness → discovery`.
`evidence` and `runtime` are leaves. `reverse_engineering` and `runtime` never import `loop` or `replay_buffer`,
which `tests/test_structure.py` enforces. Each package keeps its own `tests/`, `roles/` and `configs/`.
The KernelWiki SDK is an external dependency and is not vendored.

## Invariants

- **Replay only reveals recorded outcomes.** It makes no agent or GPU calls and never synthesizes branches or
  continuations, so a missing transition can't buy a cheaper qualifying score.
- **Facts stay separate from judgments.** Sources, measurements and Director decisions keep their raw
  provenance. Analyst reviews can't rewrite them, and a partial search proves no absence.
- **Primary support comes before candidates.** Without train/validation primary support, the inner loop
  spends zero agent calls and keeps the incumbent.
- **Selection is separate from publication.** Real cycles need explicit budget approval, and release needs
  fresh paired replay and online non-regression gates.
- **The agent is fixed** at `claude-opus-5.5 --effort high`, with no model fallback, tools/MCP, session reuse or
  schema retries. Reverse investigation defaults to `--model-calls 0`.

## Quick start

Requirements: Python ≥ 3.10, PyYAML, and a KernelWiki checkout that provides `KnowledgeClient`.
Run these from the **GEAK repository root**:

```bash
export KERNELWIKI_ROOT=/absolute/path/to/KernelWiki
export PYTHONPATH="$KERNELWIKI_ROOT:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export POOL=/absolute/path/to/pool-schema2
export RSI_OUT="$(mktemp -d /tmp/kernelwiki-rsi.XXXXXX)"

python3 -m kernel_workflow.rsi --help

# Coverage report and diagnostic replay over an existing buffer
python3 -m kernel_workflow.rsi report --pool "$POOL" --output "$RSI_OUT/coverage.md"
python3 -m kernel_workflow.rsi replay --pool "$POOL" --wiki "$KERNELWIKI_ROOT" \
  --split train --cohort diagnostic --output "$RSI_OUT/replay.json"

# Reverse-investigate one attempt directly from a raw GEAK run (no buffer needed)
python3 -m kernel_workflow.rsi investigate --run "$AUDIT_RUN" \
  --kernel "$KERNEL" --campaign "$CAMPAIGN" --attempt "$ATTEMPT" \
  --max-steps 4 --model-calls 0 --output "$RSI_OUT/reverse"
```

- `trace` and `investigate` take exactly one of `--run`, `--online` or `--pool`. To merge a report into an
  existing buffer, `CAMPAIGN` must match that buffer's original records.
- Every output is a new immutable artifact and never overwrites existing results.
- `ingest` / `freeze` build buffers, and `evolve` runs the inner loop. The authorization and resume flow for
  `calibrate` / `cycle` / `release` is described in
  [Commands and authorization](../../docs/kernelwiki-rsi-v1.md#commands-and-authorization).

## Current results

| Evidence | Result |
| --- | --- |
| Schema-2 historical buffer | 1,498 attempts · 528 replay-ready · 153 qualified outcomes · 337 diagnostic worlds · **0 primary worlds** · 3 reverse reports |
| Inner-loop gate | `insufficient_primary_support`: 0 agent calls, no promotion, no release |
| Archived schema-1 Opus run | 4 Developer + 4 Reviewer calls. Held-out objective −0.00841 → −0.00809, but full-budget quality regressed 0.00555 → 0.00541, so the result was `no_change` |

No qualified better KernelWiki policy or new discovery tree has been found yet. Recovering historical
provenance is not a new discovery and doesn't prove that knowledge causes the gain.

## Testing

```bash
export KERNEL_EXTRACT_LOCK_DIR="$(mktemp -d /tmp/rsi-test-locks.XXXXXX)"
python3 -m pytest -q kernel_workflow/rsi
"${GEAK_NODE_BIN:-node}" interface/runtime/engine/selftest.mjs   # runtime regression (requires Node)
```

Remote paired CI pins a published KernelWiki SDK commit through `KERNELWIKI_RSI_REF`, a 40-character SHA.
A passing test suite doesn't mean experimental qualification or a Knowledge release.

## Further reading

- [Engineering design & refactor evidence](../../docs/kernelwiki-rsi-v1.md)
- [Archived Opus experiment: candidates, metrics, rejection reasons](../../docs/kernelwiki-rsi-v1-experiment.md)
- [Replay objective analysis](docs/replay-objective-analysis.md) · [Roadmap](docs/KernelWiki-RSI/roadmap.md)
