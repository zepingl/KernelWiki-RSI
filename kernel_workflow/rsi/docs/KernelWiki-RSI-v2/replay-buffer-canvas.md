# Replay Buffer Canvas · `pool-schema4-01`

> **Data:** GLM 7 kernels · 2 campaigns (node039, node056-0922) · KernelWiki `fb56454` · as of 2026-11-03
>
> **Path:** `/shared_nfs/zepingl/geak_audit/checks/kernelwiki_rsi_schema3_shape_20261029/pool-schema4-01/`
>
> **Scripts that regenerate the numbers:** `…/kernelwiki_rsi_policy_v2_exp_20261029/scripts/` (`buffer_atlas.py`,
> `tree_mermaid.py`, `knowledge_identity.py`, `knowledge_usefulness.py`)

---

## 1 · Structure: how one optimization run becomes a tree

```mermaid
flowchart LR
  subgraph RAW["GEAK raw archive (read-only)"]
    A1["agent_calls<br/>plan_round / optimize / verify"]
    A2["benchmarks<br/>independent measurements"]
    A3["knowledge_lineage<br/>which docs were read"]
    A4["snapshots<br/>source / knowledge copies"]
  end
  subgraph POOL["pool-schema4-01/"]
    M["manifest.json<br/>hashes · splits · budgets · cohorts"]
    T["trace.jsonl<br/>10,568 records"]
    W["worlds / attempts.jsonl<br/>v1, kept in parallel"]
  end
  RAW -- ingest --> POOL
  T --> CT["24 code trees<br/>one per campaign × kernel"]
  T --> KT["8 knowledge / external trees<br/>KernelWiki git projection"]
```

```mermaid
flowchart TB
  Tree["Tree<br/>contract = kernel's measurement contract"] --> Root["Node · root<br/>baseline source"]
  Root --> E["Expansion = one plan_round<br/>Plan: K planned directions · dispatched d"]
  E --> N1["Node · direction d0<br/>source after the change"]
  E --> N2["Node · direction d1"]
  N1 --> MS["Measure<br/>independent timing (lower median)"]
  N1 --> D["Decide<br/>committed / incorrect / stop"]
  N1 -.-> U["Use → KernelWiki node<br/>(pointed / read / cited)"]
  N1 -.-> G["Gap<br/>missing evidence, said explicitly"]
  N1 --> E2["next Expansion<br/>(continues the committed node)"]
```

| Record kind | Count | Meaning |
| :-- | --: | :-- |
| tree | 32 | 24 code trees + 7 knowledge trees + 1 external tree |
| expansion | 1,618 | 356 GEAK plan rounds + 1,262 KernelWiki knowledge changes |
| node | 1,850 | 24 roots + 558 directions + 1,262 knowledge versions + 6 external commits |
| measure | 5,450 | independent measurements (a direction can be measured several times) |
| decide | 818 | committed / incorrect / stop / retired |
| gap | 2,830 | **2,475 of them are "knowledge read outside KernelWiki"** (see § 4) |

---

## 2 · Data: what the 24 code trees contain

| kernel | split | trees | rounds | directions | outcome unknown | primary (fully replayable) |
| :-- | :-- | --: | --: | --: | --: | --: |
| dsa-decode | train | 4 | 54 | 84 | 3 | 0 |
| dsa-prefill | train | 2 | 42 | 74 | 5 | 0 |
| dsa-sparse-mla | train | 5 | 55 | 80 | 3 | 3 |
| mxfp4-moe | validation | 5 | 46 | 80 | 2 | 3 |
| mla-bmm | test | 4 | 56 | 76 | 1 | 2 |
| qkv-a-proj | test | 2 | 45 | 70 | 3 | 1 |
| o-proj | test | 2 | 58 | 94 | 5 | 0 |
| **total** | | **24** | **356** | **558** | **22** | **9** |

- **Directions per round:** 1, 2, 3 or 4 directions in 206, 97, 52 and 1 rounds. This is a greedy lane: it mostly
  continues the round's winner.
- **Which direction history continued from:** 267 rounds continued one of their own directions, from position
  d0 / d1 / d2 in 184 / 58 / 25 of them.
- **Why only 9 trees are primary:** 22 directions have unknown outcomes, and one unknown keeps the whole tree out.
  Their sources are all archived, so about 10 GPU-minutes of re-measurement would fill them in.

---

## 3 · One real tree: `node039/qkv-a-proj` (first 7 rounds)

Green nodes are the trunk history committed. Each node lists the KernelWiki documents its direction read, after
§ 4's path resolution.

```mermaid
flowchart LR
  R(["root · baseline"])
  r1d0["r1·d0 compute<br/><b>+3.8%</b><br/>(no knowledge)"]
  R --> r1d0
  class r1d0 trunk
  r1d1["r1·d1 host_runtime<br/><b>+1.2%</b><br/>backend/aiter.md<br/>backend/hipblaslt.md"]
  R --> r1d1
  r2d0["r2·d0 compute<br/><b>+3.3%</b><br/>backend/flydsl.md<br/>flydsl/knobs.md"]
  r1d0 --> r2d0
  r2d1["r2·d1 algorithm<br/><b>+3.5%</b><br/>backend/flydsl.md<br/>flydsl/authoring_gemm_levers.md"]
  r1d0 --> r2d1
  r3d0["r3·d0 algorithm<br/><b>+3.5%</b><br/>backend/flydsl.md<br/>∉ KernelWiki"]
  r1d0 --> r3d0
  class r3d0 trunk
  r4d0["r4·d0 compute<br/><b>+6.8%</b><br/>backend/flydsl.md<br/>dense_gemm/tuning.md"]
  r3d0 --> r4d0
  class r4d0 trunk
  r4d1["r4·d1 memory<br/><b>+3.6%</b><br/>backend/flydsl.md<br/>flydsl/authoring_gemm_levers.md"]
  r3d0 --> r4d1
  r5d0["r5·d0 compute<br/><b>+6.6%</b><br/>backend/flydsl.md<br/>flydsl/authoring_gemm_levers.md"]
  r4d0 --> r5d0
  r5d1["r5·d1 algorithm<br/><b>+7.3%</b><br/>backend/flydsl.md<br/>flydsl/authoring_gemm_levers.md"]
  r4d0 --> r5d1
  class r5d1 trunk
  r5d2["r5·d2 host_runtime<br/><b>+7.9%</b><br/>backend/hipblaslt.md"]
  r4d0 --> r5d2
  r6d0["r6·d0 host_runtime<br/><b>+8.2%</b><br/>backend/hipblaslt.md<br/>∉ KernelWiki"]
  r5d1 --> r6d0
  r6d1["r6·d1 compute<br/><b>+9.9%</b><br/>flydsl/authoring_gemm_levers.md<br/>splitk_streamk_gemm/tuning.md"]
  r5d1 --> r6d1
  class r6d1 trunk
  r7d0["r7·d0 algorithm<br/><b>+11.3%</b><br/>backend/flydsl.md<br/>flydsl/authoring_gemm_levers.md"]
  r6d1 --> r7d0
  r7d1["r7·d1 host_runtime<br/><b>+11.0%</b><br/>backend/hipblaslt.md"]
  r6d1 --> r7d1
  class r7d1 trunk
  classDef trunk fill:#dff3e4,stroke:#2e7d32,stroke-width:2px
```

What this tree shows:
- **Same parent, several rounds.** Rounds 2 and 3 both start from r1·d0: when no direction beats the incumbent,
  the lane replans from the same parent.
- **The committed direction is not always the best measured one** (r5: d1 +7.3% committed, d2 +7.9%). The lane
  commits on its own verify measurement; the buffer uses the lower median of independent measurements. The gap
  between the two is about 0.5%.
- **The first committed direction used no knowledge.** The flydsl documents then dominate the rest of the trunk.

---

## 4 · Knowledge in the buffer: read heavily, almost unlinked

```mermaid
flowchart LR
  R["2,475 knowledge reads<br/>read 1,558 · pointed 784 · cited 133"] --> H{"exact content hash<br/>equals a KernelWiki file?"}
  H -- "0 (today)" --> X["all recorded as gaps<br/>→ code trees have no knowledge Use"]
  R --> P{"same document by path?<br/>operators/X/backends/Y ↔ operator/X/backend/Y"}
  P -- "1,953 = 79%" --> K["100 KernelWiki documents<br/>content similarity 0.94"]
  P -- "522 = 21%" --> Z["not in KernelWiki<br/>expert_skills, directory reads, learned-card slugs"]
```

- **Why 0/2,475 is an artifact:** GEAK reads its own copy, which drifted slightly from KernelWiki. Matching exact
  content hashes misses all of them; matching by document identity resolves 79%.
- **Coverage:** of 708 KernelWiki documents, only **70** were read by any measured direction. These runs only
  touched GLM's 7 kernels, so unused does not mean useless.
- **Where trunk progress came from:** 123 of 267 trunk steps **used no knowledge**. They are candidates for new
  knowledge.

| Document (after path resolution) | Directions using it | Improved >0.5% | Sibling pairs | Mean gain vs siblings without it |
| :-- | --: | --: | --: | --: |
| general/backend/gluon/programming_model.md | 50 | 22% | 12 | **−5.1%** |
| operator/sparse_attention_nsa/tuning.md | 47 | 26% | 23 | −0.4% |
| operator/fused_moe_grouped_gemm/backend/flydsl.md | 44 | 30% | 9 | +1.7% |
| operator/dense_gemm/backend/hipblaslt.md | 20 | 60% | 17 | −0.4% |
| general/backend/flydsl/authoring_gemm_levers.md | 27 | 41% | 16 | +0.5% |
| operator/sparse_attention_nsa/backend/triton.md | 25 | 20% | 15 | +1.2% |
| ∉ expert_skills/index.yaml | 30 | 20% | 21 | +1.8% |
| ∉ languages/flydsl/ (directory) | 19 | 32% | 6 | **+3.8%** |

> **Read the contrasts as associations, not causation.** Which document a direction reads is chosen together with
> what the direction tries, and samples are small. They locate knowledge worth a closer look; they do not decide
> it.

---

## 5 · What KernelWiki-RSI can do today, and what is missing

```mermaid
flowchart LR
  subgraph NOW["Today: policy only"]
    P1["replay"] --> P2["Developer / Reviewer<br/>edit policy rules"] --> P3["sign → publish → rollback"]
  end
  subgraph NEED["Missing: reverse-optimize knowledge content"]
    K1["① resolve reads by document identity<br/>(drop exact-hash gateway)"] --> K2["② knowledge ledger<br/>use · outcome · trunk contribution · contrast"]
    K2 --> K3["③ knowledge Developer proposes"]
    K3 --> K4a["revise: write measured field evidence<br/>and limits into the page"]
    K3 --> K4b["distill: new card from trunk steps<br/>that used no knowledge (123)"]
    K3 --> K4c["ingest: used but missing<br/>(expert_skills, flydsl dir …)"]
    K3 --> K4d["merge / split / retire:<br/>co-use clusters, unused, contradicted"]
    K4a & K4b & K4c & K4d --> K5["④ evidence-faithfulness review + human sign<br/>→ the same publish / rollback"]
  end
  P3 -.reuse.-> K5
```

| | Policy (done) | Knowledge content (missing) |
| :-- | :-- | :-- |
| Object | `policy.yaml` + rule cards | Pages, learned cards, decision cards |
| Signal | replay score | ledger: who used it, what happened, paired contrast, trunk contribution |
| Gate | pooled replay gate | Every claim traces to a measured node; replay cannot score text changes |
| Proof of effect | online pairing | online pairing: same runs with the old vs the new knowledge revision |
| Change record | `knowledge:policy` revise | `knowledge:*` revise / distill / ingest / merge / retire (v3 already defines these ops) |

---

## 6 · Root fix: identity recorded at read time, KernelWiki the only source (2026-11-03, experiment branch)

```mermaid
flowchart LR
  KW["KernelWiki @ pinned commit"] -- "knowledge_view.materialize<br/>one rule table" --> V["GEAK-layout view<br/>+ manifest {view path → KernelWiki doc}"]
  V --> RUN["RSI online run<br/>agents read the view"]
  RUN -- "kk_refs looked up in manifest" --> U["knowledge use = doc @ commit<br/>(observed, no matching)"]
  H["history (GEAK's own copy)"] -- "same rule table, by path" --> UH["knowledge use = live doc<br/>(inferred, basis recorded)"]
```

| | Before | After |
| :-- | --: | --: |
| Historical reads resolved to a KernelWiki document | 0 / 2,475 | **2,277 / 2,475 (92%)** |
| Conventional paths found by RSI online agents (`operators/<op>/…`) for the 7 GLM kernels | none (KernelWiki layout was handed over) | operator and language docs found for all 7 |
| Identity of an online knowledge use | not recorded | `doc @ commit` from the view manifest |

**What the 198 still-unresolved reads are:**
- **133 are learned cards GEAK distilled during these runs and never ingested into KernelWiki.** Write-back is broken.
- **49 are directory listings.** A directory is not a document.
- **15 are operators that exist only in GEAK.** KernelWiki lacks 34 of GEAK's 54 operators.

**New finding:** the dsa kernels use gluon (155 plans). KernelWiki has `sparse_attention_nsa/backend/gluon_aiter.md`,
but agents look for `backends/gluon.md`, which neither side has, so this card is never reached.

**Ledger** (`pool-schema5-01-trace-report.md`, "Knowledge findings"):
- 76 of 760 live KernelWiki file versions were read;
- 21 documents have mixed evidence;
- 21 trunk steps improved more than 0.5% with no knowledge (distillation candidates, up to +12.4%).

**Not yet covered:**
- Online runs observe what the plan points to (`kk_refs`), not every file an engineer opens. That needs a read
  hook, as campaign audits already have.
- Online runs still emit v1 worlds; converting them to schema-4 trees is a separate step.

---

## 7 · Corrections, new data paths, first knowledge migration (2026-11-03)

**Corrections**

| | Before (§6) | After |
| :-- | :-- | :-- |
| Learned cards "never written back" | 133 reads | Wrong: 123 of GEAK's 135 cards were imported as `kernel-<slug>`. Only **12** are missing (quantize_cast and memory_movement have no KernelWiki operator). |
| History reads resolved | 92% | **97% (2,403 / 2,475)**. The remaining 72 are 50 directory listings, 10 GEAK-only operator pages and 12 reads of the missing cards. |
| gluon page | "missing" | It exists as `<lang>_aiter.md`. The view now also shows it at the conventional `backends/<lang>.md` (40 such aliases), and the manifest records each alias. |

```mermaid
flowchart LR
  subgraph ONLINE["RSI online run (data, step 1)"]
    V["KernelWiki view @ commit"] --> A["agents (claude via agent_events)"]
    A -- "stream-json tool events" --> R["read_uses: Read / Grep / Bash paths → manifest"]
    A -- "plan kk_refs" --> P["view_uses → manifest"]
    R & P --> T["online-trace.jsonl (schema 4)"] --> F["freeze: one projection, new tree_cohorts"]
  end
  subgraph CHANGE["KernelWiki change (step 2)"]
    G["GEAK 02bfbb0b learned cards"] --> PR["propose-ingest: 12 pages, rendered byte-exact"]
    PR --> RQ["request → sign → publish → rollback / restore"]
  end
  subgraph LANE["Production lane (step 3, opt-in only)"]
    KV["knowledge-view"] --> LA["perf_knowledge_dir · learned_kb_dir · learned_inbox_dir · run_id"]
  end
```

- **Step 1:** an online run now records both what plans pointed to and which knowledge files agents actually opened,
  each as `document @ commit`, and writes a schema-4 tree that `freeze` merges into the buffer.
- **Step 2:** `migration-02` proposes 12 learned cards. They are built from the pinned GEAK blob plus KernelWiki's
  import header and footer, with no other change. It **awaits signature**.
- **Step 3:** the production lane can read KernelWiki through 4 opt-in args. With the args unset, all 11 prompts
  of a stubbed run are byte-identical to before. Nothing in production has been switched.
