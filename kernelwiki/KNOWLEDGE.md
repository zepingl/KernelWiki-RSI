# Knowledge policy API

`KnowledgeClient` is separate from `KernelClient`'s runnable-source API. It reads
operator/general Markdown and a versioned `knowledge/policy.yaml`, normalizes
ordinary pages as non-executable references, and exposes immutable knowledge
values to GEAK. Existing front matter remains plain text unless it declares the
explicit root-level `kernelwiki:` namespace.

```python
from kernelwiki import KernelState, KnowledgeClient

client = KnowledgeClient("/absolute/path/to/KernelWiki")
snapshot = client.snapshot()
records = client.query(KernelState(operator="dense_gemm", gfx="gfx950"), top_k=3)
print(snapshot.revision, [record.id for record in records])
```

Unknown state fields are empty strings and do not match conditions. Ordinary
operator paths imply applicability; explicit records contain `id`, `title`,
`body`, `applies_to`, `evidence` and optional `policy_rule`. Only records listed
by the active policy are executable. Records do not execute arbitrary code.
Conditions and allowed values are validated at the public boundary.

```yaml
schema_version: 1
rule_ids: []
top_k: 3
workers: 2
```

`KnowledgePolicy` allows one to three retrieved rules and one or two workers.
`PolicyRule` effects are prefer/defer/stop; GEAK owns their fixed interpretation
and replay qualification. New scheduling rules cite exact file SHA-256 and line
ranges. Evidence kind distinguishes observation, claim and review; a citation
is not proof of independent causal knowledge gain.

## Publication and rollback

`publish(KnowledgeUpdate(...), expected_revision)` validates policy references,
supporting evidence and the status `replay_validated` or `evidence_update`.
Evidence-only updates cannot change policy behavior. GEAK supplies the measured
qualification record; KernelWiki enforces the local publication contract, not
the experiment itself. Human callers must not fabricate that qualification.

Publication holds a nonblocking POSIX flock, checks the expected revision and
managed worktree, persists immutable versions under `.kernelwiki-rsi/versions`,
then updates the editable documents and active pointer. Files are content-addressed
and checked on reads; managed symlinks are forbidden. Small evidence files are
copied into the version; large raw artifacts stay externally referenced.
Concurrent writers, stale revisions, changed evidence and user edits fail closed.

`rollback(target_revision, expected_revision)` restores a previous stored version
through the same gates. `recover()` resumes an interrupted publication only if
current files are from its old/new snapshots; conflicting user edits abort.
During interrupted publication, readers retain the old pinned revision. No Git
commit, branch operation or network push is performed by this API.

```bash
python3 -m kernelwiki --root /absolute/path/to/KernelWiki snapshot
python3 -m kernelwiki --root /absolute/path/to/KernelWiki query --state '{"operator":"dense_gemm"}'
```

Run `python3 -m kernelwiki --help` for explicit publish, rollback and recovery
arguments. GEAK's `docs/kernelwiki-rsi-v1.md` explains the paired experiment,
objective, held-out gates and online feedback. KernelWiki does not import GEAK.
