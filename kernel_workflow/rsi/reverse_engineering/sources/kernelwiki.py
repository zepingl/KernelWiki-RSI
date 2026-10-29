"""Project KernelWiki git history onto knowledge trees as append-only trace records.

Reads committed git objects only; nothing is checked out or written to the repository. Each knowledge
change of a non-merge commit is one expansion producing one node. A node's state is the SHA-256 of the
file at that commit: the identity KnowledgeClient snapshots and GEAK audit snapshots already carry, so a
GEAK read resolves to it. Sources become uses of external nodes (pinned commits) or of earlier KernelWiki
nodes; what cannot be resolved stays an explicit gap. Gate results are not in git, so no measure is made.
"""

from __future__ import annotations

import hashlib
import re
import subprocess

import yaml
from kernelwiki.knowledge import EvidenceRef

from ...trace.gaps import Gap
from ...trace.records import Decide, Expansion, GapRecord, Node, Tree, Use

GATES = {"evidence_set": "corpus re-extraction and test_corpus", "decision_card": "_render_decisions --check",
         "bundle": "bundle validation on GPU"}
FILE_KINDS = (("evidence_set", r"corpus/evidence/[^/]+\.yaml"), ("decision_file", r"corpus/decisions/[^/]+\.yaml"),
              ("learned_card", r".+/learned_knowledge/(?!README\.md$)[^/]+\.md"),
              ("backend_page", r"knowledge/operator/[^/]+/backend/[^/]+\.md"),
              ("operator_page", r"knowledge/operator/[^/]+/[^/]+\.md"), ("general_page", r"knowledge/general/.+\.md"),
              ("bundle", r"kernels/.+/metadata\.yaml"))
PIN = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/(?:tree|blob|commit)/([0-9a-f]{12,40})")
EVIDENCE_ID = re.compile(r"^\s*-?\s*\w+_id:\s*\"?((?:src|cfg|sel)_[0-9a-f]+)", re.M)
RUN_KEYS = {"run", "run_id", "run_ref", "session_id", "exp_dir", "eval_dir", "source_run"}
EXTERNAL = "external"


def kind_of(path):
    return next((kind for kind, pattern in FILE_KINDS if re.fullmatch(pattern, path)), None)


class History:
    def __init__(self, repo, rev="HEAD"):
        self.repo, self.rev = str(repo), rev
        self.records, self.latest, self.owner, self.external = [], {}, {}, set()
        self.trees = set()

    def git(self, *args, text=True):
        return subprocess.run(["git", "-C", self.repo, *args], capture_output=True, text=text, check=True).stdout

    def blob(self, commit, path):
        return self.git("show", f"{commit}:{path}", text=False)

    def project(self):
        commits = self.git("rev-list", "--reverse", "--topo-order", self.rev).split()
        self.records.append(Tree(EXTERNAL, EXTERNAL, EXTERNAL))
        for commit in commits:
            parents = self.git("rev-list", "--parents", "-n", "1", commit).split()[1:]
            if len(parents) > 1:
                continue  # a merge repeats the changes its branch commits already contributed
            changes = []
            for row in self.git("diff-tree", "--root", "-r", "-M", "--name-status", "--no-commit-id", commit).splitlines():
                status, *paths = row.split("\t")
                if kind_of(paths[-1]):
                    changes.append((status[0], paths, kind_of(paths[-1])))
            bulk = len(changes) > 50
            # Cards cite evidence and pages of the same commit, so they are projected after them.
            for status, paths, kind in sorted(changes, key=lambda item: item[2] == "decision_file"):
                if kind == "decision_file":
                    self.cards(commit, parents[0] if parents else None, paths[-1], bulk)
                elif status == "D":
                    self.retire(paths[0], commit)
                else:
                    self.change(commit, status, paths, kind, bulk)
        return self.records

    def tree(self, kind):
        tree_id = f"kernelwiki/{kind}"
        if tree_id not in self.trees:
            self.trees.add(tree_id)
            gate = GATES.get(kind)
            self.records.append(Tree(tree_id, f"knowledge:{kind}", f"kernelwiki:{kind}",
                                     params={"gate": gate} if gate else {}))
        return tree_id

    def pinned(self, repo, sha):
        node_id = f"{EXTERNAL}/{repo}@{sha}"
        if node_id not in self.external:
            self.external.add(node_id)
            self.records.append(Node(node_id, EXTERNAL, None, 0, sha))
        return node_id

    def emit(self, kind, key, commit, op, parent, state, uses, gaps, bulk):
        tree = self.tree(kind)
        node_id = f"kernelwiki:{key}@{commit[:12]}"
        expansion = Expansion(f"{node_id}#{op}", tree, op, (parent,) if parent else (), uses=tuple(uses),
                              facets={"kernelwiki": {"commit": commit, "bulk_import": bulk}})
        self.records += [expansion, Node(node_id, tree, expansion.id, 0, state)]
        self.records += [GapRecord(node_id, Gap(f"{node_id}#{question}", classification, question, reasons))
                         for question, classification, reasons in gaps]
        self.latest[key], self.owner[node_id] = node_id, tree
        return node_id

    def retire(self, key, commit):
        node_id = self.latest.pop(key, None)
        if node_id:
            self.records.append(Decide(self.owner[node_id], node_id, "retired", f"deleted in {commit[:12]}"))

    def change(self, commit, status, paths, kind, bulk):
        path, previous = paths[-1], self.latest.pop(paths[0], None) if status == "R" else self.latest.get(paths[-1])
        payload = self.blob(commit, path)
        state = hashlib.sha256(payload).hexdigest()
        evidence = (EvidenceRef(f"git:{commit}:{path}", state),)
        text = payload.decode("utf-8", errors="replace")
        uses, gaps = [], []
        if kind == "evidence_set":
            provenance = (yaml.safe_load(text) or {}).get("provenance", {})
            sha = str(provenance.get("aiter_commit", ""))
            if re.fullmatch(r"[0-9a-f]{40}", sha):
                uses.append(Use(self.pinned("ROCm/aiter", sha), "source", evidence))
            if provenance.get("aiter_dirty_sources"):
                gaps.append(("evidence extracted from a dirty source", "resolvable_local",
                             tuple(map(str, provenance["aiter_dirty_sources"]))))
        elif kind == "bundle":
            provenance = (yaml.safe_load(text) or {}).get("provenance", {}) or {}
            sha, repo = str(provenance.get("commit", "")), str(provenance.get("repository", ""))
            if re.fullmatch(r"[0-9a-f]{12,40}", sha) and "github.com/" in repo:
                uses.append(Use(self.pinned(repo.split("github.com/")[-1].removesuffix(".git"), sha), "source", evidence))
        else:
            uses += [Use(self.pinned(repo, sha), "source", evidence) for repo, sha in sorted(set(PIN.findall(text)))
                     if not repo.endswith("/KernelWiki")]
        if kind == "learned_card":
            gaps += self.card_gaps(text)
        op = ("revise" if previous else "ingest" if uses and kind != "learned_card"
              else "distill" if kind == "learned_card" else "author")
        self.emit(kind, path, commit, op, previous, state, uses, gaps, bulk)

    @staticmethod
    def card_gaps(text):
        block = text.split("---\n", 2)[1] if text.startswith("---\n") else ""
        try:
            fields, valid = yaml.safe_load(block) or {}, True
        except yaml.YAMLError:
            fields, valid = dict(re.findall(r"^([a-z_]+):\s*(.*)$", block, re.M)), False
        gaps = [] if valid else [("front matter is not valid YAML", "resolvable_local", ("yaml.safe_load failed",))]
        if not isinstance(fields, dict) or not RUN_KEYS & set(fields):
            origin = fields.get("origin", "none") if isinstance(fields, dict) else "none"
            gaps.append(("no structured run reference", "needs_online_control", (f"origin: {origin}",)))
        return gaps

    def cards(self, commit, parent, path, bulk):
        def load(rev):
            try:
                data = yaml.safe_load(self.blob(rev, path)) if rev else None
            except subprocess.CalledProcessError:
                data = None
            return {str(card["id"]): card for card in (data or {}).get("cards", []) if isinstance(card, dict)}
        payload = self.blob(commit, path)
        state = hashlib.sha256(payload).hexdigest()
        evidence = (EvidenceRef(f"git:{commit}:{path}", state),)
        before, after = load(parent), load(commit)
        owners = self.evidence_owners(commit)
        for card_id in sorted(before.keys() | after.keys()):
            key, card = f"{path}#{card_id}", after.get(card_id)
            if before.get(card_id) == card:
                continue
            if card is None:
                self.retire(key, commit)
                continue
            cited = (*card.get("source_evidence", []), *card.get("shipped_evidence", []))
            missing = [ref for ref in cited if ref not in owners]
            pages = [str(reference) for reference in card.get("references", [])]
            unresolved = [reference for reference in pages if reference.split(":")[0] not in self.latest]
            targets = {owners[ref] for ref in cited if ref in owners}
            targets |= {self.latest[reference.split(":")[0]] for reference in pages if reference not in unresolved}
            uses, gaps = [Use(target, "source", evidence) for target in sorted(targets)], []
            if missing:
                gaps.append(("cited evidence id not found", "resolvable_local", tuple(missing)))
            if unresolved:
                gaps.append(("cited page not found at this commit", "resolvable_local", tuple(unresolved)))
            previous = self.latest.get(key)
            self.emit("decision_card", key, commit, "revise" if previous else "author", previous, state, uses, gaps, bulk)

    def evidence_owners(self, commit):
        """Evidence id -> the evidence-set node that holds it at this commit."""
        owners = {}
        for path in self.git("ls-tree", "-r", "--name-only", commit, "corpus/evidence").split():
            if path in self.latest:
                for evidence_id in EVIDENCE_ID.findall(self.blob(commit, path).decode("utf-8", errors="replace")):
                    owners[evidence_id] = self.latest[path]
        return owners


def kernelwiki_trees(repo, rev="HEAD"):
    return History(repo, rev).project()


def resolved_commit(repo, rev="HEAD"):
    return History(repo, rev).git("rev-parse", rev).strip()


def knowledge_index(records):
    """Content SHA-256 -> latest KernelWiki file node with that content; GEAK reads whole files, not cards."""
    index = {}
    for record in records:
        if isinstance(record, Node) and record.id.startswith("kernelwiki:") and "#" not in record.id and record.state:
            index[record.state] = record.id
    return index
