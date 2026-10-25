"""Evidence-bounded method-role analysis; it cannot promote historical facts."""

from dataclasses import asdict
from pathlib import Path

from kernelwiki.knowledge import EvidenceRef, KnowledgeError, strict_fields

from ..evidence import evidence_refs, reference, verify, write_json
from ..runtime import AgentRole, RuntimeAgent
from .model import Relation, ReverseReport
from .trace import investigation_input

METHOD_STATUSES = ("proposed", "confirmed", "excluded", "adopted", "reverted", "unknown")
ROLE = AgentRole({"type": "object", "required": ["method_status", "rationale", "evidence"],
                  "properties": {"method_status": {"enum": list(METHOD_STATUSES)},
                                 "rationale": {"type": "string"}, "evidence": {"type": "array"}}},
                 Path(__file__).parent / "roles/analyst.md")


def method_evidence(report):
    findings = report.findings
    return {key: findings.get(key) for key in ("original_inputs", "diffs", "knowledge", "plan", "boundaries",
                                               "agent_inputs", "measurement", "internal_measurements", "recovered_sources")}


def apply_review(report, response, review_ref):
    strict_fields(response, {"method_status", "rationale", "evidence"}, {"method_status", "rationale", "evidence"})
    if response["method_status"] not in METHOD_STATUSES:
        raise KnowledgeError("invalid method review role")
    if (not isinstance(response["rationale"], str) or not response["rationale"].strip()
            or not isinstance(response["evidence"], list) or not response["evidence"]):
        raise KnowledgeError("method review must cite supplied evidence and explain its limits")
    allowed = set(evidence_refs(report.to_trace()))
    for value in response["evidence"]:
        item = EvidenceRef.from_dict(value)
        if item not in allowed:
            raise KnowledgeError("method review invented or crossed its evidence scope")
        verify(item)
    if review_ref.kind != "review":
        raise KnowledgeError("method-role judgments must remain reviews")
    verify(review_ref)
    findings = report.findings
    findings.setdefault("method_reviews", []).append({**response, "certainty": "review", "review": asdict(review_ref)})
    relations, gaps = report.relations, report.gaps
    if response["method_status"] != "unknown":
        findings["effect"]["method_status"] = response["method_status"]
        gaps = tuple(item for item in gaps if item.id != "method_role")
        kind = "method_adopted" if response["method_status"] == "adopted" else "method_reviewed"
        relations = (*relations, Relation(report.subject.attempt.id, response["method_status"], kind, (review_ref,), "review"))
    return ReverseReport(report.subject, findings, relations, gaps)


def analyze(report, gap, directory, *, agent=None):
    directory = Path(directory)
    response = (agent or RuntimeAgent(roles={"reverse_analyst": ROLE}))(
        "reverse_analyst", {"question": gap.to_dict(), "evidence": investigation_input(report)}, directory / "agent")
    path = directory / "response.json"
    write_json(path, response)
    review_ref = reference(path, kind="review")
    return apply_review(report, response, review_ref), response, review_ref
