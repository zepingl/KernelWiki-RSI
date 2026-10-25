"""Deterministic per-gap investigation with explicit coverage and bounded model review."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path

from kernelwiki.knowledge import EvidenceRef, KnowledgeError

from ..evidence import artifact_hashes, evidence_refs, write_json
from .analyst import analyze, apply_review, method_evidence
from .sources.archive import AuditReader
from .trace import trace

PRIORITY = {"installation": 0, "parent_measurement": 1, "input_provenance": 2, "method_role": 3, "director": 4}


def investigate(subject, output, *, max_steps=4, model_calls=0, agent=None, reader=None, read_limit=64):
    if (type(max_steps) is not int or not 1 <= max_steps <= 16 or type(model_calls) is not int
            or not 0 <= model_calls <= max_steps or type(read_limit) is not int or read_limit < 1):
        raise KnowledgeError("investigation needs explicit bounded step, model-call and document-read budgets")
    if reader is None and subject.attempt.campaign != "online":
        reader = AuditReader.for_subject(subject)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = trace(subject, reader=reader)
    write_json(destination / "initial.json", report.to_dict())
    documents, coverages, steps, reviews, review_inputs = {}, {}, [], [], []
    action_counts, calls = Counter(), 0
    for index in range(1, max_steps + 1):
        candidates = []
        for gap in report.gaps:
            coverage = coverages.get(gap.id)
            if gap.classification == "resolvable_local" and reader and (coverage is None or coverage.remaining):
                candidates.append((gap, "read_related_original_archive_evidence"))
            elif (gap.classification == "needs_method_review" and calls < model_calls
                  and method_evidence(report) not in review_inputs):
                candidates.append((gap, "evidence_bounded_method_review"))
        if not candidates:
            break
        gap, action = min(candidates, key=lambda item: (action_counts[item[0].id], PRIORITY.get(item[0].id, 5)))
        action_counts[gap.id] += 1
        before = set(evidence_refs(report.to_dict()))
        step = {"step": index, "question": gap.to_dict(), "action": action,
                "gaps_before": [item.to_dict() for item in report.gaps]}
        if action == "read_related_original_archive_evidence":
            found, coverage = reader.search(subject, gap, report.findings["original_inputs"],
                                             previous=coverages.get(gap.id), limit=read_limit)
            coverages[gap.id] = coverage
            for document in found:
                documents[EvidenceRef.from_dict(document["evidence"])] = document
            report = trace(subject, reader=reader, documents=tuple(documents.values()), coverages=coverages)
            for response, review_ref in reviews:
                report = apply_review(report, response, review_ref)
            step["coverage"] = coverage.to_dict()
            inspected = coverage.inspected
        else:
            review_inputs.append(method_evidence(report))
            calls += 1
            report, response, review_ref = analyze(report, gap, destination / f"method-{index}", agent=agent)
            reviews.append((response, review_ref))
            step["coverage"] = {"scope": ["supplied source, plan, diff and receipt evidence"], "status": "review",
                                "inspected": response["evidence"], "remaining": [], "unavailable": []}
            inspected = ()
        after = tuple(dict.fromkeys((*evidence_refs(report.to_dict()), *inspected)))
        step.update(gaps_after=[item.to_dict() for item in report.gaps],
                    new_evidence=[asdict(item) for item in after if item not in before])
        steps.append(step)
        write_json(destination / f"step-{index}.json", step)
    result = {**report.to_dict(), "status": "audited_with_limits" if report.gaps else "resolved",
              "steps": steps, "model_calls": calls, "max_steps": max_steps,
              "coverage": {key: value.to_dict() for key, value in coverages.items()},
              "stop_reason": "step_budget" if len(steps) == max_steps else "no_new_authorized_evidence_action",
              "artifacts": artifact_hashes(destination)}
    write_json(destination / "result.json", result)
    return result
