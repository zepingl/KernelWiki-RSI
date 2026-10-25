"""Interpret verified GEAK evidence; observations, method reviews and effects stay distinct."""

from dataclasses import asdict, replace

from kernelwiki.knowledge import EvidenceRef

from ..discovery import Measurement
from ..evidence import evidence_refs, verify
from .model import Gap, Relation, ReverseReport
from .sources.archive import AuditReader
from .sources.capture import capture_evidence


def effect_summary(*, candidate_source=None, installed_source=None, parent_source=None,
                   parent_gain=None, threshold=None, director_status="unknown", method_status="unknown"):
    retained = None if not candidate_source or not installed_source else candidate_source == installed_source
    return {"method_status": method_status, "parent_relative_gain": parent_gain,
            "threshold_met": None if parent_gain is None or threshold is None else parent_gain >= threshold,
            "retained": retained,
            "rolled_back": (None if not all((candidate_source, installed_source, parent_source)) else
                            candidate_source != parent_source and installed_source == parent_source),
            "director_status": director_status, "independent_knowledge_causal_gain": None}


def _parent_measurement(subject):
    candidate = subject.attempt.measurement
    parent = subject.parent.measurement if subject.parent else None
    if (parent and candidate.status == parent.status == "qualified" and candidate.contract == parent.contract):
        return parent, candidate.primary_score / parent.primary_score - 1
    return None, None


def _method_reviews(knowledge, changes):
    observed = {row.get(key) for row in changes for key in ("before_fingerprint", "after_fingerprint")}
    reviews = []
    for item in knowledge:
        review = item.get("review", {})
        status, selected = review.get("disposition", ""), review.get("selected_candidate_fingerprint")
        if (any(word in status for word in ("adoption", "implemented", "method_"))
                and (selected is None or selected in observed)):
            reviews.append({"status": status, "scope": review.get("method_scope", review.get("method_route", "")),
                            "rationale": review.get("implementation_and_validation", review.get("limits", "")),
                            "evidence": item["review_evidence"], "certainty": "review"})
    return reviews


def _archive_trace(subject, reader, documents, coverages):
    attempt = subject.attempt
    findings = reader.trace_evidence(subject, documents)
    inputs, changes, knowledge = (findings[key] for key in ("original_inputs", "source_changes", "knowledge"))
    relations = []
    if attempt.parent_source and attempt.audit:
        relations.append(Relation(attempt.parent_source, attempt.id, "implementation_inherited", attempt.audit))
    for item in knowledge:
        delivery = item["delivery"]
        if delivery["certainty"] != "observed":
            continue
        kind = "body_delivered" if item["kind"] == "direct_read" else "direction_forwarded"
        refs = [EvidenceRef.from_dict(item["evidence"]), EvidenceRef.from_dict(delivery["source"])]
        if delivery.get("returned"):
            refs.append(EvidenceRef.from_dict(delivery["returned"]))
        relations.append(Relation(delivery["source"]["sha256"], attempt.id, kind, tuple(refs)))
    reviews = _method_reviews(knowledge, changes)
    for review in reviews:
        relations.append(Relation(attempt.id, review["status"], "method_adopted",
                                  (EvidenceRef.from_dict(review["evidence"]),), "review"))
    for row in findings["internal_measurements"]:
        value = Measurement.from_dict(row["measurement"])
        if value.status == "qualified":
            relations.append(Relation(value.source, value.contract, "measured", value.evidence))
    installed = findings["installed_observations"]
    for item in installed:
        relations.append(Relation(item["source"], inputs["CANONICAL"], "installed",
                                  (EvidenceRef.from_dict(item["evidence"]),)))
    parent, gain = _parent_measurement(subject)
    if gain is None and "parent_measurement" in coverages:
        paired = reader.independent_pair(subject, inputs)
        if paired:
            parent, candidate = paired
            gain = candidate.primary_score / parent.primary_score - 1
            findings["recovered_parent_pair"] = {"parent": asdict(parent), "candidate": asdict(candidate)}
    candidate = attempt.candidate_source
    retained = attempt.state.fingerprint if attempt.state else None
    restored = bool(retained and retained == attempt.parent_source and candidate != retained)
    recovered = findings["recovered_sources"]
    if recovered.get("candidate") and recovered.get("installed"):
        candidate, retained = recovered["candidate"]["fingerprint"], recovered["installed"]["fingerprint"]
        installed.append({"source": retained, "scope": "saved_best_at_stop", "evidence": recovered["installed"]["evidence"][0]})
        relations.append(Relation(retained, "saved_best_at_stop", "installed",
                                  tuple(EvidenceRef.from_dict(item) for item in recovered["installed"]["evidence"])))
    installed_source = retained if restored else installed[-1]["source"] if installed else None
    effect = effect_summary(candidate_source=candidate, installed_source=installed_source,
                            parent_source=attempt.parent_source, parent_gain=gain,
                            threshold=findings["threshold"]["value"],
                            method_status=reviews[0]["status"] if reviews else "unknown")
    effect["installed_scope"] = ("saved_best_at_stop" if recovered else
                                 "attempt_restored_parent" if restored else "latest_observed_canonical")
    gaps = []
    if attempt.disposition == "trace_only":
        gaps.append(Gap("input_provenance", "resolvable_local",
                        "Recover the exact original source AND dispatched context.", attempt.gaps))
    if gain is None:
        gaps.append(Gap("parent_measurement", "resolvable_local",
                        "Find comparable independent measurements of the direct parent and retained candidate."))
    if not reviews:
        gaps.append(Gap("method_role", "needs_method_review" if findings["diffs"] and knowledge else "resolvable_local",
                        "Which delivered method was adopted, rejected, confirmed, or reverted by these edits?"))
    if not installed and not restored:
        gaps.append(Gap("installation", "resolvable_local",
                        "Find the actual saved source boundary, separately from the attempted candidate."))
    gaps.extend((Gap("director", "resolvable_local",
                     "Find final Director acceptance bound to the installed source and evaluator."),
                 Gap("knowledge_causality", "needs_online_control",
                     "A separate isolated A/B is required for knowledge's independent causal gain.")))
    findings.update(method_reviews=reviews, parent_measurement=asdict(parent) if parent else None, effect=effect,
                    limits=["Internal edits and measurements are not extra free replay nodes.",
                            "Source recovery does not recover missing dispatch context or requalify a trace-only attempt.",
                            "Observed delivery, method review, measured increment, saving and Director are separate.",
                            "Retrospective comparisons and method reviews do not prove isolated knowledge gain."])
    return ReverseReport(subject, findings, relations, gaps)


def _capture_trace(subject):
    attempt = subject.attempt
    findings = capture_evidence(subject)
    request, disposition = findings["original_inputs"], findings.pop("disposition")
    _, gain = _parent_measurement(subject)
    effect = effect_summary(candidate_source=attempt.candidate_source,
                            installed_source=disposition["installed_source"]["fingerprint"],
                            parent_source=request["source"]["fingerprint"], parent_gain=gain,
                            director_status=disposition["director_status"])
    effect["installed_scope"] = disposition["installed_scope"]
    relations = [Relation(request["source"]["fingerprint"], attempt.id, "parent", (attempt.input,))]
    if attempt.measurement.status == "qualified":
        relations.append(Relation(attempt.measurement.source, attempt.measurement.contract,
                                  "measured", attempt.measurement.evidence))
    gaps = (Gap("knowledge_delivery", "unrecoverable_history",
                "Captured prompts and plans prove dispatch, not the bodies returned by Agent file reads."),
            Gap("method_role", "needs_method_review" if findings["diffs"] and findings["plan"] else "unrecoverable_history",
                "Which planned method is actually implemented or reverted by the captured source diff?"),
            Gap("knowledge_causality", "needs_online_control",
                "Scheduling replay does not isolate reference-knowledge causality."))
    findings.update(effect=effect, knowledge=[], method_reviews=[], measurement=asdict(attempt.measurement),
                    limits=["Candidate is retained as a generation artifact, not installed into the user's source.",
                            "No final Director or independent knowledge-causality experiment ran."])
    return ReverseReport(subject, findings, relations, gaps)


def trace(subject, *, reader=None, documents=(), coverages=None):
    for item in subject.sources:
        verify(item)
    coverages = coverages or {}
    report = (_capture_trace(subject) if subject.attempt.campaign == "online" else
              _archive_trace(subject, reader or AuditReader.for_subject(subject), documents, coverages))
    gaps = []
    for gap in report.gaps:
        coverage = coverages.get(gap.id)
        if coverage:
            classification = ("unrecoverable_history" if coverage.status == "complete"
                              and gap.classification == "resolvable_local" else gap.classification)
            gap = replace(gap, classification=classification, coverage=coverage)
        gaps.append(gap)
    return ReverseReport(subject, report.findings, report.relations, gaps)


def investigation_input(report):
    value = report.to_trace()
    return {"attempt": {key: value["attempt"][key] for key in
                        ("id", "kernel", "parent_source", "candidate_source", "disposition", "gaps")},
            **{key: value.get(key) for key in
               ("original_inputs", "diffs", "knowledge", "method_reviews", "plan", "boundaries", "agent_inputs",
                "measurement", "internal_measurements", "reviewed_comparisons", "recovered_sources",
                "relations", "effect", "gaps")},
            "allowed_evidence": [asdict(item) for item in evidence_refs(value)]}
