"""Normalize original audit invocations, source boundaries and evaluator receipts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median

from kernelwiki.knowledge import (
    EvidenceRef,
    KernelState,
    KnowledgeError,
    digest,
    finite,
)

from ...discovery import (
    Attempt,
    ContextSnapshot,
    DiscoveryNode,
    DiscoveryWorld,
    GenerationSpec,
    Measurement,
    SourceState,
    reconstruct,
)
from ...evidence import read_json, read_jsonl, reference, verify
from ..model import Coverage, TraceSubject
from .archive_tree import lane_trees


@dataclass(frozen=True)
class Dispatch:
    round: str
    order: int
    state: KernelState
    generation: GenerationSpec
    baseline: Measurement
    parent: SourceState
    input: ContextSnapshot


def _tool_result(value, tool_id):
    if isinstance(value, dict):
        if value.get("type") == "tool_result" and value.get("tool_use_id") == tool_id:
            content = value.get("content", "")
            if isinstance(content, str):
                return content
            return "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
        for child in value.values():
            result = _tool_result(child, tool_id)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _tool_result(child, tool_id)
            if result is not None:
                return result
    return None


def _body_delivery(run, row):
    card, event = row.get("card", {}), row.get("evidence", {})
    path = run / card.get("snapshot_path", "")
    event_path = run / event.get("event_file", "")
    if not path.is_file() or not event_path.is_file() or not event.get("result_event_line"):
        return {"certainty": "unknown", "reason": "missing source or returned-body event"}
    source_ref = reference(path)
    if source_ref.sha256 != card.get("snapshot_sha256"):
        return {"certainty": "unknown", "reason": "source document version mismatch"}
    number = event["result_event_line"]
    record = next((value for index, value in enumerate(read_jsonl(event_path), 1) if index == number), None)
    returned = _tool_result(record, event.get("tool_use_id"))
    if returned is None:
        return {"certainty": "unknown", "reason": "raw result does not match the claimed tool invocation"}
    source_lines, result_lines = path.read_text().splitlines(), returned.splitlines()
    ranges = card.get("ranges", [])
    matched = bool(ranges) and all(
        1 <= item["source_start_line"] <= item["source_end_line"] <= len(source_lines)
        and 1 <= item["result_start_line"] <= item["result_end_line"] <= len(result_lines)
        and source_lines[item["source_start_line"] - 1:item["source_end_line"]]
        == result_lines[item["result_start_line"] - 1:item["result_end_line"]]
        for item in ranges)
    return {"certainty": "observed" if matched else "unknown", "source": asdict(source_ref),
            "returned": asdict(reference(event_path, start_line=number, end_line=number)),
            "ranges": ranges, "excerpts": ["\n".join(source_lines[item["source_start_line"] - 1:
                                                            item["source_end_line"]]) for item in ranges] if matched else [],
            "reason": "exact frozen excerpts returned" if matched else "excerpt mismatch"}


def result_objects(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                yield value
        except json.JSONDecodeError:
            continue


def threshold_from_inputs(inputs):
    prompt = inputs.get("DIRECTION", {}).get("prompt", "")
    matches = list(re.finditer(r"(?:promotion (?:bar|needs|threshold)|promote only if the primary gain is)"
                              r"[^\n]{0,180}?(?:\+|>\s*|at least\s+)(\d+(?:\.\d+)?)\s*%", prompt, re.IGNORECASE))
    values = {float(match[1]) / 100 for match in matches}
    return {"value": next(iter(values)) if len(values) == 1 else None,
            "excerpts": [match[0] for match in matches], "scope": "numeric increment only, not all promotion gates"}


def prompt_inputs(text):
    section = text.split("## Inputs\n", 1)
    if len(section) != 2:
        raise KnowledgeError("delivered prompt has no structured Inputs section")
    result = {}
    for match in re.finditer(r"^- ([A-Za-z_][A-Za-z_0-9]*): (.*)$", section[1], re.MULTILINE):
        key, value = match.groups()
        if key in result:
            raise KnowledgeError("ambiguous delivered input field")
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    if not all(result.get(key) for key in ("KERNEL_PATH", "CANONICAL", "DIRECTION")):
        raise KnowledgeError("delivered input lacks parent or direction")
    return result


class AuditReader:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self._references = {}
        self._sources = {}
        self._measurements = {}
        self.changes = defaultdict(list)
        self.benchmarks = defaultdict(list)
        self.verifiers = defaultdict(list)
        self.calls = {}
        for change in read_jsonl(self.run / "index/source_changes.jsonl"):
            self.changes[change.get("call_id")].append(change)
        for row in read_jsonl(self.run / "index/benchmarks.jsonl"):
            self.benchmarks[row.get("candidate_fingerprint")].append(row)
        for call in read_jsonl(self.run / "index/agent_calls.jsonl"):
            if call["call_id"] in self.calls:
                raise KnowledgeError("duplicate invocation in original audit inventory")
            self.calls[call["call_id"]] = call
        archive = self.run / "recording/records/tasks"
        for path in sorted(archive.glob("*/launches/*/agents/*/invocation.json")):
            call = read_json(path)
            existing = self.calls.get(call["call_id"])
            if existing and (self.run / existing["path"] / "invocation.json").is_file():
                continue
            self.calls[call["call_id"]] = {**call, "path": str(path.parent.relative_to(self.run))}
        for call in self.calls.values():
            context = call.get("context", {})
            if context.get("phase") == "verify" and context.get("PATCH"):
                self.verifiers[str(Path(context["PATCH"]).parent)].append(call)

    def inventory(self, tasks, excluded_tasks=()):
        if set(tasks) & set(excluded_tasks):
            raise KnowledgeError("an optimization task cannot be both selected and excluded")
        for call in self.calls.values():
            context = call.get("context", {})
            if str(context.get("phase", "")).lower() != "optimize" or not (
                    context.get("role") in {"engineer", "deep_engineer"}
                    or "DIRECTION" in context or "SPECIALTY" in context):
                continue
            if call.get("task") not in tasks and call.get("task") not in excluded_tasks:
                raise KnowledgeError(f"optimization task lacks a kernel assignment: {call.get('task')}")
            yield tasks.get(call["task"]), call, self.run / call["path"] / "invocation.json"

    def ref(self, path, **kwargs):
        path = Path(path)
        if not path.is_absolute():
            path = self.run / path
        key = (str(path), tuple(sorted(kwargs.items())))
        if key not in self._references:
            self._references[key] = replace(self.ref(path), **kwargs) if kwargs else reference(path)
        return self._references[key]

    def source(self, fingerprint):
        if fingerprint in self._sources:
            return self._sources[fingerprint]
        path = self.run / "index/candidate_manifests" / f"{fingerprint}.json"
        manifest = read_json(path)
        actual = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
        if manifest.get("fingerprint") != fingerprint or actual != fingerprint:
            raise KnowledgeError("source manifest fingerprint mismatch")
        for entry in manifest["files"].values():
            blob = self.run / "objects" / entry["sha256"][:2] / entry["sha256"]
            if self.ref(blob).sha256 != entry["sha256"]:
                raise KnowledgeError("source blob fingerprint mismatch")
        self._sources[fingerprint] = self.ref(path)
        return self._sources[fingerprint]

    def measurement(self, row):
        path = row["summary_path"]
        if path in self._measurements:
            return self._measurements[path]
        directory = (self.run / path).parent
        summary = read_json(directory / "summary.json")
        before = read_json(directory / "before.json")
        refs = [self.ref(directory / "summary.json"), self.ref(directory / "before.json")]
        unknown = Measurement("unknown", evidence=tuple(refs))
        self._measurements[path] = unknown
        processes = summary.get("processes", [])
        if processes and all(process.get("source_unchanged") is True and process.get("contract_unchanged") is True
                             and (process.get("report_result") or {}).get("correctness_pass") is False
                             for process in processes):
            failed = Measurement("failed", summary.get("candidate_fingerprint"), evidence=tuple(refs),
                                 reason="correctness failed in every process")
            self._measurements[path] = failed
            return failed
        if not all(summary.get(key) is True for key in
                   ("two_independent_processes", "measurement_valid", "full_task_coverage")):
            return unknown
        if (summary.get("default_source_promotion", {}).get("eligible") is not True
                or before.get("candidate_runtime", {}).get("overrides")):
            return unknown
        source = before.get("candidate", {}).get("fingerprint")
        if source != summary.get("candidate_fingerprint") or not before.get("contracts"):
            return unknown
        refs.append(self.source(source))
        processes = summary.get("processes", [])
        if len(processes) < 2 or len({process.get("pid") for process in processes}) != len(processes):
            return unknown
        latencies, contracts = [], []
        for index, process in enumerate(processes, 1):
            process_dir = directory / f"process_{index}"
            receipt = read_json(process_dir / "receipt.json")
            invocation = read_json(process_dir / "invocation.json")
            if receipt != process or invocation.get("candidate") != source:
                return unknown
            if (receipt.get("exit_code") != 0 or receipt.get("source_unchanged") is not True
                    or receipt.get("contract_unchanged") is not True):
                return unknown
            report = receipt.get("report_result") or {}
            if not report.get("runtime_fingerprint"):
                return unknown
            if not all(report.get(key) is True for key in
                       ("measurement_valid", "correctness_pass", "performance_scope_complete")):
                return unknown
            cases = report.get("per_case") or []
            if (not cases or len(cases) != report.get("expected_case_count")
                    or len({case.get("name") for case in cases}) != len(cases)):
                return unknown
            cases = sorted(cases, key=lambda case: case["name"])
            numerator, denominator = 0.0, 0.0
            for case in cases:
                if case.get("measurement_valid") is not True:
                    return unknown
                for key in ("calls", "baseline_ms", "optimized_ms"):
                    if finite(case[key]) <= 0:
                        return unknown
                numerator += case["calls"] * case["baseline_ms"]
                denominator += case["calls"] * case["optimized_ms"]
            latencies.append(denominator)
            contracts.append(digest({"files": before["contracts"], "entry": before.get("entry"),
                                     "runtime": report.get("runtime_fingerprint"),
                                     "cases": [(case["name"], case["calls"], case.get("timer"))
                                               for case in cases],
                                     "metric": "inverse-call-weighted-latency-v2"}))
            refs.extend((self.ref(process_dir / "receipt.json"), self.ref(process_dir / "invocation.json")))
        if len(set(contracts)) != 1:
            return unknown
        result = Measurement("qualified", source, 1 / median(latencies), contracts[0], tuple(refs))
        self._measurements[path] = result
        return result

    def baseline(self, fingerprint, canonical, before):
        """Latest qualified measurement of the canonical workspace taken before a dispatch."""
        rows = sorted((row for row in self.benchmarks[fingerprint]
                       if row.get("workspace") == canonical and row.get("time") and row["time"] < before),
                      key=lambda row: row["time"], reverse=True)
        return next((result for row in rows if (result := self.measurement(row)).status == "qualified"), None)

    def candidate_measurement(self, fingerprint, contract):
        """A result belongs to a source and contract, whoever ran it: the lower median of qualified runs,
        or a correctness failure that no qualified run contradicts."""
        results = []
        for row in self.benchmarks[fingerprint]:
            try:
                results.append(self.measurement(row))
            except OSError:  # a run whose receipts are gone supplies no evidence; it must not void the attempt
                continue
        qualified = sorted((item for item in results if item.status == "qualified" and item.contract == contract),
                           key=lambda item: item.primary_score)
        failed = [item for item in results if item.status == "failed"]
        if qualified and not failed:
            return qualified[(len(qualified) - 1) // 2]
        if failed and not qualified:
            return failed[0]
        return Measurement("unknown", fingerprint, reason="conflicting correctness evidence" if failed
                           else "no compatible independent candidate measurement")

    def initial_and_final(self, call, inputs):
        changes = sorted((row for row in self.changes[call["call_id"]]
                          if row.get("workspace") == inputs["KERNEL_PATH"]
                          and not row.get("asynchronous")),
                         key=lambda row: (row["interval_start"], row["event_line"]))
        if not changes:
            raise KnowledgeError("missing private-workspace source observations")
        first, last = changes[0], changes[-1]
        before = read_json(self.run / first["pre_observation"])
        after = read_json(self.run / first["post_observation"])
        final_snapshot = read_json(self.run / last["post_observation"])
        if before.get("snapshot_complete") is False or after.get("snapshot_complete") is False:
            raise KnowledgeError("incomplete initial source snapshot")
        parents = [state for state in before.get("source_states", [])
                   if state["workspace"] == inputs["CANONICAL"]]
        if len(parents) != 1:
            raise KnowledgeError("canonical parent is missing or ambiguous")
        initial = (first.get("after_fingerprint") if first.get("kind") == "workspace_created"
                   else first.get("before_fingerprint"))
        if initial != parents[0]["fingerprint"]:
            raise KnowledgeError("private workspace does not start at the observed parent")
        first_states = [state for state in after.get("source_states", [])
                        if state["workspace"] == inputs["KERNEL_PATH"]]
        last_states = [state for state in final_snapshot.get("source_states", [])
                       if state["workspace"] == inputs["KERNEL_PATH"]]
        if (len(first_states) != 1 or first_states[0]["fingerprint"] != first.get("after_fingerprint")
                or len(last_states) != 1 or last_states[0]["fingerprint"] != last.get("after_fingerprint")):
            raise KnowledgeError("source index disagrees with original boundary snapshots")
        parent_source = self.source(initial)
        final = last.get("after_fingerprint")
        refs = (self.ref(first["pre_observation"]), self.ref(first["post_observation"]),
                self.ref(last["post_observation"]), parent_source)
        if final:
            self.source(final)
        artifacts = {item["path"]: item["sha256"] for item in before.get("artifacts", [])}
        candidate = final
        if final == initial:
            candidate = next((row.get("after_fingerprint") for row in reversed(changes)
                              if row.get("after_fingerprint") not in {None, initial}), final)
        return initial, final, candidate, refs, artifacts

    def attempt(self, kernel, campaign, invocation_path):
        invocation_path = Path(invocation_path)
        invocation_ref = self.ref(invocation_path)
        call = read_json(invocation_path)
        input_ref = None
        parent = candidate = state = context = input_context = None
        audit = ()
        measurement = Measurement("unknown")
        try:
            input_ref = self.ref(invocation_path.parent / "delivered_prompt.txt")
            if input_ref.sha256 != call.get("delivered_prompt_sha256"):
                raise KnowledgeError("delivered prompt hash mismatch")
            inputs = prompt_inputs((invocation_path.parent / "delivered_prompt.txt").read_text())
            parent, retained, candidate, audit, artifacts = self.initial_and_final(call, inputs)
            state = SourceState(retained, (self.source(retained), *audit))
            measurement = Measurement("unknown", retained, reason="no compatible independent candidate measurement")
            versions = {}
            for key in ("COMMANDMENT", "codebase_context", "profiling_summary"):
                path = inputs.get(key)
                if not path:
                    continue
                if path not in artifacts:
                    raise KnowledgeError(f"missing initial context version: {key}")
                blob = self.run / "objects" / artifacts[path][:2] / artifacts[path]
                if self.ref(blob).sha256 != artifacts[path]:
                    raise KnowledgeError("context blob fingerprint mismatch")
                versions[key] = artifacts[path]
            shared = {key: value for key, value in inputs.items()
                      if key not in {"DIRECTION", "SPECIALTY", "GPU_ID", "KERNEL_PATH", "OUTPUT_DIR", "TARGET", "KK_REFS"}}
            input_context = digest({"inputs": shared, "versions": versions})
            context = ContextSnapshot(digest({"input": input_context, "attempt": call["call_id"],
                                               "retained": retained, "audit": [asdict_ref.sha256 for asdict_ref in audit]}),
                                      (input_ref, *audit), input_context)
            root_measurement = self.baseline(parent, inputs["CANONICAL"], call["time"])
            if root_measurement is None:
                raise KnowledgeError("no qualified parent measurement available before dispatch")
            measurement = self.candidate_measurement(retained, root_measurement.contract)
            argv = call["argv"]
            model = argv[argv.index("--model") + 1]
            effort = argv[argv.index("--effort") + 1]
            if "DIRECTION" not in inputs:
                raise KnowledgeError("missing original direction")
            order = re.fullmatch(r"engineer_(\d+)", Path(inputs["OUTPUT_DIR"]).name)
            if order is None:
                raise KnowledgeError("missing recorded dispatch order")
            kernel_state = {field: inputs[key] for field, key in (("operator", "KK_OPERATOR"),
                                                         ("backend", "KK_LANGUAGE"), ("gfx", "GFX"))
                     if isinstance(inputs.get(key), str) and inputs[key]}
            mode = "deep_explore" if inputs.get("SPECIALTY") == "deep_explore" else "normal"
            generation = GenerationSpec(mode, model, effort, digest({"inputs": shared, "versions": versions,
                                                                     "recorder": call.get("recorder_sources")}),
                                        root_measurement.contract)
            attempt = Attempt(call["call_id"], kernel, campaign, invocation_ref, input_ref,
                              parent, candidate, measurement, "replay_ready", (), audit,
                              state, context, input_context)
            dispatch = Dispatch(str(Path(inputs["OUTPUT_DIR"]).parent), int(order[1]),
                                KernelState.from_dict(kernel_state), generation, root_measurement,
                                SourceState(parent, (self.source(parent),)),
                                ContextSnapshot(input_context, (input_ref, *audit)))
            return attempt, dispatch
        except (OSError, ValueError, KeyError, IndexError) as error:
            return Attempt(call["call_id"], kernel, campaign, invocation_ref, input_ref, parent, candidate,
                           measurement, "trace_only", (str(error),), audit, state, context, input_context), None

    def knowledge(self, call_id, inputs):
        path = self.run / "index/knowledge_lineage.jsonl"
        if not path.exists():
            return []
        knowledge = []
        for number, row in enumerate(read_jsonl(path), 1):
            if row.get("call_id") != call_id:
                continue
            item = {"kind": row.get("kind"), "card": row.get("card"),
                    "evidence": asdict(self.ref(path, start_line=number, end_line=number)),
                    "delivery": {"certainty": "unknown", "reason": "no verified delivery path"}}
            if row.get("kind") == "direct_read":
                item["delivery"] = _body_delivery(self.run, row)
            elif row.get("kind") == "direction_transfer":
                transfer = row.get("direction_transfer", {})
                origins = transfer.get("origin_candidates", [])
                if len(origins) == 1:
                    origin = origins[0]
                    try:
                        result_ref = self.ref(origin["result_path"])
                        if result_ref.sha256 != origin["result_sha256"]:
                            raise KnowledgeError("original plan result changed")
                        plans = [value for value in result_objects(Path(result_ref.path).read_text())
                                 if isinstance(value.get("directions"), list)]
                        if len(plans) != 1:
                            raise KnowledgeError("ambiguous original plan")
                        direction = plans[0]["directions"][origin["direction_index"]]
                        delivered = inputs.get("DIRECTION", {})
                        if not delivered or any(direction.get(key) != value for key, value in delivered.items()):
                            raise KnowledgeError("original plan does not match delivered direction")
                        item["original_plan"] = plans[0]
                        item["delivery"] = {"certainty": "observed", "source": asdict(result_ref),
                                            "producer": origin["call_id"], "reason": "exact original direction delivered"}
                    except (OSError, ValueError, KeyError, IndexError) as error:
                        item["delivery"] = {"certainty": "unknown", "reason": str(error)}
            review = row.get("review")
            if review and review.get("call_id", call_id) == call_id:
                item["review"] = review
                item["review_evidence"] = asdict(self.ref(path, kind="review", start_line=number, end_line=number))
                if review.get("review_path"):
                    item["review_source"] = asdict(self.ref(review["review_path"], kind="review"))
            knowledge.append(item)
        return knowledge

    def recovered_sources(self, documents, inputs):
        result = {}
        workspace = inputs.get("KERNEL_PATH", "")
        if "/tasks/" not in workspace:
            return result
        task_root = "tasks/" + workspace.split("/tasks/", 1)[1].split("/", 1)[0]
        prefixes = {"candidate": "tasks/" + workspace.split("/tasks/", 1)[1] + "/kernel_src/",
                    "installed": task_root + "/state/best/kernel_src/"}
        for item in documents:
            name = item["document"].get("manual_retention_manifest") if isinstance(item["document"], dict) else None
            if not name:
                continue
            manifest_ref = self.ref(name)
            manifest = read_json(manifest_ref.path)
            for label, prefix in prefixes.items():
                captures = [capture for capture in manifest.get("captures", [])
                            if capture.get("source_path", "").startswith(prefix)]
                if not captures:
                    continue
                files, refs = {}, [manifest_ref]
                for capture in captures:
                    if capture.get("stable_during_capture") is not True or capture.get("archive_hash_verified") is not True:
                        raise KnowledgeError("unstable retained source cannot prove installation")
                    source_ref = self.ref(capture["object_path"])
                    if source_ref.sha256 != capture["sha256"]:
                        raise KnowledgeError("retained source object changed")
                    files[capture["source_path"][len(prefix):]] = {"sha256": source_ref.sha256,
                                                                  "size": capture["size"], "symlink": None}
                    refs.append(source_ref)
                identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
                result[label] = {"fingerprint": identity, "scope": "captured_kernel_src",
                                 "files": files, "evidence": [asdict(item) for item in refs]}
        return result

    @classmethod
    def for_subject(cls, subject):
        path = Path(subject.attempt.invocation.path)
        run = next((parent for parent in reversed(path.parents)
                    if (parent / "index/agent_calls.jsonl").is_file()), None)
        if run is None:
            raise KnowledgeError("raw audit run is unavailable")
        return cls(run)

    def subject(self, attempt_id, kernel, campaign=None):
        if attempt_id not in self.calls:
            raise KnowledgeError(f"unknown archive attempt: {attempt_id}")
        call = self.calls[attempt_id]
        path = self.run / call["path"] / "invocation.json"
        try:
            attempt, dispatch = self.attempt(kernel, campaign or self.run.name, path)
        except OSError as error:
            attempt = Attempt(attempt_id, kernel, campaign or self.run.name, self.ref("index/agent_calls.jsonl"),
                              None, None, None, Measurement("unknown"), "trace_only",
                              (f"original invocation unavailable: {error}",))
            dispatch = None
        parent = (DiscoveryNode(attempt.id + "/parent", None, None, dispatch.baseline, dispatch.input.evidence,
                                state=dispatch.parent, context=dispatch.input) if dispatch else None)
        return TraceSubject(attempt, parent)

    def search(self, subject, gap, inputs, *, previous=None, limit=64, max_bytes=16 * 1024 * 1024):
        if type(limit) is not int or limit < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise KnowledgeError("archive search needs positive document and byte limits")
        indexes = {
            "input_provenance": ("agent_calls", "source_changes", "artifact_catalog", "artifact_versions", "artifact_transfers"),
            "parent_measurement": ("benchmarks",),
            "installation": ("source_changes", "artifact_catalog", "artifact_versions", "artifact_transfers"),
            "method_role": ("knowledge_lineage", "knowledge_events", "tool_events"),
            "director": ("agent_calls", "tool_events"),
        }.get(gap.id)
        if indexes is None:
            raise KnowledgeError(f"no local archive evidence action for gap: {gap.id}")
        scope = tuple("index/" + name + ".jsonl" for name in indexes) + (
            "reviews/*.json", "monitor/*.json", "history/lineage_reviews.json")
        inspected = list(previous.inspected if previous else ())
        unavailable = list(previous.unavailable if previous else ())
        paths = []
        for name in indexes:
            path = self.run / "index" / (name + ".jsonl")
            if path.is_file():
                paths.append(path)
            elif str(path) not in unavailable:
                unavailable.append(str(path))
        task = self.calls.get(subject.attempt.id, {}).get("task", "")
        terms = {term for term in task.lower().split("_") if len(term) >= 3
                 and term not in {"glm52", "task", "bf16", "dsa", "fused"}}
        related_paths = sorted({*(self.run / "reviews").glob("*.json"),
                                *(self.run / "monitor").glob("*.json"),
                                *(self.run / "history").glob("lineage_reviews.json")})
        paths.extend(path for path in related_paths if path.name == "lineage_reviews.json"
                     or any(term in path.stem.lower() for term in terms))
        seen = {item.path for item in inspected} | set(unavailable)
        pending = [path for path in dict.fromkeys(paths) if str(path.resolve()) not in seen]
        output, workspace = inputs.get("OUTPUT_DIR", ""), inputs.get("KERNEL_PATH", "")
        terms = {subject.attempt.id, output, workspace, subject.attempt.parent_source, subject.attempt.candidate_source}
        if "/tasks/" in output:
            terms.add("tasks/" + output.split("/tasks/", 1)[1])
        terms.discard(None)
        terms.discard("")
        documents = []
        for path in pending[:limit]:
            if path.stat().st_size > max_bytes:
                unavailable.append(str(path.resolve()))
                continue
            item = self.ref(path, kind="observed" if path.suffix == ".jsonl" else "review")
            payload = verify(item).decode()
            inspected.append(item)
            try:
                if path.suffix == ".jsonl":
                    rows = [json.loads(line) for line in payload.splitlines() if line.strip()
                            and any(term in line for term in terms)]
                    document = {"archive_index": str(path.relative_to(self.run)), "records": rows}
                    matched = bool(rows)
                else:
                    document = json.loads(payload)
                    matched = any(term in payload for term in terms)
            except (ValueError, TypeError):
                unavailable.append(str(path.resolve()))
                continue
            if matched:
                documents.append({"document": document, "evidence": asdict(item)})
        remaining = tuple(str(path.resolve()) for path in pending[limit:])
        return tuple(documents), Coverage(scope, tuple(inspected), remaining, tuple(dict.fromkeys(unavailable)))

    def trace_evidence(self, subject, documents=()):
        attempt = subject.attempt
        invocation_path = Path(attempt.invocation.path)
        invocation = read_json(invocation_path) if invocation_path.name == "invocation.json" else self.calls[attempt.id]
        inputs = {}
        if attempt.input:
            payload = verify(attempt.input)
            if hashlib.sha256(payload).hexdigest() == invocation.get("delivered_prompt_sha256"):
                inputs = prompt_inputs(payload.decode())
        changes, diffs = [], []
        for row in self.changes[attempt.id]:
            if row.get("workspace") != inputs.get("KERNEL_PATH") or row.get("asynchronous"):
                continue
            original = read_json(self.run / row["post_observation"])
            if not any(state.get("workspace") == row["workspace"] and state.get("fingerprint") == row.get("after_fingerprint")
                       for state in original.get("source_states", [])):
                continue
            changes.append({**row, "evidence": asdict(self.ref(row["post_observation"]))})
            if row.get("diff_path") and (self.run / row["diff_path"]).is_file():
                diff_ref = self.ref(row["diff_path"])
                body = verify(diff_ref).decode()
                diffs.append({"before": row.get("before_fingerprint"), "after": row.get("after_fingerprint"),
                              "evidence": asdict(diff_ref), "excerpt": body[:16000], "truncated": len(body) > 16000})
        verifiers = {item["call_id"] for item in self.verifiers[inputs.get("OUTPUT_DIR")]}
        measurements = []
        for rows in self.benchmarks.values():
            for row in rows:
                if row.get("call_id") in {attempt.id, *verifiers}:
                    value = self.measurement(row)
                    measurements.append({"call_id": row.get("call_id"), "source": row.get("candidate_fingerprint"),
                                         "role": "verifier" if row.get("call_id") in verifiers else "engineer_internal",
                                         "measurement": asdict(value), "summary_path": row["summary_path"]})
        installed = []
        for rows in self.changes.values():
            for row in rows:
                if row.get("workspace") != inputs.get("CANONICAL") or row.get("interval_start", "") <= invocation.get("time", ""):
                    continue
                original = read_json(self.run / row["post_observation"])
                identity = row.get("after_fingerprint")
                if identity and any(state.get("workspace") == inputs["CANONICAL"] and state.get("fingerprint") == identity
                                    for state in original.get("source_states", [])):
                    installed.append({"source": identity, "time": row["interval_start"],
                                      "evidence": asdict(self.ref(row["post_observation"]))})
        installed.sort(key=lambda item: item["time"])
        threshold = threshold_from_inputs(inputs)
        comparisons = []
        for item in documents:
            document = item["document"]
            if not isinstance(document, dict) or not isinstance(document.get("comparisons"), dict):
                continue
            for name, ratio in document["comparisons"].items():
                if isinstance(ratio, (float, int)) and ratio > 0:
                    comparisons.append({"name": name, "gain": ratio - 1, "certainty": "review",
                                        "threshold_met": ratio - 1 >= threshold["value"] if threshold["value"] else None,
                                        "evidence": item["evidence"]})
        return {"original_inputs": inputs, "source_changes": changes, "diffs": diffs,
                "knowledge": self.knowledge(attempt.id, inputs), "internal_measurements": measurements,
                "installed_observations": installed, "recovered_sources": self.recovered_sources(documents, inputs),
                "reviewed_comparisons": comparisons, "threshold": threshold,
                "local_evidence": [item["evidence"] for item in documents]}

    def independent_pair(self, subject, inputs):
        attempt = subject.attempt
        if not attempt.parent_source or not attempt.state:
            return None
        parent = self.baseline(attempt.parent_source, inputs.get("CANONICAL"), self.calls[attempt.id].get("time", ""))
        if parent is None:
            return None
        candidate = self.candidate_measurement(attempt.state.fingerprint, parent.contract)
        return (parent, candidate) if candidate.status == "qualified" else None


@dataclass(frozen=True)
class ArchiveHistory:
    attempts: tuple[Attempt, ...]
    worlds: tuple[DiscoveryWorld, ...]
    evidence: tuple[EvidenceRef, ...]
    excluded: tuple[dict, ...]
    trace: tuple = ()


def read_archive(run, *, campaign, tasks, excluded_tasks=(), states=None, knowledge=None):
    attempts, worlds, excluded = [], [], []
    reader = AuditReader(run)
    groups = defaultdict(list)
    dispatches, blocked_rounds = {}, set()
    for kernel, call, invocation in reader.inventory(tasks, excluded_tasks):
        if kernel is None:
            excluded.append({"id": call["call_id"], "campaign": campaign, "task": call["task"],
                             "reason": "explicitly outside the configured kernel split",
                             "inventory": asdict(reader.ref(invocation) if invocation.is_file()
                                                 else reader.ref("index/agent_calls.jsonl"))})
            continue
        try:
            attempt, dispatch = reader.attempt(kernel, campaign, invocation)
            if attempt.id != call["call_id"]:
                raise KnowledgeError("invocation identity disagrees with original inventory")
        except OSError as error:
            attempt = Attempt(call["call_id"], kernel, campaign, reader.ref("index/agent_calls.jsonl"),
                              None, None, None, Measurement("unknown"), "trace_only",
                              (f"original invocation unavailable: {error}",))
            dispatch = None
        attempts.append(attempt)
        output_dir = call.get("context", {}).get("OUTPUT_DIR")
        if output_dir:
            dispatches[attempt.id] = output_dir
            if dispatch is None:
                blocked_rounds.add(str(Path(output_dir).parent))
        if dispatch is not None:
            key = (kernel, dispatch.round, dispatch.parent.fingerprint, dispatch.input.revision,
                   dispatch.generation, dispatch.baseline)
            groups[key].append((attempt, dispatch))
    duplicates = {path for path, count in Counter(dispatches.values()).items() if count > 1}
    blocked_rounds.update(str(Path(path).parent) for path in duplicates)
    attempts = [replace(attempt, disposition="trace_only",
                        gaps=("round contains unsupported or reused dispatch input",))
                if attempt.campaign == campaign and attempt.disposition == "replay_ready"
                and str(Path(dispatches[attempt.id]).parent) in blocked_rounds else attempt
                for attempt in attempts]
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        kernel, round_path, *_ = key
        if round_path in blocked_rounds:
            continue
        rows.sort(key=lambda row: row[1].order)
        if len({row[1].order for row in rows}) != len(rows):
            raise KnowledgeError("ambiguous historical dispatch order")
        dispatch = rows[0][1]
        world_id = f"{campaign}/{kernel}/{digest([row[0].id for row in rows])[:16]}"
        root = DiscoveryNode(world_id + "/root", None, None, dispatch.baseline,
                             dispatch.input.evidence, state=dispatch.parent, context=dispatch.input)
        state = KernelState.from_dict({**(states or {}).get(kernel, {}),
                                       **{key: value for key, value in asdict(dispatch.state).items()
                                          if value is not None}})
        worlds.append(reconstruct(world_id, kernel, campaign, state, dispatch.generation,
                                  root, [attempt for attempt, _ in rows]))
    trace = lane_trees(reader, attempts, campaign=campaign, tasks=tasks, states=states, knowledge=knowledge)
    return ArchiveHistory(tuple(attempts), tuple(worlds), tuple(reader._references.values()), tuple(excluded),
                          tuple(trace))
