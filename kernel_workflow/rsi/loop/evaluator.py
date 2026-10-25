"""Independent fixed-contract host measurements for online discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from threading import Lock

from kernelwiki._atomic import atomic_bytes
from kernelwiki.knowledge import KnowledgeError, digest, finite, relative_path

from ..discovery import Measurement
from ..evidence import persist, reference, snapshot
from ..runtime import run_process


class FixedEvaluator:
    def __init__(self, config, root_workspace, source_paths, *, deadline=None, device=None):
        self.config = config
        self.source_paths = tuple(source_paths)
        self.weights = config["case_weights"]
        if not self.weights or any(finite(value) <= 0 for value in self.weights.values()):
            raise KnowledgeError("fixed evaluator needs positive weights for every case")
        if (not isinstance(config["command"], list) or not config["command"]
                or any(not isinstance(value, str) for value in config["command"])):
            raise KnowledgeError("evaluator command must be an argument vector")
        relative_path(config["result_file"])
        self.contract_files = tuple(config["contract_files"])
        if not self.contract_files:
            raise KnowledgeError("fixed evaluator must declare its immutable contract files")
        result = config["result_file"]
        if result in self.contract_files or any(result == name or result.startswith(name + "/")
                                               for name in self.source_paths):
            raise KnowledgeError("evaluator output cannot overwrite source or contract files")
        self.expected = self.contract(root_workspace)
        self.protocol = digest({"config": config, "files": self.expected,
                                "device": device,
                                "metric": "fixed-reference-weighted-latency-v2"})
        self.revision = self.protocol
        self.runtime = None
        self.normalization = None
        self.deadline = deadline
        self.device = device
        self._lock = Lock()

    def contract(self, workspace):
        values = {}
        for name in self.contract_files:
            relative_path(name)
            values[name] = hashlib.sha256((Path(workspace) / name).read_bytes()).hexdigest()
        return values

    def measure(self, workspace, directory, gpu_id):
        with self._lock:
            return self._measure(workspace, directory, gpu_id)

    def _measure(self, workspace, directory, gpu_id):
        workspace, directory = Path(workspace).resolve(), Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        before = snapshot(workspace, self.source_paths, directory)
        input_ref = persist(directory / "input.json", {"source": before, "contract": self.expected,
                                                       "evaluator_revision": self.revision})
        if self.contract(workspace) != self.expected:
            measurement = Measurement("unknown", before["fingerprint"], evidence=(input_ref,), reason="contract_changed")
            persist(directory / "measurement.json", asdict(measurement))
            return measurement
        latencies, numerators, runtimes, refs = [], [], [], [input_ref]
        status = "qualified"
        reason = ""
        for index in (1, 2):
            result_file = workspace / self.config["result_file"]
            if result_file.is_symlink():
                raise KnowledgeError("evaluator result cannot be a symlink")
            if result_file.exists():
                result_file.unlink()
            command = [argument.replace("{gpu_id}", str(gpu_id)).replace("{workspace}", str(workspace))
                       for argument in self.config["command"]]
            process = run_process(command, workspace, directory / f"process-{index}",
                                  timeout=self.config.get("timeout_seconds", 600),
                                  environment={"GEAK_GPU_ALLOWED": str(gpu_id)}, deadline=self.deadline)
            refs.append(process["evidence"])
            refs.append(reference(directory / f"process-{index}/process.json"))
            after = snapshot(workspace, self.source_paths, directory)
            if after["fingerprint"] != before["fingerprint"] or self.contract(workspace) != self.expected:
                status = "unknown"
                reason = "source_or_contract_changed"
                break
            if process["exit_code"] != 0 or not result_file.is_file():
                status = "unknown"
                reason = "evaluation_process_failed_or_report_missing"
                break
            payload = result_file.read_bytes()
            atomic_bytes(directory / f"report-{index}.json", payload, durable=True)
            refs.append(reference(directory / f"report-{index}.json"))
            try:
                report = json.loads(payload)
                runtime = report.get("runtime_fingerprint")
                if not isinstance(runtime, str) or not runtime:
                    raise KnowledgeError("evaluator must identify its runtime")
                runtimes.append(runtime)
                if len(set(runtimes)) != 1:
                    raise KnowledgeError("independent evaluator processes use different runtimes")
                if self.runtime is not None and runtime != self.runtime:
                    raise KnowledgeError("candidate runtime differs from the root runtime")
                if self.device is not None and report.get("device_identity") != self.device:
                    raise KnowledgeError("evaluator result is not bound to the assigned physical GPU")
                if report.get("correctness_pass") is False:
                    status, reason = "failed", "correctness_failed"
                    break
                if not all(report.get(key) is True for key in
                           ("correctness_pass", "measurement_valid", "performance_scope_complete")):
                    raise KnowledgeError("unqualified evaluator result")
                cases = report["per_case"]
                if len(cases) != len(self.weights) or {case["name"] for case in cases} != set(self.weights):
                    raise KnowledgeError("evaluator case coverage mismatch")
                numerator, denominator = 0.0, 0.0
                for case in cases:
                    if case.get("measurement_valid") is not True:
                        raise KnowledgeError("invalid case timing")
                    baseline, candidate = finite(case["baseline_ms"]), finite(case["optimized_ms"])
                    if baseline <= 0 or candidate <= 0:
                        raise KnowledgeError("nonpositive case timing")
                    numerator += self.weights[case["name"]] * baseline
                    denominator += self.weights[case["name"]] * candidate
                numerators.append(numerator)
                latencies.append(denominator)
            except (ValueError, KeyError, TypeError) as error:
                status = "unknown"
                reason = str(error)
                break
        if status == "qualified" and len(latencies) == 2:
            if self.runtime is None:
                self.runtime, self.normalization = runtimes[0], median(numerators)
                self.revision = digest({"protocol": self.protocol, "runtime": self.runtime,
                                        "normalization": self.normalization})
            measurement = Measurement("qualified", before["fingerprint"], self.normalization / median(latencies),
                                      self.revision, tuple(refs))
        else:
            measurement = Measurement(status, before["fingerprint"], evidence=tuple(refs), reason=reason)
        persist(directory / "measurement.json", asdict(measurement))
        return measurement
