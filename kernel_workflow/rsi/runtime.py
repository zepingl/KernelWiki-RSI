"""Fixed experimental Agent execution, bounded processes and implementation sources."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import kernelwiki
from kernelwiki.knowledge import KnowledgeError, canonical

from .evidence import persist, read_json, write_json


def remaining_seconds(deadline, limit):
    remaining = limit if deadline is None else min(limit, deadline - time.monotonic())
    if remaining <= 0:
        raise KnowledgeError("online wall-clock budget exhausted before launch")
    return remaining


def run_process(command, cwd, directory, *, timeout, environment=None, deadline=None):
    timeout = remaining_seconds(deadline, timeout)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    invocation = persist(directory / "invocation.json", {"command": list(command), "cwd": str(cwd),
                                                          "timeout_seconds": timeout})
    started = time.monotonic()
    status = {"exit_code": None, "timed_out": False}
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(list(command), cwd=cwd, env={**os.environ, **(environment or {})},
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                status["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                status.update(exit_code=process.wait(), timed_out=True)
        except OSError as error:
            status.update(exit_code=-1, error=str(error))
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    status["elapsed_seconds"] = time.monotonic() - started
    persist(directory / "process.json", status)
    return {**status, "evidence": invocation}


MODEL = "claude-opus-5.5"
EFFORT = "high"
SKILL_ROOT = Path("/home/zepingl/.cursor/skills/five-step-engineering")
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parent


class AgentUnavailable(RuntimeError):
    """The requested experimental model failed; no fallback is permitted."""


class FrozenExperimentChanged(KnowledgeError):
    """A frozen input changed; all downstream gates must stop."""


@dataclass(frozen=True)
class AgentRole:
    schema: dict
    instructions: Path | str


class RuntimeAgent:
    def __init__(self, *, node=None, claude=None, timeout=300, skill_root=SKILL_ROOT, deadline=None, roles=None):
        self.node = node or os.environ.get("GEAK_NODE_BIN") or shutil.which("node")
        self.claude = claude or os.environ.get("GEAK_CLAUDE_BIN") or shutil.which("claude")
        if not self.node or not self.claude:
            raise AgentUnavailable("set GEAK_NODE_BIN and GEAK_CLAUDE_BIN to the existing standalone runtime CLIs")
        self.timeout = timeout
        self.skill_root = Path(skill_root)
        self.deadline = deadline
        self.roles = dict(roles or {})

    def __call__(self, role, payload, directory):
        remaining_seconds(self.deadline, self.timeout)
        if role not in self.roles:
            raise KnowledgeError(f"unconfigured experimental Agent role: {role}")
        spec = self.roles[role]
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        schema = spec.schema
        prompt = spec.instructions.read_text() if isinstance(spec.instructions, Path) else spec.instructions
        if isinstance(spec.instructions, Path):
            for relative in ("SKILL.md", "references/structural-design.md", "references/closed-loop-engineering.md"):
                prompt += f"\n\n## Local engineering source: {relative}\n" + (self.skill_root / relative).read_text()
        prompt += "\n\n## Frozen experiment input\n" + canonical(payload).decode()
        registry = {"default_profile": "rsi", "profiles": {"rsi": {"agent": "claude"}},
                    "models": {MODEL: {"id": MODEL}}, "agents": {"claude": {"bin": self.claude, "prompt": "stdin",
                    "dialect": "anthropic", "model_flag": "--model",
                    "args": ["-p", "--bare", "--tools", "", "--strict-mcp-config", "--mcp-config",
                             '{"mcpServers":{}}', "--setting-sources", "", "--disable-slash-commands",
                             "--no-session-persistence", "--effort", EFFORT]}}}
        write_json(directory / "registry.json", registry)
        write_json(directory / "args.json", {"prompt": prompt, "schema": schema, "role": role,
                                             "model": MODEL, "cwd": str(directory.resolve())})
        command = [self.node, str(REPO_ROOT / "interface/runtime/engine/run_workflow.mjs"),
                   str(WORKFLOW_ROOT / "rsi/agent_workflow.js"), "--registry", str(directory / "registry.json"),
                   "--agent", "claude", "--model", MODEL, "--concurrency", "1", "--agent-timeout-ms",
                   str(self.timeout * 1000), "--args-file", str(directory / "args.json"),
                   "--result-file", str(directory / "result.json"), "--metrics-file", str(directory / "metrics.json")]
        write_json(directory / "invocation.json", {"command": command, "model": MODEL, "effort": EFFORT,
                                                   "schema_retries": 0, "tools": [], "session_reuse": False})
        process = run_process(command, directory, directory / "process", timeout=self.timeout + 20,
                              environment={"GEAK_SCHEMA_RETRIES": "0"}, deadline=self.deadline)
        if process["timed_out"]:
            raise AgentUnavailable("experimental Agent timed out; process group terminated")
        if process["exit_code"]:
            metrics = read_json(directory / "metrics.json") if (directory / "metrics.json").exists() else {}
            message = metrics.get("error") or (directory / "process/stderr.log").read_text()[-1200:]
            if metrics.get("schema_failures"):
                raise KnowledgeError(f"invalid agent output: {message}")
            raise AgentUnavailable(f"{MODEL}/{EFFORT} failed: {message}")
        return read_json(directory / "result.json")


def frozen_sources():
    wiki_root = Path(kernelwiki.__file__).resolve().parent
    directories = (("GEAK", WORKFLOW_ROOT / "rsi", REPO_ROOT, "*.py"),
                   ("GEAK", WORKFLOW_ROOT / "rsi", REPO_ROOT, "*.js"),
                   ("GEAK", WORKFLOW_ROOT / "rsi", REPO_ROOT, "*.md"),
                   ("GEAK", REPO_ROOT / "interface/runtime/engine", REPO_ROOT, "*.mjs"),
                   ("KernelWiki", wiki_root, wiki_root.parent, "*.py"),
                   ("skills", SKILL_ROOT, SKILL_ROOT, "*.md"))
    result = {f"{label}/{path.relative_to(root)}": path
              for label, directory, root, pattern in directories
              for path in sorted(directory.rglob(pattern))}
    result["GEAK/kernel_workflow/kernel_lane.js"] = WORKFLOW_ROOT / "kernel_lane.js"
    return result
