"""Explicit offline investigation, bounded discovery cycles, and final release."""

import argparse
import json
import sys
from pathlib import Path

from kernelwiki import KnowledgeClient
from kernelwiki.knowledge import KnowledgeError

from .evidence import read_json, reference, write_json
from .harness import Policy
from .loop.inner import ROLES, evolve
from .replay_buffer.build import ingest
from .replay_buffer.evaluate import evaluate
from .replay_buffer.store import open_buffer
from .report import diagnostics, render_report
from .reverse_engineering.analyst import ROLE as ANALYST_ROLE
from .reverse_engineering.investigate import investigate
from .reverse_engineering.model import subject_for
from .reverse_engineering.sources.archive import AuditReader
from .reverse_engineering.sources.capture import load_online
from .reverse_engineering.trace import trace
from .runtime import AgentUnavailable, RuntimeAgent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Knowledge-as-a-Policy / Dream-RSI")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("ingest", help="read raw GLM7 history into immutable replay artifacts")
    importer.add_argument("--config", default=str(Path(__file__).parent / "replay_buffer/configs/glm7.yaml"))
    importer.add_argument("--output", required=True)
    freezer = commands.add_parser("freeze", help="combine completed online history with a verified replay pool")
    freezer.add_argument("--pool", required=True)
    freezer.add_argument("--online", action="append", default=[])
    freezer.add_argument("--investigation", action="append", default=[])
    freezer.add_argument("--output", required=True)
    tracer = commands.add_parser("trace", help="reverse trace one external attempt")
    trace_source = tracer.add_mutually_exclusive_group(required=True)
    trace_source.add_argument("--pool")
    trace_source.add_argument("--run", help="original GEAK audit archive, without any replay pool")
    trace_source.add_argument("--online", help="completed fixed-harness capture, without any replay pool")
    tracer.add_argument("--kernel", help="required with --run")
    tracer.add_argument("--campaign")
    tracer.add_argument("--attempt", required=True)
    tracer.add_argument("--output")
    investigator = commands.add_parser("investigate", help="bounded local evidence search and optional isolated method review")
    investigation_source = investigator.add_mutually_exclusive_group(required=True)
    investigation_source.add_argument("--pool")
    investigation_source.add_argument("--run", help="original GEAK audit archive, without any replay pool")
    investigation_source.add_argument("--online", help="completed fixed-harness capture, without any replay pool")
    investigator.add_argument("--kernel", help="required with --run")
    investigator.add_argument("--campaign")
    investigator.add_argument("--read-limit", type=int, default=64)
    investigator.add_argument("--attempt", required=True)
    investigator.add_argument("--output", required=True)
    investigator.add_argument("--max-steps", type=int, default=4)
    investigator.add_argument("--model-calls", type=int, default=0)
    investigator.add_argument("--node")
    investigator.add_argument("--claude")
    investigator.add_argument("--agent-timeout", type=int, default=300)
    replayer = commands.add_parser("replay", help="evaluate a pinned policy without Agent or GPU execution")
    replayer.add_argument("--pool", required=True)
    replayer.add_argument("--wiki", required=True)
    replayer.add_argument("--revision")
    replayer.add_argument("--split", choices=["train", "validation", "test"], default="train")
    replayer.add_argument("--neutral", action="store_true")
    replayer.add_argument("--policy", help="experimental working-policy JSON, without changing active knowledge")
    replayer.add_argument("--cohort", choices=["primary", "diagnostic"], default="primary")
    replayer.add_argument("--output", required=True)
    evolver = commands.add_parser("evolve", help="bounded inner train/validation evolution; never publishes or evaluates test")
    evolver.add_argument("--pool", required=True)
    evolver.add_argument("--wiki", required=True)
    evolver.add_argument("--output", required=True)
    evolver.add_argument("--publish", action="store_true", help="rejected legacy option; use release after a qualified cycle")
    evolver.add_argument("--policy")
    evolver.add_argument("--max-candidates", type=int, default=4)
    evolver.add_argument("--node")
    evolver.add_argument("--claude")
    evolver.add_argument("--agent-timeout", type=int, default=300)
    reporter = commands.add_parser("report", help="render derived coverage and qualification report")
    reporter.add_argument("--pool", required=True)
    reporter.add_argument("--experiment")
    reporter.add_argument("--output", required=True)
    explorer = commands.add_parser("explore", help="explicit online exploration with the fixed Agent/evaluator")
    explorer.add_argument("--config", required=True)
    explorer.add_argument("--wiki", required=True)
    explorer.add_argument("--output", required=True)
    explorer.add_argument("--revision")
    explorer.add_argument("--policy")
    calibration = commands.add_parser("calibrate", help="one normal direction, one GPU, at most 30 minutes; no full-cycle authorization")
    calibration.add_argument("--config", required=True)
    calibration.add_argument("--wiki", required=True)
    calibration.add_argument("--output", required=True)
    cycler = commands.add_parser("cycle", help="online -> reverse -> new pool -> evolve; requests approval before real full runs")
    cycler.add_argument("--config", required=True)
    cycler.add_argument("--wiki", required=True)
    cycler.add_argument("--output", required=True)
    cycler.add_argument("--resume", action="store_true")
    publisher = commands.add_parser("release", help="reproduce real final gates, then publish with KernelWiki compare-and-swap")
    publisher.add_argument("--cycle", required=True)
    publisher.add_argument("--wiki", required=True)
    args = parser.parse_args(argv)
    progress = lambda value: print(json.dumps(value, ensure_ascii=False), file=sys.stderr, flush=True)
    runtime = None

    def agent(role, payload, directory):
        nonlocal runtime
        if runtime is None:
            runtime = RuntimeAgent(node=args.node, claude=args.claude, timeout=args.agent_timeout,
                                   roles={**ROLES, "reverse_analyst": ANALYST_ROLE})
        return runtime(role, payload, directory)

    def trace_subject():
        if args.pool:
            return open_buffer(args.pool).subject(args.attempt)
        if args.run:
            if not args.kernel:
                raise KnowledgeError("--run requires the original kernel assignment via --kernel")
            return AuditReader(args.run).subject(args.attempt, args.kernel, args.campaign)
        _, world, attempts = load_online(args.online)
        attempt = next((item for item in attempts if item.id == args.attempt), None)
        if attempt is None:
            raise KnowledgeError(f"unknown online attempt: {args.attempt}")
        return subject_for(attempt, (world,), (reference(Path(args.online) / "result.json"),))

    def working_policy():
        if not args.policy:
            return None
        snapshot = KnowledgeClient(args.wiki).snapshot(getattr(args, "revision", None))
        return Policy.from_dict(read_json(args.policy), snapshot.records)

    try:
        if args.command == "ingest":
            result = ingest(args.config, args.output, progress=progress)
        elif args.command == "freeze":
            from .replay_buffer.build import freeze
            if not args.online and not args.investigation:
                raise KnowledgeError("freeze needs new online history or a completed investigation")
            result = freeze(args.pool, args.online, args.output, investigations=args.investigation)
        elif args.command == "trace":
            result = trace(trace_subject()).to_dict()
            if args.output:
                write_json(args.output, result)
                result = {"trace": args.output, "attempt": args.attempt, "effect": result["trace"]["effect"]}
        elif args.command == "investigate":
            result = investigate(trace_subject(), args.output, max_steps=args.max_steps, read_limit=args.read_limit,
                                 model_calls=args.model_calls, agent=agent if args.model_calls else None)
            result = {"status": result["status"], "model_calls": result["model_calls"],
                      "gaps": result["trace"]["gaps"], "output": args.output}
        elif args.command == "replay":
            buffer = open_buffer(args.pool)
            snapshot = KnowledgeClient(args.wiki).snapshot(args.revision)
            selected = buffer.select(args.split, cohort=args.cohort)
            if not selected:
                raise KnowledgeError(f"no {args.cohort} worlds in {args.split}; diagnostic traces are not primary qualification")
            policy = working_policy() or Policy(snapshot.policy, snapshot.records)
            result = evaluate(selected, Policy(policy.knowledge, policy.records, neutral=args.neutral),
                              buffer.budgets, buffer.objective)
            result["diagnostics"] = diagnostics(result["traces"])
            result["knowledge_revision"] = snapshot.revision
            result["cohort"] = args.cohort
            write_json(args.output, result)
            result = {"summary": result["summary"], "diagnostics": result["diagnostics"]}
        elif args.command == "evolve":
            result = evolve(args.pool, args.wiki, args.output,
                            agent=agent, policy=working_policy(), max_candidates=args.max_candidates,
                            publish=args.publish, progress=progress)
            result = {key: value for key, value in result.items() if key not in {"opportunities", "artifacts"}}
        elif args.command == "report":
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(render_report(args.pool, args.experiment))
            result = {"report": str(path)}
        elif args.command == "explore":
            from .loop.online import explore
            result = explore(args.config, args.wiki, args.output, revision=args.revision, policy=working_policy())
        elif args.command == "calibrate":
            from .loop.online import calibrate
            result = calibrate(args.config, args.wiki, args.output)
        elif args.command == "cycle":
            from .loop.outer import cycle
            result = cycle(args.config, args.wiki, args.output, resume=args.resume, progress=progress)
            result = {key: value for key, value in result.items() if key != "artifacts"}
        else:
            from .loop.qualification import release
            result = release(args.cycle, args.wiki)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (KnowledgeError, AgentUnavailable, OSError, ValueError) as error:
        print(f"RSI gate failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
