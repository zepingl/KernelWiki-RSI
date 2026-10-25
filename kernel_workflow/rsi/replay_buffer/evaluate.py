"""Paired fixed-buffer policy evaluation and qualification."""

from kernelwiki.knowledge import KnowledgeError

from .cohorts import DEFAULT_BUDGETS, qualification_gaps
from .objective import ReplayObjective, aggregate
from .simulator import replay


def evaluate(worlds, policy, budgets=DEFAULT_BUDGETS, objective=None):
    if (not budgets or len(set(budgets)) != len(budgets)
            or any(type(budget) is not int or budget < 1 for budget in budgets)):
        raise KnowledgeError("budgets must be unique positive integers")
    objective = objective or ReplayObjective()
    worlds = tuple(worlds)
    traces = [replay(world, policy, budget) for world in worlds for budget in sorted(budgets)]
    gaps = {world.id: found for world in worlds if (found := qualification_gaps(world, max(budgets)))}
    unsupported = [{"world": trace["world_id"], "budget": trace["budget"], "stop_reason": trace["stop_reason"],
                    "unsupported_requests": trace["unsupported_requests"]}
                   for trace in traces if not trace["support_complete"]]
    return {"summary": aggregate(traces, objective), "traces": traces,
            "qualification": {"eligible": not gaps and not unsupported, "world_gaps": gaps,
                              "unsupported_pairs": unsupported}}


def qualifies(candidate, incumbent, *, held_out=False):
    candidate_pairs = {(row["world_id"], row["kernel"], row["budget"]) for row in candidate["traces"]}
    incumbent_pairs = {(row["world_id"], row["kernel"], row["budget"]) for row in incumbent["traces"]}
    if candidate_pairs != incumbent_pairs or len(candidate["traces"]) != len(incumbent["traces"]):
        raise KnowledgeError("qualification requires identical world-budget pairs without dropping unsupported requests")
    if not candidate.get("qualification", {}).get("eligible") or not incumbent.get("qualification", {}).get("eligible"):
        return False
    candidate, incumbent = candidate["summary"], incumbent["summary"]
    if set(candidate["kernels"]) != set(incumbent["kernels"]):
        raise KnowledgeError("qualification requires paired kernels")
    improvement = candidate["objective"] - incumbent["objective"]
    if improvement < 0 or (not held_out and improvement <= 1e-12):
        return False
    if candidate["full_budget_quality"] + 1e-12 < incumbent["full_budget_quality"]:
        return False
    return not held_out or all(candidate["kernels"][kernel]["full_budget_quality"] + 1e-12
                              >= row["full_budget_quality"]
                              for kernel, row in incumbent["kernels"].items())
