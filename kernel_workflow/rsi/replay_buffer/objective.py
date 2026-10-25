"""Deterministic replay objective with explicit units and macro aggregation."""

from dataclasses import dataclass
from statistics import mean

from kernelwiki.knowledge import KnowledgeError, finite


@dataclass(frozen=True)
class ReplayObjective:
    beta_cost: float = 0.01
    beta_parallel: float = 0.0

    def __post_init__(self):
        if finite(self.beta_cost) < 0 or finite(self.beta_parallel) < 0:
            raise KnowledgeError("objective penalties must be nonnegative")

    def score(self, trace):
        quality = trace["best_quality"]
        cost = self.beta_cost * trace["charged_budget"]
        parallel = self.beta_parallel * trace["attempted_nodes"] / max(1, trace["decision_batches"])
        return {"objective": quality - cost + parallel, "quality": quality,
                "cost_penalty": cost, "parallel_bonus": parallel}


def aggregate(traces, objective=None):
    objective = objective or ReplayObjective()
    grouped = {}
    for trace in traces:
        grouped.setdefault(trace["kernel"], {}).setdefault(trace["world_id"], []).append(
            (trace, objective.score(trace)))
    if not grouped:
        raise KnowledgeError("cannot qualify an empty replay split")
    kernels = {}
    for kernel, worlds in grouped.items():
        world_scores, full_quality = [], []
        for rows in worlds.values():
            world_scores.append(mean(score["objective"] for _, score in rows))
            full_quality.append(max(rows, key=lambda row: row[0]["budget"])[1]["quality"])
        kernels[kernel] = {"objective": mean(world_scores), "full_budget_quality": mean(full_quality),
                           "worlds": len(worlds)}
    return {"objective": mean(row["objective"] for row in kernels.values()),
            "full_budget_quality": mean(row["full_budget_quality"] for row in kernels.values()),
            "kernels": kernels, "worlds": sum(row["worlds"] for row in kernels.values())}
