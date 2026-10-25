"""Policy-independent qualification support and kernel-level partitions."""

from kernelwiki.knowledge import KnowledgeError

DEFAULT_SPLITS = {"train": ("dsa-decode", "dsa-prefill", "dsa-sparse-mla"),
                  "validation": ("mxfp4-moe",), "test": ("mla-bmm", "qkv-a-proj", "o-proj")}
DEFAULT_BUDGETS = (1, 2, 4, 6)


def qualification_gaps(world, budget):
    nodes = [node for branch in world.branches for node in branch]
    gaps = []
    if len(world.branches) < 2 or not any(len(branch) > 1 for branch in world.branches):
        gaps.append("no supported choice between a new branch and continuation")
    if len(nodes) * world.generation.cost < budget:
        gaps.append("recorded cost does not cover the largest budget")
    if any(node.measurement.status == "unknown" for node in nodes):
        gaps.append("unknown evaluation outcomes")
    return gaps


def cohorts(worlds, budgets=DEFAULT_BUDGETS):
    if not budgets or any(type(budget) is not int or budget < 1 for budget in budgets):
        raise KnowledgeError("cohorts require frozen positive direction budgets")
    excluded = {world.id: gaps for world in worlds if (gaps := qualification_gaps(world, max(budgets)))}
    return {"all_traces": "attempts.jsonl", "diagnostic": [world.id for world in worlds],
            "primary": [world.id for world in worlds if world.id not in excluded], "excluded": excluded,
            "rule": "branching + continuation + complete outcomes + largest recorded cost before policy evaluation"}


def split_worlds(worlds, splits=None):
    splits = splits or DEFAULT_SPLITS
    if set(splits) != {"train", "validation", "test"}:
        raise KnowledgeError("train, validation, and test are required")
    kernels = [kernel for names in splits.values() for kernel in names]
    if len(set(kernels)) != len(kernels):
        raise KnowledgeError("a kernel must stay in exactly one split across all campaigns")
    if len({world.id for world in worlds}) != len(worlds):
        raise KnowledgeError("duplicate world id")
    if any(world.kernel not in kernels for world in worlds):
        raise KnowledgeError("world has no assigned kernel split")
    result = {name: tuple(world for world in worlds if world.kernel in names)
              for name, names in splits.items()}
    for name, names in splits.items():
        missing = set(names) - {world.kernel for world in result[name]}
        if missing:
            raise KnowledgeError(f"{name} lacks verified worlds for {sorted(missing)}")
    return result
