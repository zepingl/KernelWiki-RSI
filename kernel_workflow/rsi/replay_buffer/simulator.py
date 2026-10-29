"""Deterministic recorded transitions; scheduling is owned by the shared Harness."""

from itertools import pairwise

from ..discovery import GenerationSpec
from ..harness import Action, Observation, ParentView, Rollout
from ..trace.tree import TreeView, replay_node
from .cohorts import kernel_of

__all__ = ["Action", "Observation", "ParentView", "ReplayWorld", "TreeReplay", "replay"]


class ReplayWorld:
    def __init__(self, world, budget):
        self._world, self._budget = world, budget
        self.reset()

    def reset(self):
        self.rollout = Rollout(self._world.state, self._world.generation, self._world.root, self._budget)
        self._branches = iter(self._world.branches)
        self._children = {parent.id: child for branch in self._world.branches
                          for parent, child in pairwise(branch)}
        self._recorded_count = sum(map(len, self._world.branches))
        return self.observe()

    def observe(self):
        return self.rollout.observe()

    @property
    def done(self):
        return self.rollout.stop is not None

    def step(self, action):
        self.rollout.validate(action)
        outcomes = []
        for parent in action.parents:
            if parent == "root":
                branch = next(self._branches, None)
                outcomes.append(branch[0] if branch else None)
            else:
                outcomes.append(self._children.get(self.rollout.nodes[parent].id))
        self.rollout.step(action, outcomes)
        if not self.done and self.rollout.attempted == self._recorded_count:
            self.rollout.stop = "support_exhausted"
        return self.observe()

    def trace(self):
        return self.rollout.trace(self._world.id, self._world.kernel, self._world.campaign)


def tree_generation(tree):
    """Rollout still schedules one or two normal workers; width and mode become actions after the SDK change."""
    generation = tree.expansions[0].plan.generation
    return GenerationSpec("normal", generation.model, generation.effort,
                          tree.id, tree.tree.contract, worker_limit=2)


class TreeReplay(ReplayWorld):
    """Selecting a node reveals its next recorded child: expansions in record order, slots in dispatch order."""

    def reset(self):
        tree = self._world
        self.rollout = Rollout(tree.tree.task, tree_generation(tree), replay_node(tree, tree.root.id), self._budget)
        self._pending = {node_id: [child.id for child in tree.children(node_id)] for node_id in tree.nodes}
        self._recorded_count = len(tree.nodes) - 1
        return self.observe()

    def step(self, action):
        self.rollout.validate(action)
        outcomes = []
        for parent in action.parents:
            pending = self._pending[self.rollout.nodes[parent].id]
            outcomes.append(replay_node(self._world, pending.pop(0)) if pending else None)
        self.rollout.step(action, outcomes)
        if not self.done and self.rollout.attempted == self._recorded_count:
            self.rollout.stop = "support_exhausted"
        return self.observe()

    def trace(self):
        return self.rollout.trace(self._world.id, kernel_of(self._world), self._world.tree.facets["geak"]["campaign"])


def replay(world, policy, budget):
    simulator = (TreeReplay if isinstance(world, TreeView) else ReplayWorld)(world, budget)
    while not simulator.done:
        simulator.step(policy.select(simulator.observe()))
    return simulator.trace()
