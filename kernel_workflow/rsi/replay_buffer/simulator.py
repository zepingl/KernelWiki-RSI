"""Deterministic recorded transitions; scheduling is owned by the shared Harness."""

from itertools import pairwise

from ..harness import Action, Observation, ParentView, Rollout

__all__ = ["Action", "Observation", "ParentView", "ReplayWorld", "replay"]


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


def replay(world, policy, budget):
    simulator = ReplayWorld(world, budget)
    while not simulator.done:
        simulator.step(policy.select(simulator.observe()))
    return simulator.trace()
