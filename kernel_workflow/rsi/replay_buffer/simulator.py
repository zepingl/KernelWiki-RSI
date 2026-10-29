"""Deterministic recorded transitions; scheduling is owned by the shared Harness."""

from itertools import pairwise

from ..discovery import GenerationSpec
from ..harness import Action, ExpansionObservation, Observation, ParentView, Rollout, SlotView, slot_cost
from ..trace.tree import TreeView, replay_node
from .cohorts import kernel_of

__all__ = ["Action", "Observation", "ParentView", "ReplayWorld", "TreeReplay", "episode_replay", "replay"]


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


PLANNED_ROLES = {"pointed", "cited"}
EPISODE_RULES = {"step", "truncate"}


def episode_replay(tree, policy, budget, rule="step"):
    """Policy schema 2 over a tree's recorded expansions, in record order.

    Each step is scored in the state history actually reached: the gain of the best selected direction over
    its parent. `step` continues along the recorded trunk whatever was selected; `truncate` stops the episode
    when the selection drops the direction history continued from. Selecting an undispatched slot is
    unsupported; running out of recorded steps is the lane's own end.
    """
    if rule not in EPISODE_RULES:
        raise ValueError(f"unknown episode rule: {rule}")
    remaining, factor, stagnation = budget, 1.0, 0
    events, attempted, batches, requested, unsupported = [], 0, 0, 0, 0
    stop_reason = "recorded_end"
    steps = tree.expansions
    for position, expansion in enumerate(steps, 1):
        children = {node.slot: node for node in tree.nodes.values() if node.expansion == expansion.id}
        slots = tuple(SlotView(index, slot.specialty, slot.mode,
                               frozenset(use.target for use in children[index].uses if use.role in PLANNED_ROLES)
                               if index in children else frozenset())
                      for index, slot in enumerate(expansion.plan.slots))
        action = policy.select_expansion(ExpansionObservation(tree.tree.task, position, stagnation, remaining,
                                                              factor - 1, slots))
        if action.stop:
            stop_reason = "policy_stop"
            events.append({"round": position, "action": [], "rule_ids": list(action.rule_ids), "reason": action.reason})
            break
        batches += 1
        parent = tree.outcome(expansion.parents[0])
        revealed, gains = [], []
        for index in action.slots:
            requested += 1
            remaining -= slot_cost(slots[index])
            node = children.get(index)
            if node is None:
                unsupported += 1
                continue
            attempted += 1
            outcome = tree.outcome(node.id)
            gain = (outcome.primary_score / parent.primary_score - 1
                    if outcome.status == "qualified" and parent.status == "qualified" else None)
            if gain is not None:
                gains.append(gain)
            revealed.append({"node_id": node.id, "slot": index, "status": outcome.status, "gain": gain})
        step_gain = max([0.0, *gains])
        factor *= 1 + step_gain
        stagnation = 0 if step_gain > 0 else stagnation + 1
        events.append({"round": position, "action": list(action.slots), "rule_ids": list(action.rule_ids),
                       "reason": action.reason, "revealed": revealed, "step_gain": step_gain,
                       "best_quality": factor - 1, "remaining_budget": remaining})
        following = steps[position] if position < len(steps) else None
        continued = next((slot for slot, node in children.items()
                          if following is not None and node.id in following.parents), None)
        if rule == "truncate" and continued is not None and continued not in action.slots:
            unsupported += 1
            stop_reason = "continuation_not_selected"
            break
        if remaining < 1:
            stop_reason = "budget_exhausted"
            break
    return {"world_id": tree.id, "kernel": kernel_of(tree), "campaign": tree.tree.facets["geak"]["campaign"],
            "budget": budget, "best_quality": factor - 1, "charged_budget": budget - remaining,
            "attempted_nodes": attempted, "decision_batches": batches, "requested_attempts": requested,
            "unsupported_requests": unsupported, "support_complete": unsupported == 0,
            "stop_reason": stop_reason, "episode_rule": rule, "events": events}


def replay(world, policy, budget, episodes=None):
    if episodes is not None:
        if not isinstance(world, TreeView):
            raise ValueError("episode replay needs a discovery tree")
        return episode_replay(world, policy, budget, episodes)
    simulator = (TreeReplay if isinstance(world, TreeView) else ReplayWorld)(world, budget)
    while not simulator.done:
        simulator.step(policy.select(simulator.observe()))
    return simulator.trace()
