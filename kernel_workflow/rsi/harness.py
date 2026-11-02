"""Knowledge-as-a-Policy and the shared online/replay scheduling state machine."""

from dataclasses import asdict, dataclass, replace

from kernelwiki.knowledge import (
    MAX_WIDTH,
    KernelState,
    KnowledgeError,
    KnowledgePolicy,
    KnowledgeRecord,
    digest,
    strict_fields,
    validate_policy,
)

INTERPRETER_VERSION = "knowledge-policy-v3"


@dataclass(frozen=True)
class ParentView:
    id: str
    node_kind: str
    measurement_status: str
    parent_gain: float | None
    created: int
    scheduled: int


@dataclass(frozen=True)
class Observation:
    state: KernelState
    parents: tuple[ParentView, ...]
    remaining_budget: int
    stagnation: int
    direction_cost: int
    worker_limit: int


@dataclass(frozen=True)
class Action:
    parents: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    reason: str = "baseline_rotation"


@dataclass(frozen=True)
class SlotView:
    """One planned direction as known before dispatch: its knowledge is what the plan pointed to."""
    slot_index: int
    specialty: str
    mode: str
    knowledge: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ExpansionObservation:
    state: KernelState
    round: int
    stagnation: int
    remaining_budget: int
    best_gain: float
    slots: tuple[SlotView, ...]


@dataclass(frozen=True)
class ExpansionAction:
    slots: tuple[int, ...] = ()
    rule_ids: tuple[str, ...] = ()
    reason: str = "dispatch_plan"

    @property
    def stop(self):
        return not self.slots


def slot_cost(slot):
    return 2 if slot.mode == "deep" else 1


class Rollout:
    def __init__(self, state, generation, root, budget):
        worker_limit = generation.worker_limit
        if type(budget) is not int or budget < 1 or worker_limit not in (1, 2):
            raise KnowledgeError("rollout requires a positive budget and one or two workers")
        self.state, self.generation, self.root = state, generation, root
        self.budget, self.remaining = budget, budget
        self.worker_limit = 1 if generation.mode == "deep_explore" else worker_limit
        self.round_limit = budget // generation.cost
        self.parents = {"root": ParentView("root", "root", "qualified", 0.0, 0, -1)}
        self.nodes = {"root": root}
        self.rounds = self.attempted = self.stagnation = 0
        self.best = 0.0
        self.events = []
        self.stop = "budget_exhausted" if self.round_limit == 0 else None

    def observe(self):
        return Observation(self.state, tuple(self.parents.values()), self.remaining, self.stagnation,
                           self.generation.cost, self.worker_limit)

    def validate(self, action):
        if self.stop:
            raise KnowledgeError("cannot act on a stopped rollout")
        if (not isinstance(action, Action) or len(action.parents) != len(set(action.parents))
                or len(action.parents) > self.worker_limit
                or len(action.parents) * self.generation.cost > self.remaining
                or any(parent not in self.parents for parent in action.parents)):
            raise KnowledgeError("illegal rollout action")

    def step(self, action, outcomes):
        self.validate(action)
        if len(outcomes) != len(action.parents):
            raise KnowledgeError("a batch must return exactly one outcome per selected parent")
        observation = self.observe()
        if not action.parents:
            self.stop = "policy_stop"
            self.events.append({"observation": asdict(observation), "action": asdict(action),
                                "revealed": [], "unsupported": [], "best_quality": self.best,
                                "remaining_budget": self.remaining})
            return
        for parent, node in zip(action.parents, outcomes):
            if node is not None:
                source = self.nodes[parent]
                if not node.continues(source):
                    raise KnowledgeError("transition does not continue the selected source and context")
                if node.measurement.status == "qualified" and node.measurement.contract != self.root.measurement.contract:
                    raise KnowledgeError("transition changed the root evaluation contract")
        self.rounds += 1
        previous_best = self.best
        revealed, unsupported = [], []
        for parent, node in zip(action.parents, outcomes):
            self.parents[parent] = replace(self.parents[parent], scheduled=self.rounds)
            if node is None:
                unsupported.append(parent)
                continue
            self.attempted += 1
            self.remaining -= self.generation.cost
            visible_id = f"node-{self.attempted}"
            measurement = node.measurement
            gain = (measurement.primary_score / self.root.measurement.primary_score - 1
                    if measurement.status == "qualified" else None)
            if gain is not None:
                self.best = max(self.best, gain)
            if parent != "root":
                del self.parents[parent]
            self.nodes[visible_id] = node
            self.parents[visible_id] = ParentView(visible_id, "leaf", measurement.status,
                                                 gain, self.rounds, self.rounds)
            revealed.append({"visible_id": visible_id, "node_id": node.id, "attempt_id": node.attempt_id,
                             "parent": parent, "status": measurement.status, "quality": gain,
                             "cost": self.generation.cost})
        self.stagnation = 0 if self.best > previous_best else self.stagnation + 1
        self.events.append({"observation": asdict(observation), "action": asdict(action),
                            "revealed": revealed, "unsupported": unsupported,
                            "best_quality": self.best, "remaining_budget": self.remaining})
        if self.remaining < self.generation.cost:
            self.stop = "budget_exhausted"
        elif self.rounds >= self.round_limit:
            self.stop = "round_limit"

    def trace(self, world_id, kernel, campaign):
        unsupported = sum(len(event["unsupported"]) for event in self.events)
        return {"world_id": world_id, "kernel": kernel, "campaign": campaign, "budget": self.budget,
                "best_quality": self.best, "charged_budget": self.budget - self.remaining,
                "attempted_nodes": self.attempted, "decision_batches": self.rounds,
                "requested_attempts": sum(len(event["action"]["parents"]) for event in self.events),
                "unsupported_requests": unsupported, "support_complete": unsupported == 0
                and not (self.stop == "support_exhausted" and self.remaining >= self.generation.cost),
                "stop_reason": self.stop, "events": list(self.events)}


class Policy:
    def __init__(self, knowledge_policy=None, records=(), *, neutral=False):
        self.knowledge = knowledge_policy or KnowledgePolicy()
        self.records = tuple(records)
        self.neutral = neutral
        validate_policy(self.knowledge, self.records)

    @classmethod
    def from_dict(cls, value, reference_records=()):
        strict_fields(value, {"policy", "records"}, {"policy", "records"})
        records = {record.id: record for record in reference_records}
        changed = tuple(KnowledgeRecord.from_dict(item) for item in value["records"])
        if len({record.id for record in changed}) != len(changed):
            raise KnowledgeError("duplicate serialized policy record")
        for record in changed:
            if record.policy_rule is None or (record.id in records and records[record.id].policy_rule is None):
                raise KnowledgeError("working policy cannot change reference knowledge")
            records[record.id] = record
        return cls(KnowledgePolicy.from_dict(value["policy"]), tuple(records.values()))

    def to_dict(self):
        return {"policy": self.knowledge.to_dict(),
                "records": [record.to_dict() for record in self.records if record.policy_rule is not None]}

    @property
    def revision(self):
        return digest(self.to_dict())

    def select_expansion(self, observation):
        """Policy schema 2: stop, then skip, then select, then width, within the remaining budget.
        Without rules every planned direction is dispatched, which is what the recorded lane did."""
        if self.knowledge.schema_version == 1 and self.knowledge.rule_ids:
            raise KnowledgeError("schema-1 rules rank parents; the expansion interpreter needs policy schema 2")
        rules = [] if self.neutral else [record for record in self.records
                 if record.id in self.knowledge.rule_ids and record.matches(observation.state)]
        rules.sort(key=lambda record: (-(len(record.applies_to) + len(record.policy_rule.when)), record.id))
        rules = rules[:self.knowledge.top_k]
        values = {"stagnation": observation.stagnation, "remaining_budget": observation.remaining_budget,
                  "round": observation.round, "best_gain": observation.best_gain}
        holds = lambda record, extra=None: all(condition.matches({**values, **(extra or {})})
                                               for condition in record.policy_rule.when)
        for record in rules:
            if record.policy_rule.effect == "stop" and holds(record):
                return ExpansionAction((), (record.id,), "matched_stop")
        used, chosen = [], []
        for slot in observation.slots:
            fields = {"slot_index": slot.slot_index, "specialty": slot.specialty, "mode": slot.mode,
                      "knowledge": slot.knowledge}
            skipped = [record.id for record in rules if record.policy_rule.effect == "skip" and holds(record, fields)]
            selected = [record.id for record in rules if record.policy_rule.effect == "select" and holds(record, fields)]
            used += skipped
            if not skipped:
                chosen.append((slot, selected))
        if any(selected for _, selected in chosen):
            chosen = [(slot, selected) for slot, selected in chosen if selected]
            used += [rule_id for _, selected in chosen for rule_id in selected]
        width = self.knowledge.max_width or MAX_WIDTH
        for record in rules:
            if record.policy_rule.effect == "width" and holds(record):
                width = record.policy_rule.value
                used.append(record.id)
                break
        dispatched, budget = [], observation.remaining_budget
        for slot, _ in chosen[:width]:
            if slot_cost(slot) <= budget:
                dispatched.append(slot.slot_index)
                budget -= slot_cost(slot)
        if not dispatched:
            return ExpansionAction((), tuple(dict.fromkeys(used)), "nothing_dispatchable")
        return ExpansionAction(tuple(dispatched), tuple(dict.fromkeys(used)),
                               "matched_rules" if used else "dispatch_plan")

    def select(self, observation, knowledge_snapshot=None):
        if knowledge_snapshot is not None:
            return Policy(knowledge_snapshot.policy, knowledge_snapshot.records,
                          neutral=self.neutral).select(observation)
        if self.knowledge.schema_version != 1:
            raise KnowledgeError("policy schema 2 decides expansions; the parent scheduler runs schema 1 only")
        if not observation.parents or observation.remaining_budget < observation.direction_cost:
            return Action(reason="no_budget_or_support")
        rules = [] if self.neutral else [record for record in self.records
                 if record.id in self.knowledge.rule_ids and record.matches(observation.state)]
        rules.sort(key=lambda record: (-(len(record.applies_to) + len(record.policy_rule.when)), record.id))
        rules = rules[:self.knowledge.top_k]
        global_values = {"stagnation": observation.stagnation,
                         "remaining_budget": observation.remaining_budget}
        for record in rules:
            if record.policy_rule.effect == "stop" and all(
                    condition.matches(global_values) for condition in record.policy_rule.when):
                return Action(rule_ids=(record.id,), reason="matched_stop")
        ranked = []
        for parent in observation.parents:
            values = {**global_values, **asdict(parent)}
            matched = next((record for record in rules if record.policy_rule.effect != "stop"
                            and all(condition.matches(values) for condition in record.policy_rule.when)), None)
            tier = {"prefer": 0, "defer": 2}[matched.policy_rule.effect] if matched else 1
            ranked.append([tier, parent.scheduled, parent.created, parent.id, matched.id if matched else None])
        count = min(self.knowledge.workers, observation.worker_limit,
                    observation.remaining_budget // observation.direction_cost)
        selected, matched_ids = [], []
        for _ in range(count):
            if not ranked:
                break
            ranked.sort(key=lambda row: row[:4])
            chosen = ranked[0]
            selected.append(chosen[3])
            if chosen[4] and chosen[4] not in matched_ids:
                matched_ids.append(chosen[4])
            ranked.pop(0)
        return Action(tuple(selected), tuple(matched_ids),
                      "matched_preference" if matched_ids else "baseline_rotation")
