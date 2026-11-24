"""Discrete event driver for explicitly controlled, synthetic diagnostics."""

import json
from dataclasses import asdict
from hashlib import sha256
from time import monotonic

from .evidence import Frame, Observation, accept
from .executor import Snapshot, async_action, permitted_observation, terminal_action
from .fixtures import MODE, Diagnostic
from .queries import QueryBinding, QueryBook, QueryReply
from .search import Search


METHODS = ("ASYNC", "PREFETCH", "A")
PROVENANCE = "controlled_symbolic_observation"


def visible_state_hash(task, state):
    encoded = json.dumps({"task": asdict(task), "state": asdict(state)},
                         sort_keys=True, ensure_ascii=False, allow_nan=False)
    return sha256(encoded.encode("utf8")).hexdigest()


class Episode:
    def __init__(self, case: Diagnostic, method: str, limits=None):
        if method not in METHODS:
            raise ValueError("Unknown diagnostic method " + method)
        self.case = case
        self.task = case.task
        self.method = method
        self.at = 0.0
        self.site = self.task.start
        self.evidence = {}
        self.attempts = {}
        self.events = []
        self.book = QueryBook()
        self.inflight = []
        self.requirements = {item.id: item for item in self.task.requirements}
        self.search = Search(self.task, **(limits or {}))

    def emit(self, kind, **fields):
        self.events.append({"event": kind, "at": self.at, **fields})

    def snapshot(self):
        return Snapshot(self.at, self.site, dict(self.book.results),
                        dict(self.evidence), dict(self.attempts))

    def advance(self, target):
        while self.inflight and self.inflight[0][0] <= target:
            when, reply = self.inflight.pop(0)
            self.at = when
            accepted = self.book.accept(reply, when)
            self.emit("query_returned", query_id=reply.binding.id, outcome=reply.outcome,
                      accepted=accepted, input_hash=reply.binding.input_hash)
        self.at = target

    def frame(self, object_id, identity):
        content = json.dumps([identity, object_id, self.at], separators=(",", ":")).encode("utf8")
        return Frame(identity, object_id, 0, self.at, content, PROVENANCE)

    def observe(self, requirement_id, frame, expiry):
        requirement = self.requirements[requirement_id]
        observation = Observation(frame, requirement.field,
                                  self.case.material_values[requirement_id], True, expiry)
        self.emit("observation_captured", requirement_id=requirement_id,
                  object_id=frame.object_id, field=observation.field, value=observation.value,
                  frame_id=frame.id, frame_hash=frame.digest, valid_until=expiry,
                  provenance=frame.provenance, readable=observation.readable)
        evidence = accept(requirement, observation, self.at)
        if evidence is None:
            self.emit("evidence_rejected", requirement_id=requirement_id,
                      observed_object_id=frame.object_id, frame_id=frame.id)
            return
        self.evidence[requirement_id] = evidence
        self.emit("evidence_accepted", evidence=asdict(evidence))

    def initialize(self):
        self.emit("episode_started", mode=MODE, site=self.site, deadline=self.task.deadline)
        for key, expiry in self.case.initial_material:
            self.observe(key, self.frame(self.requirements[key].object_id, "initial-" + key), expiry)
        truth = {row.id: row for row in self.case.query_truth}
        for query in self.task.queries:
            observed = self.frame(query.object_id, "source-" + query.id)
            binding = QueryBinding.bind(query, observed, self.task.instruction, self.at)
            self.book.register(binding)
            schedule = truth[query.id]
            if not self.at <= schedule.ready_at <= query.latest_ready:
                raise ValueError("Fixture violates query timing contract")
            self.inflight.append((schedule.ready_at, QueryReply(binding, schedule.outcome)))
            self.emit("query_submitted", query=asdict(binding))
        self.inflight.sort(key=lambda pair: pair[0])
        self.advance(self.at)

    def act(self, action, reason):
        state = self.snapshot()
        if action.skill is None:
            current = terminal_action(self.task, state)
        else:
            current = permitted_observation(self.task, state, action.skill)
        if current != action:
            self.emit("action_rejected", reason="state_changed")
            return
        skill_id = action.skill.id if action.skill else None
        self.emit("action_started", skill_id=skill_id, origin=self.site, target=action.site,
                  reason=reason, queries_pending=sorted(self.book.pending))
        self.advance(self.at + action.seconds)
        self.site = action.site
        self.emit("action_finished", skill_id=skill_id, site=self.site)
        if action.skill is None:
            self.emit("terminal_reached", site=self.site)
            return
        skill = action.skill
        attempt = self.attempts.get(skill.id, 0) + 1
        self.attempts[skill.id] = attempt
        expiry = self.at + skill.fresh_for if skill.fresh_for is not None else None
        for key in skill.covers:
            observed_object = self.requirements[key].object_id
            if skill.id in self.case.reject_first and attempt == 1:
                observed_object = "different-object"
            frame = self.frame(observed_object, f"{skill.id}-{attempt}-{observed_object}")
            self.observe(key, frame, expiry)

    def prepare(self, state):
        self.emit("proposal_started", visible_state_hash=visible_state_hash(self.task, state))
        started = monotonic()
        found = self.search.propose(state, shared_prefetch=self.method == "PREFETCH")
        elapsed = monotonic() - started
        self.advance(min(self.at + elapsed, self.task.deadline))
        candidate = found.choice
        gain = found.baseline_finish - candidate.expected_finish if candidate else None
        self.emit("proposal_finished", spent_seconds=elapsed, expanded=found.expanded,
                  witnessed=found.witnessed, baseline_finish=found.baseline_finish,
                  expected_gain=gain, exhausted_budget=found.exhausted_budget,
                  proposed=[action.skill.id for action in candidate.prepare] if candidate else [],
                  required_evidence=list(candidate.required_evidence) if candidate else [])
        if self.book.results != state.results:
            return True
        if candidate and gain > elapsed:
            self.act(candidate.prepare[0], self.method)
            return True
        return False

    def report(self):
        completed = self.at + self.task.report_seconds
        requirements = [item for item in self.task.requirements if item.applies(self.book.results)]
        state = self.snapshot()
        if completed > self.task.deadline or not all(state.supports(item, completed) for item in requirements):
            return False
        self.advance(completed)
        self.emit("report_delivered", findings={item.id: self.evidence[item.id].value
                                                for item in requirements if item.report})
        return True

    def run(self):
        self.initialize()
        while self.at <= self.task.deadline:
            state = self.snapshot()
            if self.site == self.task.terminal and not self.book.pending:
                if self.report():
                    return self.events
                break
            action = async_action(self.task, state)
            if action is not None:
                self.act(action, "ordinary")
                continue
            if self.method != "ASYNC" and self.book.pending and self.site != self.task.terminal:
                if self.prepare(state):
                    continue
            if not self.inflight or self.inflight[0][0] > self.task.deadline:
                break
            until = self.inflight[0][0]
            self.emit("blocked_wait", until=until, site=self.site,
                      queries_pending=sorted(self.book.pending))
            self.advance(until)
        self.emit("episode_incomplete", site=self.site)
        return self.events
