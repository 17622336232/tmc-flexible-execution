"""Development-only factor tests. No camera, network, robot hardware, or test pool."""
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import permutations, product
from statistics import mean
from time import monotonic
import json

from experiments.flexible_execution.evaluate import score
from experiments.flexible_execution.fixtures import Diagnostic, QueryFixture
from experiments.flexible_execution.gpt_vision import OUTCOMES
from experiments.flexible_execution.live_planning import select
from experiments.flexible_execution.revision_audit import OUT, save, write_table
from experiments.flexible_execution.simulation import Episode
from experiments.flexible_execution.task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract


TYPES = ("independent", "conditional", "forbidden", "expensive", "reuse", "shared",
         "multi_pending", "uncertain", "no_opportunity", "failed_capture")
METHODS = ("conditional-async", "conditional-prefetch", "conditional-planner", "common-anchor", "adaptive-progress")


def fixture(kind, delay, move, permitted):
    q = QuerySpec("front", "subject", OUTCOMES, delay, delay + 2)
    cond = Condition(q.id, ("positive",))
    # Values and realized replies are visible only to the event driver and evaluator.
    outcome = "uncertain" if kind == "uncertain" else "negative" if kind == "no_opportunity" else "positive"
    queries = [q]
    truth = [QueryFixture(q.id, delay, outcome)]
    needs = [Requirement("front-view", "subject", "front")]
    initial = [("front-view", None)]
    skills = []
    side = "far" if kind == "expensive" else "side"
    if kind != "no_opportunity":
        condition = None if kind == "independent" else cond
        if kind == "uncertain":
            condition = Condition(q.id, ("uncertain",))
        needs.append(Requirement("label", "subject", "label", condition, report=True))
        gate = (condition or Condition(q.id, OUTCOMES)) if not permitted or kind == "forbidden" else None
        skills = [Skill("label-view", side, ("label",), 1, gate), Skill("other-label-view", "alternate", ("label",), 1.7, gate)]
        if kind == "reuse":
            initial.append(("label", None))
        if kind in ("shared", "multi_pending"):
            extra_cond = cond
            if kind == "multi_pending":
                extra = QuerySpec("surface", "subject", OUTCOMES, delay + 3, delay + 5)
                queries.append(extra)
                truth.append(QueryFixture(extra.id, extra.estimated_ready, "negative"))
                extra_cond = Condition(extra.id, ("negative",))
            needs.append(Requirement("mark", "subject", "mark", extra_cond, report=True))
            extra_gate = extra_cond if not permitted else None
            skills.append(Skill("mark-view", "alternate", ("mark",), 1, extra_gate))
            # A shared exposure that would violate either independent gate is unavailable.
            if permitted:
                skills.append(Skill("combined-view", side, ("label", "mark"), 1.2))
    locations = {"front": 0, "side": 1, "alternate": 3, "far": -16, "finish": 5}
    routes = tuple((a, b, move * abs(locations[a]-locations[b])) for a,b in permutations(locations,2))
    task = TaskContract("Inspect the designated subject within the declared observation contract.", "front", "finish",
                        120., .8, tuple(needs), tuple(queries), tuple(skills), routes)
    return Diagnostic(f"{kind}-d{delay}-m{move}-p{int(permitted)}", kind, "synthetic-subject",
                      task, tuple(truth), tuple(initial), {r.id: r.field + "-observed" for r in needs},
                      ("label-view",) if kind == "failed_capture" else ())


class DevelopmentEpisode(Episode):
    def __init__(self, case, method):
        super().__init__(case, "ASYNC" if method == "conditional-async" else "A")
        self.policy = method

    def prepare(self, state):
        t = monotonic()
        choice = select(self.task, state, "conditional-a" if self.policy == "common-anchor" else self.policy,
                        lambda *args, **kwargs: None)
        elapsed = monotonic() - t
        self.advance(min(self.at + elapsed, self.task.deadline))
        self.emit("proposal_finished", spent_seconds=elapsed)
        if state.results != self.book.results:
            return True
        if choice:
            self.act(choice.prepare[0], self.policy)
            return True
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    trace_path = OUT / "symbolic_development.jsonl"
    with trace_path.open("w", encoding="utf8") as stream:
        for kind, delay, move, allowed in product(TYPES, (2, 7, 13), (.5, 1.5), (False, True)):
            case = fixture(kind, delay, move, allowed)
            for method in METHODS:
                events = DevelopmentEpisode(case, method).run()
                verdict = score(case, events)
                stream.write(json.dumps(dict(case=case.id, method=method, contract=asdict(case.task), events=events))+"\n")
                rows.append(dict(case=case.id, type=kind, latency=delay, unit_travel=move, scope_grant=allowed,
                                 method=method, **verdict))
    write_table(OUT / "symbolic_development.csv", rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["method"]].append(r)
    summary = {k: dict(attempts=len(rs), passes=sum(r["success"] for r in rs),
                       errors=sum(bool(r["trace_errors"]) for r in rs),
                       restricted_model_seconds=mean(r["restricted_completion_seconds"] for r in rs),
                       measured_search_seconds=mean(r["proposal_wall_seconds"] for r in rs)) for k,rs in groups.items()}
    save(OUT / "symbolic_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
