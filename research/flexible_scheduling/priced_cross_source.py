"""Use physical source identities and actual prices when forming a feasible incumbent."""
from .cross_source_workflow import CrossSourcePlanner
from .event_policy import order
from .event_workflow import ALWAYS, UNKNOWN, indices


class PricedCrossSourcePlanner(CrossSourcePlanner):
    def _greedy(self, state, style):
        if style=="cost":
            return super()._greedy(state,style)
        guards = self.model.guards(state.values)
        pending = [i for i,value in enumerate(state.values) if value==UNKNOWN and guards[i]]
        due = {i for i in pending if guards[i]==ALWAYS}

        def priority(dispatch):
            queried = {i for _,mask in dispatch.reads for i in indices(mask)}
            covered = due & queried
            sources = {self.m.queries[i].sites[0] for i in queried}
            downstream = sum(parent in {p for clause in guards[i] for p,_ in clause}
                             for parent in queried for i in pending)
            seconds = sum(self.model.read_ticks(source,mask) for source,mask in dispatch.reads)
            quality = (8*len(covered)+2.5*downstream+1.5*(len(queried)-len(covered)))/(max(1,seconds)**.33)
            movement = 0.
            if dispatch.motion:
                site,capture = dispatch.motion
                cost = self.m.travel[state.site][site]+(self.m.capture[site] if capture else 0)
                if capture:
                    site_fields = {i for i in pending if site in self.m.queries[i].sites and not state.raw&(1<<site)}
                    importance = (5*len(site_fields&due)+7*bool(self.m.mandatory_raw&(1<<site))
                                  +(1.5 if style=="prepare" else .3)*len(site_fields-due))
                elif site==self.m.finish:
                    missing_view = any(not state.raw&(1<<s) for i in pending for s in self.m.queries[i].sites)
                    importance = .08 if missing_view else 1.
                else:
                    importance = .15
                movement = importance/(1+cost/5)
            if style=="prepare" and not downstream:
                quality -= 2*sum(any(i not in due and i not in queried and s in self.m.queries[i].sites for i in pending)
                                 for s in sources)
            return -(quality+movement),-len(queried),-int(bool(dispatch.motion)),order(dispatch)

        return min(self.options(state),key=priority)
