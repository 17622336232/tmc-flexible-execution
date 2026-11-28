"""Run source-scoped scheduling inside the ordinary GPT task control loop."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from math import isfinite
from statistics import median
from time import monotonic, sleep

from experiments.flexible_execution.adapters.original_controller import upstream
from experiments.task_attack.agent import run_agent
from experiments.task_attack.runtime import append_json
from .cross_source_workflow import CrossSourceWorkflow
from .event_workflow import ALWAYS, indices
from .native_frontier_adapter import NativeFrontierObserver, scheduling_state
from .natural_frontier import released, source_map
from .ordinary_frontier import camera_goals, pose_matches
from .priced_cross_source import PricedCrossSourcePlanner


class NativePlanner(PricedCrossSourcePlanner):
    def _options(self, state):
        candidates=super()._options(state)
        # The next native observation after moving to a new viewpoint takes a real photo.
        # The joint controller cannot advertise transit through an unlicensed camera waypoint.
        return tuple(a for a in candidates if not (a.motion and not a.motion[1]
                     and not state.raw&(1<<a.motion[0])))


class PendingCamera:
    """Retain a successful post-move exposure for the next ordinary observation."""
    def __init__(self, backend, trace):
        self.backend=backend
        self.trace=trace
        self.pending=None

    def capture(self, step, goal):
        started=monotonic()
        try:
            frame,geometry=self.backend.observe()
            pose={**geometry["position"],"yaw":geometry["yaw"]}
            if not pose_matches(pose,goal,metres=.10,radians_tol=.10):
                raise ValueError("Captured viewpoint differs from the permitted destination")
        except (RuntimeError, ValueError, TimeoutError) as error:
            self.trace.event("joint_capture_unavailable",step,error=type(error).__name__)
            return
        self.pending=(frame,geometry)
        self.trace.event("joint_pending_capture",step,started_monotonic_seconds=started,
                         image_sha256=sha256(frame["data"]).hexdigest(),
                         observed_pose=pose,capture_ros=frame["ros_stamp"])

    def observe(self):
        pending,self.pending=self.pending,None
        if pending is not None:
            return pending
        return self.backend.observe()


class NativeJointExecution:
    def __init__(self, task, directory, trace, mode, prices, camera=None):
        self.task=task
        self.directory=directory
        self.trace=trace
        self.mode=mode
        self.price=prices
        self.observer=NativeFrontierObserver(task)
        self.prompt=upstream().SYSTEM
        self.records={}
        self.photos={}
        self.scene=None
        self.anchor=None
        self.enabled=self.observer.plan is not None
        self.move_times={}
        self.service_times={}
        self.pending_motion=None
        self.camera=camera

    def note(self, kind, step, **event):
        self.trace.event(kind,step,**event)

    def observe(self, context, images):
        photo=context["current_image_id"]
        geometry=context["geometry"]
        self.anchor=self.anchor or geometry
        current=dict(image_id=photo,position=geometry["position"],yaw=geometry["yaw"])
        names=[*context["recent_images"],current]
        by_id={n["image_id"]:(n,data) for n,data in zip(names,images)}
        if len(names)!=len(images) or len(by_id)!=len(names):
            self.enabled=False
            return None
        scene=context.get("scene_conditions") or {}
        version=scene.get("state_version")
        if self.scene is None:
            self.scene=(version,)
        if self.scene!=(version,) or scene.get("dynamic"):
            self.enabled=False
            return None
        self.observer.anchor=self.anchor
        views=source_map(self.observer.plan,self.anchor,names)
        if photo not in views:
            self.enabled=False
            return None
        if self.pending_motion:
            start,target,at=self.pending_motion
            if views[photo]==target and (context["last_tool_feedback"] or {}).get("ok"):
                edges=("front","side")
                key=(edges.index(start),edges.index(target))
                self.move_times.setdefault(key,[]).append(max(.01,monotonic()-at-self.price["photo_s"]))
            self.pending_motion=None
        for name,view in views.items():
            meta,blob=by_id[name]
            digest=sha256(blob["data"]).hexdigest()
            previous=self.photos.get(name)
            if previous is not None and previous!=(view,digest):
                self.enabled=False
                return None
            self.photos[name]=(view,digest)
        return views,by_id,views[photo]

    def estimate(self, readings, key, default):
        values=readings.get(key,())
        # A few in-episode successful timings move the median only after a return.
        return median([default,default,*values[-4:]])

    def planned(self, views, here, remaining):
        travel=self.price["travel_s"]
        def ticks(seconds):
            return max(1,round(100*seconds))
        directed=tuple(tuple(0 if i==j else ticks(self.estimate(self.move_times,(i,j),travel[i][j]))
                             for j in range(2)) for i in range(2))
        costs={k:self.estimate(self.service_times,k,self.price[k+"_s"]) for k in ("single","same","cross")}
        m,state=scheduling_state(self.observer,set(views.values()),here,directed,
                (ticks(self.price["photo_s"]),)*2,(0,ticks(costs["single"]),ticks(costs["same"])))
        allow_cross=costs["cross"]<remaining and self.price.get("cross_enabled",False)
        graph=CrossSourceWorkflow(m,ticks(costs["cross"]) if allow_cross else None)
        # Early physical permission remains a user property, independent of the planner selected.
        p=NativePlanner(graph,quotient=self.mode!="generic",ablation="serial" if self.mode=="common-serial" else None)
        if graph.complete(state):
            return None, True,costs
        if self.mode=="strong-prefetch":
            cost,action=p.strong(state)
        else:
            result=p.solve(state,budget=256)
            cost,action=result["value_ticks"],result["action"]
        return action,False,{**costs,"expected_remaining_s":cost/100}

    def authorized_motion(self, action, view):
        if action is None or not action.motion:
            return None
        plan=self.observer.plan
        dest,_=action.motion
        target=("front","side")[dest]
        if view==target:
            return None
        legal=released(plan,self.observer.prior)
        if target=="side" and not plan.early_capture:
            # Without early permission defer the side trip until no open front
            # root can affect whether the side observation is required.
            before_root=any(f.view=="front" and f.any_of==((),) and i not in self.observer.prior
                            for i,f in enumerate(plan.fields))
            required_side=any(plan.fields[i].view=="side" for i in legal)
            completed_for_final=not legal and plan.finish=="side"
            if before_root or action.reads or not (required_side or completed_for_final):
                return None
        if target=="front" and action.reads:
            # Preserve one-shot final execution semantics until return is calibrated.
            return None
        goal=camera_goals(dict(direction=plan.direction,angle_degrees=plan.angle_degrees),
                         self.anchor["position"],self.anchor["yaw"])[target]
        return dict(type="move",goal=goal)

    def scope(self, action, views, current):
        if action is None or not action.reads:
            return {}
        allowed=released(self.observer.plan,self.observer.prior)
        chosen={}
        for _,mask in action.reads:
            for i in indices(mask):
                field=self.observer.plan.fields[i]
                if i not in allowed:
                    raise ValueError("Planner tried to read a condition that has not returned")
                matches=[n for n,v in views.items() if v==field.view]
                if not matches:
                    raise ValueError("No source image for the scheduled reading")
                chosen[i]=current if current in matches else matches[-1]
        return chosen

    def bind(self, reply, scope, step):
        if not isinstance(reply,dict):
            return reply
        plan=self.observer.plan
        raw=reply.get("records")
        all_rows=list(raw) if isinstance(raw,list) else []
        if (reply.get("action") or {}).get("type")=="submit":
            report=reply.get("report")
            if isinstance(report,dict) and isinstance(report.get("records"),list):
                all_rows+=report["records"]
        accepted={}
        conflict=set()
        for row in all_rows:
            if not isinstance(row,dict) or not isinstance(row.get("facts"),dict):
                continue
            value=row["facts"].get("visible")
            if type(value) is not bool:
                continue
            identity=row.get("object_id")
            photo=row.get("evidence_image_id")
            matching=[i for i in scope if plan.fields[i].name==identity and scope[i]==photo]
            if len(matching)==1:
                i=matching[0]
                if i in accepted and accepted[i]["facts"]["visible"]!=value:
                    conflict.add(i)
                accepted[i]=row
            elif not any(r.get("object_id")==identity and r.get("evidence_image_id")==photo
                         and r["facts"]["visible"]==value for r in self.records.values()):
                self.note("joint_claim_withheld",step,field=str(identity),image=str(photo))
        staged={i:r for i,r in accepted.items() if i not in conflict}
        for i,row in staged.items():
            self.observer.prior[i]=row["facts"]["visible"]
            self.records[i]=row
            photo=scope[i]
            self.note("joint_fact",step,field=plan.fields[i].name,index=i,value=self.observer.prior[i],
                      image_id=photo,image_sha256=self.photos[photo][1])
        return {**reply,"records":[{**row,"facts":{"visible":row["facts"]["visible"]}} for row in staged.values()]}

    def fitting_report(self, report, _records, _step):
        if not self.enabled or report.get("outcome")!="accepted":
            return None
        if released(self.observer.plan,self.observer.prior):
            return "At least one permitted required finding is still unresolved."
        def key(row):
            return (row.get("object_id"),row.get("evidence_image_id"),
                    row.get("facts",{}).get("visible"))
        reported=report.get("records",[])
        keys=[key(r) for r in reported]
        valid={key(r) for r in self.records.values()}
        if len(keys)!=len(valid) or any(type(k[-1]) is not bool or k not in valid for k in keys) or set(keys)!=valid:
            return "Report only the complete bound findings and their actual source photos."
        if self.here!=self.observer.plan.finish:
            return "Return to the requested final viewpoint before submitting."
        return None

    def decide(self, model, context, images, *, execute, trace, step, image_sha256):
        if not self.enabled:
            return model.request(self.prompt,context,images),None
        observed=self.observe(context,images)
        if observed is None:
            return model.request(self.prompt,context,images),None
        views,by_id,here=observed
        self.here=here
        remaining=context["execution_clock"]["remaining_seconds"]
        action,complete,priced=self.planned(views,here,remaining)
        move=self.authorized_motion(action,here)
        current=context["current_image_id"]
        selected=self.scope(action,views,current)
        self.note("joint_plan",step,mode=self.mode,move=move,prices=priced,fields=list(selected),
                  prior=dict(self.observer.prior),current_image_id=current,source_sha256=image_sha256)
        if move and not selected:
            # The common agent executes and measures this authorized move, then makes a
            # fresh observation. No model reply or semantic value is manufactured.
            self.pending_motion=(here,"side" if here=="front" else "front",monotonic())
            return dict(action=move,records=[]),None
        if not selected and not complete:
            self.enabled=False
            self.note("joint_native_fallback",step,reason="no authorized dispatch")
            return model.request(self.prompt,context,images),None
        ask=[dict(field=self.observer.plan.fields[i].name,evidence_image_id=src) for i,src in selected.items()]
        packet={**context,"this_request_scope":{
            "open_for_reading":ask,"allowed_final_pose":self.observer.plan.finish,
            "observed_facts":list(self.records.values()),
            "rule":"Record new findings only from the listed photos. A new result cannot authorize a dependent finding in this same reply. Use the exact named subject and Boolean facts.visible. You may submit in this reply if the listed findings complete all required work, you are already at the final pose, and the report cites every confirmed source."}}
        if complete:
            packet["this_request_scope"]["rule"]+=" All active checks have now returned. Submit the complete report from confirmed records."
        with_scope={*selected.values(),current}
        scoped=[(name,meta,blob) for name,(meta,blob) in by_id.items() if name in with_scope]
        packet["recent_images"]=[meta for name,meta,_ in scoped if name!=current]
        attachments=[blob for _,_,blob in scoped]
        begin=monotonic()
        before=model.calls
        feedback=None
        if move:
            with ThreadPoolExecutor(max_workers=1) as worker:
                future=worker.submit(model.request,self.prompt,packet,attachments,attempt_limit=1)
                while model.calls==before and not future.done() and monotonic()<model.deadline_monotonic:
                    sleep(.005)
                if model.calls>before and not future.done():
                    started=monotonic()
                    self.note("joint_motion_during_request",step,action=move)
                    feedback=execute(move)
                    took=monotonic()-started
                    target=1 if here=="front" else 0
                    if feedback.get("ok"):
                        self.move_times.setdefault((1-target,target),[]).append(took)
                        if self.camera is not None and not future.done():
                            self.camera.capture(step,move["goal"])
                reply=future.result()
        else:
            reply=model.request(self.prompt,packet,attachments,attempt_limit=1)
        finished=monotonic()
        captured=len(set(selected.values()))
        shape="single" if len(selected)==1 else "same" if captured==1 else "cross"
        if selected and isinstance(reply,dict):
            # Use the actual transport log clock when available, so concurrent movement
            # does not get folded into the inference cost.
            from experiments.flexible_execution.timeline import read_trace_rows
            log=getattr(model,"log",None)
            rows=read_trace_rows(log)[0] if log and log.is_file() else []
            last=rows[-1] if rows else {}
            t0,t1=last.get("dispatched_monotonic_seconds"),last.get("return_monotonic_seconds")
            elapsed=t1-t0 if t0 is not None and t1 is not None else finished-begin
            if isfinite(elapsed) and elapsed>0:
                self.service_times.setdefault(shape,[]).append(elapsed)
        append_json(self.directory/"joint_native_replies.jsonl",dict(step=step,reply=reply))
        decision=self.bind(reply,selected,step)
        if feedback is not None:
            # The reply was planned from the old pose. Preserve its source-bound evidence
            # and reconsider its next action from the freshly measured actual pose.
            return {**decision,"action":move},feedback
        native_action=(decision.get("action") or {}).get("type")
        if native_action in ("move","approach"):
            # The next ordinary observation is attached to the actual pose of
            # this turn. Only the separately admitted planner action can alter it.
            self.note("joint_native_action_deferred",step,action=native_action)
            return {**decision,"action":dict(type="record")},None
        return decision,None


def run_joint(backend,task,directory,event,timeout,*,model,trace,mode,prices,retain_pending_photo=False):
    camera=PendingCamera(backend,trace) if retain_pending_photo else None
    controller=NativeJointExecution(task,directory,trace,mode,prices,camera=camera)
    if not controller.enabled:
        from experiments.flexible_execution.adapters.original_controller import run
        return run(backend,task,directory,event,timeout,model=model,trace=trace)
    return run_agent(backend,task,directory,event,timeout,model=model,trace=trace,
                     decision_pipeline=controller.decide,submit_guard=controller.fitting_report,
                     observe_next=camera.observe if camera is not None else None)
