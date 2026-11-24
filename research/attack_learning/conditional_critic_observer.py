"""Online public-gateway observer for the unpromoted approximate candidate."""
import argparse
import json
from pathlib import Path
import pickle
import threading
import time
import urllib.request

from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import append_json,write_json
from experiments.task_attack_value.programs import known_program
from experiments.task_attack_value.vision import enrich_window
from conditional_critic_policy import Policy,MODES


class PublicCollector:
    def __init__(self,url,initial,dynamic):
        self.url=url;self.dynamic=dynamic;self.history=enrich_window(initial,{});self.lock=threading.Lock();self.stop=threading.Event();self.requests=0;self.errors=0;self.processing_seconds=0.
        self.thread=threading.Thread(target=self.collect,daemon=True)
    def collect(self):
        while not self.stop.is_set() and self.dynamic:
            start=time.monotonic()
            try:
                window=_get(self.url,'frames');processing=time.monotonic()
                with self.lock:
                    enrich_window(window,self.history);self.requests+=1;self.processing_seconds+=time.monotonic()-processing
            except Exception:self.errors+=1
            self.stop.wait(max(.01,.5-(time.monotonic()-start)))
    def snapshot(self):
        with self.lock:return [dict(f) for f in sorted(self.history.values(),key=lambda x:(x['capture_ros'],x['id']))]


def run(a):
    root=a.output;root.mkdir(parents=True,exist_ok=True);bundle=json.loads(a.timing_model.read_text());candidate=bundle['conditional_critic_candidate'];perception=json.loads(a.public_perception.read_text())
    if (a.policy=='belief_known_task')!=(a.known_task_file is not None):raise ValueError('Information condition mismatch')
    catalog=_get(a.gateway,'catalog');initial=_get(a.gateway,'initial');collector=PublicCollector(a.gateway,initial,a.policy!='belief_blind');collector.thread.start()
    with Path(candidate['model']).open('rb') as f:model=pickle.load(f)
    known=known_program(a.known_task_file.read_text(),bundle['programs'],perception['stations']) if a.known_task_file else None
    if a.known_task_file and known is None:raise ValueError('Authorized task outside supported arguments')
    if a.policy=='redvla_external':
        from common_resource_red_anchor import Policy as RedPolicy
        predictor=RedPolicy(model,bundle['legacy_onset'],perception,catalog['editable_actions'])
    else:
        with Path(candidate['stopping'][MODES[a.policy]]).open('rb') as f:stopping=pickle.load(f)['head']
        predictor=Policy(model,stopping,bundle['operation_model']['pose_heads'],perception,catalog['editable_actions'],a.policy,known)
    summary=dict(policy=a.policy,controller='learned_read_dependency_conditional_critic_and_stopping',candidate_not_promoted=True,
        approximate_inference=True,full_paper_KDE_reproduction=False,live_inference_api_requests=0,shared_static_preparation_requests=1,
        information='initial actual RGB and public clock' if a.policy=='belief_blind' else 'actual public external RGB stream, c0 and public clock',
        critic_target='OWN clean-route counterfactual acquisition/adoption proxy; actual attack outcome is scored separately by unchanged native runner',decision='collecting',calls=0)
    if a.policy=='redvla_external':summary.update(controller='original_trajectory_anchor_adaptation_with_common_static_head',paper_adaptation=True)
    write_json(root/'result.json',summary);next_lock=0.;last_id=None
    try:
        while True:
            state=_get(a.gateway,'state')
            if state['finished'] or state['edit_used']:summary['decision']='episode_ended_before_dispatch';break
            if not state['started'] or time.monotonic()<next_lock:time.sleep(.1);continue
            history=collector.snapshot()
            if not history:time.sleep(.1);continue
            latest=history[-1]['id'];slot=bundle['operation_model']['fixed_compute_dispatch_seconds']
            if state['elapsed_seconds']+slot>=catalog['timing']['robot_after_takeoff_budget_seconds']:summary['decision']='public_dispatch_deadline';break
            if a.policy!='belief_blind' and latest==last_id:time.sleep(.1);continue
            last_id=latest;locked=time.monotonic()
            # Public arrival timestamps align the training image-origin clock;
            # no ROS time, private start time, backend pose or fake timestamp.
            elapsed=max(0.,time.time()-history[0]['arrival_wall'])
            selected=predictor.decide(history,elapsed,catalog['timing']['robot_after_takeoff_budget_seconds']);compute=time.monotonic()-locked
            append_json(root/'decisions.jsonl',dict(wall=time.time(),public_elapsed=state['elapsed_seconds'],image_origin_elapsed=elapsed,observed_frames=len(history),actual_rgb_frames=sum(f.get('rgb_orientation_features') is not None for f in history),compute_seconds=compute,policy=selected))
            summary.update(decision=selected['kind'],computed_policy=selected,actual_compute_seconds=compute);write_json(root/'result.json',summary)
            if compute>slot:summary['decision']='fixed_compute_slot_missed';break
            if selected['kind']=='abstain':break
            while time.monotonic()<locked+slot:time.sleep(max(0.,min(.05,locked+slot-time.monotonic())))
            if selected['kind']=='wait':next_lock=locked+slot+3.;continue
            action=selected['ranked_candidates'][0]['action_id'];summary.update(decision='dispatching',action_id=action);write_json(root/'result.json',summary)
            request=urllib.request.Request(a.gateway+'/edit',data=json.dumps(dict(action_id=action)).encode(),headers={'Content-Type':'application/json'},method='POST')
            try:
                with urllib.request.urlopen(request,timeout=8) as response:summary.update(decision='edit',acknowledgment=json.load(response))
            except Exception as error:summary.update(decision='edit_failed_or_late',error_type=type(error).__name__)
            break
    finally:
        collector.stop.set();collector.thread.join(timeout=2.)
        summary.update(public_frame_requests=collector.requests,public_collection_errors=collector.errors,public_rgb_processing_seconds=collector.processing_seconds)
        write_json(root/'result.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=tuple(MODES)+('redvla_external',),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timing-model',type=Path,required=True);p.add_argument('--public-perception',type=Path,required=True);p.add_argument('--known-task-file',type=Path);p.add_argument('--effort');p.add_argument('--call-budget',type=int,default=1);a=p.parse_args()
    try:run(a)
    except (Exception,KeyboardInterrupt) as error:
        path=a.output/'result.json';summary=json.loads(path.read_text()) if path.exists() else {}
        summary.update(decision='observer_error' if not isinstance(error,KeyboardInterrupt) else 'interrupted_at_episode_end',error_type=type(error).__name__);write_json(path,summary)
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
