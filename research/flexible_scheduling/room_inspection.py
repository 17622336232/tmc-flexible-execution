"""Closed-loop, public validation-house missions with scoped early capture and reading."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import monotonic,sleep
from statistics import median
import csv

from experiments.flexible_execution.public_task import Room,TURNS,MIN_OBJECT_PIXELS,_same_pose
from experiments.flexible_execution.public_visual import _house,VALIDATION_SOURCE,open_controller
from experiments.flexible_execution.public_audit import save_segmentation,_grade_segmentation
from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.timeline import Timeline,read_trace_rows
from research.flexible_scheduling.workflow import View,Workflow,DecisionProblem


PROMPT="""Use the provided actual robot camera images. For each question id, inspect its labelled
image to tell whether any instance of the requested class is visibly present. Return a JSON mapping
every question id to positive, negative, or uncertain. Negative means no identifiable instance
in that field of view. Each answer refers only to its own image. Follow the supplied instruction.
Do not infer facts about an unseen angle. Emit only the requested JSON object."""
OUTCOMES={"positive","negative","uncertain"}


def instruction(case):
    route="右" if case["direction"]=="right" else "左"
    notes=["任务开始先向"+route+"转90度保存一张未判读的侧面照片，再回到起始朝向。" ] if case["mandatory_initial_side_archive"] else []
    for front,side in zip(case["front"],case["side"]):
        notes.append(f"报告正面是否可见 {front}。如果可见，还要报告{route}侧照片中是否可见 {side}。")
    notes.append("报告中附所用原图的编号与指纹，结束时恢复原朝向。")
    notes.append("允许在正面判断前或等待期间取得侧面照片。" if case["early_capture"] else "必须等正面条件成立才可取得侧面照片。")
    notes.append("允许提前判读侧面照片，只提交条件成立的侧面项目。" if case["early_semantic"] else "必须等正面条件成立才可判读对应侧面内容。")
    return "".join(notes)


def model_plan(case,method,config):
    import math
    if len(case["front"])!=len(case["side"]) or not case["side"]:
        raise ValueError("Each front predicate must have a corresponding conditional side field")
    assay=Path("outputs/continuing/paired-rgb-assay01")
    score=read(assay/"summary.json")
    if not (score["timed_valid_pairs"]==6 and score["median_separate_minus_joint_service_s"]>0
            and score["joint_both_correct"]>=score["separate_both_correct"]-1):
        raise ValueError("Prospective joint-reading development gate not met")
    pairs=list(csv.DictReader((assay/"paired.csv").open()))
    front,joint,side=(median(float(p[k]) for p in pairs) for k in ("front_s","joint_s","side_s"))
    frozen=read(Path("outputs/protocol-v2-optimization/public-final/frozen_evaluation.json"))
    costs=frozen["runtime"]["cost_calibration"]["motion"]
    p=config["per_field_prior"]
    factors=tuple(((0,1-p),(1<<i,p)) for i in range(len(case["front"])))
    view=View("side",(1<<len(factors))-1,2*costs["turn"]+costs["photo"],case["early_capture"],case["early_semantic"],
              case["mandatory_initial_side_archive"])
    task=Workflow(factors,(view,),0,(front,joint),(0.,side),0.)
    solver="prefetch" if method=="physical-prefetch" else method
    selected=DecisionProblem(task).solve(solver)
    return {**selected,"cost_estimates":{"one_front":front,"one_side":side,"joint":joint,"side_roundtrip":view.trip},
            "assay_sha256":digest(assay/"summary.json")}


def query(client,trace,index,case,frames,questions):
    images=[dict(label=frame.id+" camera RGB",data=frame.payload) for frame in frames]
    input_data=dict(mission=instruction(case),questions=[dict(id=name,image=frame,requested_class=target)
                                                          for name,frame,target in questions])
    with trace.measure("model_request",index,questions=[x[0] for x in questions],image_sha256=[f.digest for f in frames]):
        result=client.request(PROMPT,input_data,images,attempt_limit=1)
    if not isinstance(result,dict) or set(result)!={q[0] for q in questions} or not all(
            isinstance(value,str) and value in OUTCOMES for value in result.values()):
        raise ValueError("Invalid perception result")
    return result


def acquire(room,case,why):
    turn,restore,degrees=TURNS[case["direction"]]
    if "side" in room.events:
        raise RuntimeError("This static side view was already acquired")
    room.trace.event("side_collection",why)
    evt=room.turn(turn,degrees)
    shot=room.take("side",evt)
    room.turn(restore,0)
    return shot


def execute(room,case,method,config,client,smoke=False):
    # Plans can access the public mission and existing observations only.
    front=room.take("front",room.controller.last_event)
    side=acquire(room,case,"mandatory-initial-archive") if case["mandatory_initial_side_archive"] else None
    if smoke:
        if side is None:
            side=acquire(room,case,"instrument-only")
        return dict(smoke=True,frames={"front":front.digest,"side":side.digest})
    selection=model_plan(case,method,config)
    save(room.output/"policy_decision.json",selection)
    f_questions=[(f"front_{i}","front",c) for i,c in enumerate(case["front"])]
    s_questions=[(f"side_{i}","side",c) for i,c in enumerate(case["side"])]
    if selection["joint"]:
        if not case["early_semantic"]:
            raise RuntimeError("Early semantic interpretation prohibited")
        if side is None:
            if not case["early_capture"]:
                raise RuntimeError("Early photo acquisition prohibited")
            side=acquire(room,case,"before-joint-read")
        results=query(client,room.trace,"joint",case,[front,side],f_questions+s_questions)
    elif selection["overlap"]:
        if not case["early_capture"] or side:
            raise RuntimeError("Invalid pending acquisition")
        dispatch_before=client.calls
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(query,client,room.trace,"front",case,[front],f_questions)
            while not pending.done() and client.calls==dispatch_before and monotonic()<client.deadline_monotonic:
                sleep(.01)
            if not pending.done() and client.calls>dispatch_before:
                side=acquire(room,case,"during-actual-front-request")
            results=pending.result()
    else:
        results=query(client,room.trace,"front",case,[front],f_questions)
    required=[s_questions[i] for i in range(len(f_questions)) if results[f_questions[i][0]]=="positive"]
    if required and not selection["joint"]:
        if side is None:
            side=acquire(room,case,"after-positive")
        results.update(query(client,room.trace,"side",case,[side],required))
    retained=f_questions+required
    hashes={"front":front.digest,"side":side.digest if side else None}
    return {name:dict(outcome=results[name],image=frame,sha256=hashes[frame]) for name,frame,_ in retained}


def independent_grade(case,root,report,restored,smoke=False):
    # Evaluate only after execution has handed back its submitted report.
    truth={}
    for f in ("front","side"):
        for i,target in enumerate(case[f]):
            key=f"{f}_{i}"
            if (root/"scoring"/(key+".json")).is_file():
                truth[key]=_grade_segmentation(root/"scoring",key)
    required=[f"front_{i}" for i in range(len(case["front"]))]
    required += [f"side_{i}" for i in range(len(case["side"])) if truth.get(f"front_{i}")=="positive"]
    if smoke:
        return dict(role="instrument-only",restored=restored,segmented_fields=len(truth),no_task_score=True)
    expected={}
    for key in required:
        side=key.split("_")[0]
        path=root/(side+".jpg")
        expected[key]=dict(outcome=truth.get(key),image=side,sha256=digest(path) if path.is_file() else None)
    exact=report==expected and all(v["outcome"] is not None and v["sha256"] for v in expected.values())
    return dict(strict=bool(restored and exact),restored=restored,expected=expected,reported=report,
                segmentation_pixel_threshold=MIN_OBJECT_PIXELS)


def run(config,case,method,root,display,smoke=False):
    from experiments.task_attack.model import ModelClient,EXTERNAL_RESPONSE_CODES
    from uav_bridge.chatgpt_auth import CredentialStore
    catalog=read(Path(config["runtime_catalog"]))
    run=catalog["runtime"]
    if case["house"] not in {r["official_house_index"] for r in catalog["cases"]}:
        raise ValueError("Development is restricted to previously exposed validation houses")
    root.mkdir(parents=True,exist_ok=False)
    started=monotonic()
    house,world_hash=_house(Path(run["archive"]),case["house"],source=VALIDATION_SOURCE)
    save(root/"source.json",dict(case=case["id"],validation_house=case["house"],world_hash=world_hash,
             method=method,mission=instruction(case),smoke=smoke,model=config["model"],effort=config["effort"]))
    trace=Timeline(root/"timeline.jsonl")
    budget=RequestBudget(root/"requests.jsonl",config["request_limit"])
    client=None if smoke else ModelClient(root/"model_calls.jsonl",effort=config["effort"],
         credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),request_budget=budget)
    if client:
        client.deadline_monotonic=started+config["deadline_s"]
    controller=None
    report=None
    failure=None
    external=False
    restored=False
    room=None
    try:
        controller=open_controller(house,Path(run["build_dir"]),"opengl",display,segmentation=True)
        room=Room(controller,root,trace,started)
        report=execute(room,case,method,config,client,smoke)
        save(root/"report.json",report)
        restored=_same_pose(room.initial_pose,controller.last_event.metadata["agent"])
    except Exception as exc:
        code=getattr(exc,"code",type(exc).__name__)
        failure=dict(code=str(code),detail=str(exc)[:300])
        external=code in EXTERNAL_RESPONSE_CODES or getattr(exc,"status",None) in (401,403,429)
    finished=monotonic()
    try:
        if room:
            for frame,event in room.events.items():
                for i,target in enumerate(case[frame]):
                    save_segmentation(root,f"{frame}_{i}",event,target,MIN_OBJECT_PIXELS)
        grade=independent_grade(case,root,report,restored,smoke)
        if not smoke and finished-started>config["deadline_s"]:
            grade["strict"]=False
        result=dict(**grade,seconds=finished-started,deadline=config["deadline_s"],method=method,case=case["id"],
                    failure=failure,endpoint_block=external,sends=budget.summary()["reserved"])
        save(root/"result.json",result)
        return result
    finally:
        if controller:
            controller.stop()
