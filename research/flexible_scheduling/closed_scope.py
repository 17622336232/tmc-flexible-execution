"""A terminal certificate for user-scoped static regions. Open-ended rows do not use this path."""
from hashlib import sha256
import json
from math import isfinite
import re

from experiments.flexible_execution.evidence import Frame,Observation,accept
from experiments.flexible_execution.task_ir import Requirement


BODY_ID=re.compile(r"[A-Z]{1,6}-[0-9]{2,6}\Z")
QUESTION="该设备的正面外壳是否存在裂纹？"
FRONT_CRACK=re.compile(r"(?:图像中)?(?:该|这台)设备的正面外壳是否存在裂纹[？?]")


def is_front_crack_question(text):
    return isinstance(text,str) and FRONT_CRACK.fullmatch(text) is not None


class ClosedInspection:
    def __init__(self,mission_id,reference,regions):
        if not mission_id or not regions or not isinstance(reference,Frame):
            raise ValueError("An explicit mission roster and measured reference frame are required")
        self.mission=mission_id
        self.reference=reference
        self.regions={}
        for label,coordinates in regions:
            box=tuple(coordinates)
            if (not isinstance(label,str) or not label or label in self.regions or len(box)!=4 or
                not all(type(v) in (float,int) and isfinite(v) for v in box) or
                not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1) or box in self.regions.values()):
                raise ValueError("Invalid or duplicate public region")
            self.regions[label]=box
        encoded=json.dumps([mission_id,reference.id,reference.digest,sorted(self.regions.items())],separators=(",",":"))
        self.scope_hash=sha256(encoded.encode()).hexdigest()
        self.conditions={}
        self.materials={}
        self.conflicts=set()

    def record_judgment(self,region,payload,image_hashes,reply):
        if region not in self.regions or not isinstance(payload,dict) or not isinstance(reply,dict):
            return False
        if (image_hashes!=[self.reference.digest] or payload.get("frame_id")!=self.reference.id or
                payload.get("target_id")!=region or not is_front_crack_question(payload.get("question")) or
                payload.get("bbox_in_attached_photo")!=list(self.regions[region]) or
                reply.get("outcome") not in ("positive","negative","uncertain")):
            return False
        value=reply["outcome"]
        previous=self.conditions.get(region)
        if previous is not None and previous!=value:
            self.conflicts.add(region)
        self.conditions[region]=value
        return True

    def needs_identity(self,region):
        return region not in self.conflicts and self.conditions.get(region)=="positive"

    def record_group(self,binding,image_hashes,reply):
        from .grouped_regions import answers
        if (not isinstance(binding,dict) or binding.get("frame_id")!=self.reference.id
                or binding.get("photo_sha256")!=self.reference.digest or not isinstance(binding.get("regions"),list)):
            return False
        ids=set()
        targets={}
        for row in binding["regions"]:
            if (not isinstance(row,dict) or not isinstance(row.get("id"),str) or row["id"] not in self.regions
                    or row["id"] in ids or row.get("bbox")!=list(self.regions[row["id"]])
                    or not is_front_crack_question(row.get("question"))):
                return False
            ids.add(row["id"])
            targets[row["id"]]=row
        if not ids:
            return False
        values=answers(binding,image_hashes,reply)
        if values is None:
            return False
        for label,val in values.items():
            row=targets[label]
            self.record_judgment(label,dict(frame_id=self.reference.id,target_id=label,question=row["question"],
                                 bbox_in_attached_photo=row["bbox"]),image_hashes,{"outcome":val})
        return True

    def record_identity(self,region,frame,hint,at):
        if not self.needs_identity(region) or not isinstance(hint,dict) or not isinstance(frame,Frame):
            return False
        if (frame.object_id!=region or frame.digest==self.reference.digest or
                hint.get("raw_image_sha256")!=frame.digest or hint.get("reference_sha256")!=self.reference.digest or
                hint.get("scope_hash")!=self.scope_hash or hint.get("reference_bbox")!=list(self.regions[region]) or
                hint.get("field")!="identity" or not isinstance(hint.get("value"),str) or not BODY_ID.fullmatch(hint["value"])):
            return False
        requirement=Requirement(region+"/identity",region,"identity")
        bound=accept(requirement,Observation(frame,"identity",hint["value"],True),at)
        if bound is None:
            return False
        if region in self.materials and self.materials[region].value!=bound.value:
            self.conflicts.add(region)
            return False
        self.materials[region]=bound
        return True

    def assemble(self,*,at_anchor):
        missing=[]
        findings=[]
        covered=[]
        for region,box in self.regions.items():
            condition=self.conditions.get(region)
            label=self.materials.get(region)
            if region in self.conflicts or condition not in ("positive","negative"):
                missing.append((region,"resolved-crack-judgment"))
                continue
            covered.append(dict(region=region,bbox=list(box),cracked=condition=="positive",image_sha256=self.reference.digest))
            if condition=="positive":
                if label is None:
                    missing.append((region,"source-bound-identity"))
                else:
                    findings.append(dict(region=region,asset_id=label.value,photo=label.frame_id,
                                         image_sha256=label.frame_hash))
        if len({r["asset_id"] for r in findings})!=len(findings):
            missing.append(("mission","distinct-body-identities"))
        if at_anchor is not True:
            missing.append(("mission","verified-return"))
        if missing:
            return None,missing
        return dict(mission=self.mission,scope_sha256=self.scope_hash,coverage=covered,findings=findings),[]
