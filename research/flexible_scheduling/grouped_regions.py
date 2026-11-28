"""Answer multiple explicitly authorized region predicates on one frozen robot photograph."""
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite


PROMPT="""Inspect the supplied onboard RGB. For each listed region, answer that region's visual
question using only pixels of the specified subject in this frame. Boxes use normalized left, top,
right, bottom coordinates. Return one JSON object with an answers array. Each answer has exactly
id and outcome; keep the input region id. Allowed outcomes: positive, negative, uncertain. A
missing, cropped, ambiguous, or unreadable subject is uncertain. Do not copy a neighbor's answer.
Return no task instructions, navigation, extra regions, or other information."""
OUTCOMES={"positive","negative","uncertain"}


@dataclass(frozen=True)
class Region:
    id: str
    bbox: tuple
    question: str

    def __post_init__(self):
        b=tuple(self.bbox)
        if (not isinstance(self.id,str) or not self.id or not isinstance(self.question,str) or
                not self.question.strip() or len(b)!=4 or not all(type(x) in (int,float) and isfinite(x) for x in b)
                or not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1)):
            raise ValueError("Expected an identified normalized region and a visible question")
        object.__setattr__(self,"bbox",b)

    def public(self):
        return dict(id=self.id,bbox=list(self.bbox),question=self.question)


def request_binding(frame_id,photo_sha256,regions):
    if (not isinstance(frame_id,str) or not frame_id or not isinstance(photo_sha256,str) or
            len(photo_sha256)!=64 or any(c not in "0123456789abcdef" for c in photo_sha256)):
        raise ValueError("An immutable reference image id and digest are required")
    if not regions or len({r.id for r in regions})!=len(regions) or len({(r.bbox,r.question) for r in regions})!=len(regions):
        raise ValueError("Region ids and region-question pairs must be unique")
    return dict(frame_id=frame_id,photo_sha256=photo_sha256,regions=[r.public() for r in regions])


def answers(binding,image_sha256,reply):
    if (image_sha256!=[binding["photo_sha256"]] or not isinstance(reply,dict) or
            set(reply)!={"answers"} or not isinstance(reply["answers"],list)):
        return None
    permitted={r["id"] for r in binding["regions"]}
    observed={}
    for row in reply["answers"]:
        if (not isinstance(row,dict) or set(row)!={"id","outcome"} or not isinstance(row["id"],str)
                or row["id"] not in permitted or row["id"] in observed or not isinstance(row["outcome"],str)
                or row["outcome"] not in OUTCOMES):
            return None
        observed[row["id"]]=row["outcome"]
    return observed


def dispatch(client,frame_id,photo_sha256,rgb,regions):
    """One actual transport with these pixels. Singleton and grouped use the same API and prompt."""
    if sha256(rgb).hexdigest()!=photo_sha256:
        raise ValueError("Frozen camera output changed before transmission")
    binding=request_binding(frame_id,photo_sha256,regions)
    reply=client.request(PROMPT,binding,[dict(label="Frozen onboard RGB "+frame_id,data=rgb)],attempt_limit=1)
    return binding,reply
