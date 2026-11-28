"""A single actual image services only the explicit, currently released visual questions."""
from hashlib import sha256

from .room_inspection import PROMPT,OUTCOMES


def request(client,frame_id,data,questions):
    if not questions or len(questions)>2 or len(set(k for k,_ in questions))!=len(questions):
        raise ValueError("Expected one or two distinct, explicitly permitted questions")
    h=sha256(data).hexdigest()
    inp=dict(image=frame_id,sha256=h,questions=[dict(id=k,image=frame_id,category=cat) for k,cat in questions])
    result=client.request(PROMPT,inp,[dict(label=frame_id+" camera RGB",data=data)],attempt_limit=1)
    if (not isinstance(result,dict) or set(result)!={k for k,_ in questions} or
            any(not isinstance(v,str) or v not in OUTCOMES for v in result.values())):
        raise ValueError("Actual model response did not bind every requested field")
    return result
