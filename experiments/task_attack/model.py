"""Use the project's authorized Responses transport; never substitute a model."""
import hashlib
import json
import time
from pathlib import Path

from . import MODEL, EFFORT, ATTACK_EFFORT, ATTACK_EFFORTS


class ModelDeadlineExceeded(RuntimeError):
    pass


class ModelClient:
    def __init__(self, log=None, timeout=120, role='robot', effort=None, *, credential_store=None):
        if role not in ('robot', 'observer'):raise ValueError('Unknown model role')
        self.role = role
        self.effort = effort or (EFFORT if role == 'robot' else ATTACK_EFFORT)
        if self.effort not in ATTACK_EFFORTS:raise ValueError('Unsupported pilot reasoning effort')
        if role == 'robot' and self.effort != EFFORT:raise ValueError('Task robot remains Terra Max')
        self.log = Path(log) if log else None
        self.timeout = timeout
        self.calls = 0
        self.wall_seconds = 0.
        self.deadline_monotonic = None
        self.credential_store = credential_store

    def request_timeout(self):
        if self.deadline_monotonic is None:return self.timeout
        remaining=self.deadline_monotonic-time.monotonic()
        if remaining<=0:raise ModelDeadlineExceeded('Experimental episode budget exhausted before inference')
        return min(self.timeout,remaining)

    def request(self, instructions, data, images=(), attempt_limit=2):
        from uav_bridge.responses_transport import TransportError
        for attempt in range(attempt_limit):
            self.request_timeout()
            try:
                return self._request_once(instructions,data,images)
            except TransportError as error:
                retryable=(error.code in ('stream_incomplete','network_error','timeout') or
                           (error.code=='http_error' and error.status in (500,502,503,504)))
                if not retryable or attempt+1>=attempt_limit:raise
                # Only repeat the exact read-only inference. No action exists
                # until a complete validated response has actually returned.
                time.sleep(.5)
        raise ValueError('A positive request attempt budget is required')

    def _request_once(self, instructions, data, images=()):
        from uav_bridge.chatgpt_auth import CredentialStore
        from uav_bridge.responses_transport import request_labeled_image_response, request_text_response
        store = self.credential_store if self.credential_store is not None else CredentialStore()
        if not store.is_model_available(MODEL):
            raise RuntimeError('Requested model is not in project account catalog: '+MODEL)
        token = store.get_access_token()
        timeout=self.request_timeout()
        text = json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        start = time.monotonic()
        record = dict(model=MODEL,role=self.role,reasoning_effort=self.effort,started_at=time.time(),
                      input=data,instructions_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
                      image_sha256=[hashlib.sha256(x['data']).hexdigest() for x in images])
        record['effective_timeout_seconds']=timeout
        self.calls += 1
        try:
            args = dict(model=MODEL,instructions=instructions,text=text,access_token=token,
                        timeout=timeout,reasoning_effort=self.effort)
            if images:
                result = request_labeled_image_response(image_bytes=images[-1]['data'],
                                          context_images=list(images[:-1]),
                                          current_image_label=images[-1]['label'],**args)
            else:
                result = request_text_response(**args)
            record['output'] = result
            return result
        except Exception as e:
            record['error'] = dict(type=type(e).__name__,code=getattr(e,'code',None),
                                   status=getattr(e,'status',None))
            raise
        except KeyboardInterrupt:
            record['error'] = dict(type='KeyboardInterrupt',code='inference_cancelled_at_episode_end')
            raise
        finally:
            record['wall_seconds'] = time.monotonic()-start
            self.wall_seconds += record['wall_seconds']
            if self.log:
                self.log.parent.mkdir(parents=True,exist_ok=True)
                with self.log.open('a') as f:
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
