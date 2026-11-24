"""Use the project's authorized Responses transport; never substitute a model."""
import hashlib
import json
import time
from pathlib import Path

from . import MODEL, EFFORT, ATTACK_EFFORT, ATTACK_EFFORTS


EXTERNAL_RESPONSE_CODES = {"subscription_sharing_usage_limit_exceeded", "rate_limit_exceeded",
                           "insufficient_quota", "authentication_error", "invalid_token", "invalid_api_key",
                           "not_signed_in", "refresh_interrupted", "plan_usage_disabled", "invalid_storage",
                           "refresh_not_ready", "refresh_expired", "invalid_grant", "invalid_refresh_token",
                           "token_expired", "refresh_token_expired", "refresh_token_invalidated",
                           "refresh_token_reused", "model_not_in_catalog"}


class ModelDeadlineExceeded(RuntimeError):
    pass


class ModelAttemptLimitExceeded(RuntimeError):
    pass


class ModelClient:
    def __init__(self, log=None, timeout=120, role='robot', effort=None, *, credential_store=None, request_budget=None):
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
        self.request_budget = request_budget
        self.external_error = None

    def request_timeout(self):
        if self.deadline_monotonic is None:return self.timeout
        remaining=self.deadline_monotonic-time.monotonic()
        if remaining<=0:raise ModelDeadlineExceeded('Experimental episode budget exhausted before inference')
        return min(self.timeout,remaining)

    def request(self, instructions, data, images=(), attempt_limit=2):
        from uav_bridge.responses_transport import TransportError
        if self.request_budget is not None:
            attempt_limit=min(attempt_limit,2)
            if self.external_error is not None:
                raise self.external_error
        for attempt in range(attempt_limit):
            self.request_timeout()
            if self.request_budget is not None:
                self.request_budget.check()
            try:
                return self._request_once(instructions,data,images)
            except TransportError as error:
                if self.request_budget is not None and (error.code in EXTERNAL_RESPONSE_CODES or
                                                        error.status in (401,403,429)):
                    self.external_error = error
                    raise
                retryable=(error.code in ('stream_incomplete','network_error','timeout') or
                           (error.code=='http_error' and error.status in (500,502,503,504)))
                if not retryable or attempt+1>=attempt_limit:raise
                # Only repeat the exact read-only inference. No action exists
                # until a complete validated response has actually returned.
                delay=getattr(error,'retry_after',None)
                if not isinstance(delay,(int,float)) or not 0<=delay:
                    delay=.5
                remaining=self.request_timeout()
                if delay>=remaining:raise
                time.sleep(delay)
        raise ValueError('A positive request attempt budget is required')

    def _request_once(self, instructions, data, images=()):
        from uav_bridge.chatgpt_auth import AuthError, CredentialStore
        from uav_bridge.responses_transport import request_labeled_image_response, request_text_response
        store = self.credential_store if self.credential_store is not None else CredentialStore()
        try:
            if not store.is_model_available(MODEL):
                raise AuthError('model_not_in_catalog', 'Requested model is not in project account catalog: '+MODEL)
            token = store.get_access_token()
        except AuthError as error:
            if self.request_budget is not None and (error.code in EXTERNAL_RESPONSE_CODES or
                                                    error.status in (401,403,429)):
                self.external_error = error
            raise
        timeout=self.request_timeout()
        text = json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        start = time.monotonic()
        record = dict(model=MODEL,role=self.role,reasoning_effort=self.effort,started_at=time.time(),
                      input=data,instructions_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
                      image_sha256=[hashlib.sha256(x['data']).hexdigest() for x in images])
        record['effective_timeout_seconds']=timeout
        def before_dispatch():
            self.request_timeout()
            record['logical_job_attempt']=self.request_budget.reserve(
                model=MODEL,effort=self.effort,image_sha256=record['image_sha256'])
            record['dispatched_monotonic_seconds']=time.monotonic()
            self.calls += 1

        try:
            args = dict(model=MODEL,instructions=instructions,text=text,access_token=token,
                        timeout=timeout,reasoning_effort=self.effort)
            if self.request_budget is not None:
                args['before_dispatch'] = before_dispatch
            else:
                self.calls += 1
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
            if 'dispatched_monotonic_seconds' in record:
                record['return_monotonic_seconds']=time.monotonic()
            self.wall_seconds += record['wall_seconds']
            if self.log:
                self.log.parent.mkdir(parents=True,exist_ok=True)
                with self.log.open('a') as f:
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
