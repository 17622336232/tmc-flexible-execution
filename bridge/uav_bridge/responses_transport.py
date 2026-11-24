"""Fixed-endpoint, completion-gated Responses SSE transport for ChatGPT plans.

Official contract checked against OpenAI's token-sharing models-and-inference,
preview-limitations, and errors-and-recovery documentation. This module does
not acquire credentials, retry requests, execute tools, or change billing paths.
"""

import base64
import http.client
import json
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.request

from .planner import PlanError, _NoRedirect, _strict_json


_ENDPOINT = "https://api.openai.com/v1/responses"
_MAX_IMAGE = 4 * 1024 * 1024
_MAX_LINE = 1024 * 1024
_MAX_EVENT = 1024 * 1024
_MAX_STREAM = 4 * 1024 * 1024
_MAX_ERROR = 64 * 1024
_MAX_EVENTS = 10000
_REMOTE_ERROR_MESSAGES = {
    "subscription_sharing_usage_limit_exceeded": (
        "本次 ChatGPT 计划调用达到使用限制，请到 ChatGPT 设置 → Usage 检查账户及本应用限额；"
        "不代表全账户积分用尽。"
    ),
}
_REASONING_EVENTS = {
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
    "response.reasoning_text.delta", "response.reasoning_text.done",
}
_ALLOWED_EVENTS = {
    "response.created", "response.queued", "response.in_progress",
    "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done", "response.completed", "response.metadata",
} | _REASONING_EVENTS


class TransportError(ValueError):
    """Sanitized transport diagnostics; raw bodies and credentials are excluded."""

    def __init__(self, code, message, status=None, request_id=None, param=None, body_shape=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.request_id = request_id
        self.param = param
        self.body_shape = body_shape


def _identifier(value, token, maximum=200):
    if (not isinstance(value, str) or not 1 <= len(value) <= maximum or
            not re.fullmatch(r"[A-Za-z0-9_.:\[\]/-]+", value) or
            token in value or "authorization" in value.lower() or "bearer" in value.lower()):
        return None
    return value


def _request_id(headers, token):
    if headers is None:
        return None
    for field in ("x-request-id", "request-id", "X-Request-Id"):
        value = _identifier(headers.get(field), token)
        if value:
            return value
    return None


def _shape(value):
    # Describe structure without copying arbitrary body keys or diagnostic text.
    if isinstance(value, dict):
        if isinstance(value.get("error"), dict):
            return "object(error_object)"
        if "detail" in value:
            return "object(detail)"
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _json(raw, status=None, request_id=None):
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = _strict_json(raw)
        def finite(item, depth=0):
            if depth > 64:
                raise ValueError("depth")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("nonfinite")
            if isinstance(item, str):
                item.encode("utf-8")
            elif isinstance(item, dict):
                for key, nested in item.items():
                    key.encode("utf-8")
                    finite(nested, depth + 1)
            elif isinstance(item, list):
                for nested in item:
                    finite(nested, depth + 1)
        finite(value)
        return value
    except (PlanError, ValueError, UnicodeError, TypeError, RecursionError, OverflowError):
        raise TransportError("invalid_json", "Responses 返回了无效或不严格的 JSON。", status, request_id) from None


def _remote_error(payload, fallback, message, status, request_id, token, body_shape=None):
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        error = payload if isinstance(payload, dict) else {}
    code = _identifier(error.get("code"), token) or fallback
    param = _identifier(error.get("param"), token)
    identifier = request_id or _identifier(error.get("request_id"), token)
    if not identifier and isinstance(payload, dict):
        identifier = _identifier(payload.get("request_id"), token)
    return TransportError(code, _REMOTE_ERROR_MESSAGES.get(code, message), status, identifier,
                          param, body_shape or _shape(payload))


def _admission_error(error, token):
    status = error.code if isinstance(error.code, int) else None
    identifier = _request_id(error.headers, token)
    try:
        raw = error.read(_MAX_ERROR + 1)
        if len(raw) > _MAX_ERROR:
            return TransportError("http_error", "Responses 请求未获准；错误正文超过诊断大小限制。", status, identifier,
                                  body_shape="oversized")
        if not raw:
            return TransportError("http_error", "Responses 请求未获准。", status, identifier, body_shape="empty")
        try:
            payload = _json(raw, status, identifier)
        except TransportError:
            return TransportError("http_error", "Responses 请求未获准；服务端返回非 JSON 诊断。", status, identifier,
                                  body_shape="non_json")
        return _remote_error(payload, "http_error", "Responses 请求未获准，请按错误代码检查权限或请求配置。",
                             status, identifier, token)
    except (OSError, http.client.HTTPException, ValueError):
        return TransportError("http_error", "Responses 请求未获准且诊断正文不可读。", status, identifier,
                              body_shape="unreadable")
    finally:
        error.close()


def _reasoning_part(part, expected, status, request_id):
    if (not isinstance(part, dict) or set(part) != {"type", "text"} or
            part.get("type") != expected or not isinstance(part.get("text"), str)):
        raise TransportError("unsupported_output", "Responses 推理元数据包含未知字段、动作或无效内容。", status, request_id)


def _reasoning_item(item, status, request_id):
    # Official ResponseReasoningItem fields only. Text/encrypted metadata are
    # intentionally ignored, never interpreted as the model's final answer.
    allowed = {"type", "id", "summary", "content", "encrypted_content", "status"}
    if (not isinstance(item, dict) or item.get("type") != "reasoning" or set(item) - allowed or
            not isinstance(item.get("id"), str) or not item["id"] or
            not isinstance(item.get("summary"), list)):
        raise TransportError("unsupported_output", "Responses 推理项结构无效或包含动作字段。", status, request_id)
    for part in item["summary"]:
        _reasoning_part(part, "summary_text", status, request_id)
    if "content" in item:
        if not isinstance(item["content"], list):
            raise TransportError("unsupported_output", "Responses 推理内容结构无效。", status, request_id)
        for part in item["content"]:
            _reasoning_part(part, "reasoning_text", status, request_id)
    if "encrypted_content" in item and item["encrypted_content"] is not None and not isinstance(item["encrypted_content"], str):
        raise TransportError("unsupported_output", "Responses 加密推理元数据结构无效。", status, request_id)
    if "status" in item and item["status"] not in ("in_progress", "completed", "incomplete"):
        raise TransportError("unsupported_output", "Responses 推理项状态无效。", status, request_id)


def _reasoning_event(event, status, request_id):
    kind = event["type"]
    summary = kind.startswith("response.reasoning_summary_")
    index = "summary_index" if summary else "content_index"
    allowed = {"type", "item_id", "output_index", "sequence_number", index}
    if "_part." in kind:
        allowed.add("part")
        if kind.endswith(".done"):
            allowed.add("status")
        if set(event) - allowed:
            raise TransportError("unsupported_output", "Responses 推理事件包含未知字段或动作。", status, request_id)
        _reasoning_part(event.get("part"), "summary_text", status, request_id)
        if "status" in event and event["status"] != "incomplete":
            raise TransportError("unsupported_output", "Responses 推理摘要状态无效。", status, request_id)
    else:
        field = "delta" if kind.endswith(".delta") else "text"
        allowed.add(field)
        # Official reasoning deltas can carry random streaming padding. It is
        # never response text, and the allowance does not extend to done events.
        if kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            allowed.add("obfuscation")
        if (set(event) - allowed or not isinstance(event.get(field), str) or
                ("obfuscation" in event and not isinstance(event["obfuscation"], str))):
            raise TransportError("unsupported_output", "Responses 推理事件包含未知字段、动作或无效文字。", status, request_id)
    if "item_id" in event and (not isinstance(event["item_id"], str) or not event["item_id"]):
        raise TransportError("unsupported_output", "Responses 推理事件标识无效。", status, request_id)
    for field in ("output_index", "sequence_number", index):
        if field in event and (isinstance(event[field], bool) or not isinstance(event[field], int) or event[field] < 0):
            raise TransportError("unsupported_output", "Responses 推理事件索引无效。", status, request_id)


def _text_only_item(item, status, request_id, require_complete=False):
    if (not isinstance(item, dict) or item.get("type") != "message" or item.get("role") != "assistant" or
            item.get("tool_calls") or item.get("function_call") or item.get("refusal")):
        raise TransportError("unsupported_output", "Responses 包含非助手文本输出、工具或拒绝信息，已拒绝结果。",
                             status, request_id)
    if require_complete and item.get("status") not in (None, "completed"):
        raise TransportError("incomplete_output", "Responses 助手消息尚未完成。", status, request_id)
    content = item.get("content")
    if not isinstance(content, list) or (require_complete and not content):
        raise TransportError("invalid_output", "Responses 助手消息内容结构无效。", status, request_id)
    parts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
            raise TransportError("unsupported_output", "Responses 包含拒绝信息或非文本内容，已拒绝结果。", status, request_id)
        parts.append(part["text"])
    return "".join(parts)



def _metadata_event(event, status, request_id):
    """Accept only the observed, non-action ChatGPT moderation extension."""
    def reject():
        raise TransportError("unsupported_output", "Responses 元数据结构未知或包含非审核字段。", status, request_id)

    if set(event) - {"type", "sequence_number", "response_id", "metadata"}:
        reject()
    if "sequence_number" in event and (isinstance(event["sequence_number"], bool) or
            not isinstance(event["sequence_number"], int) or event["sequence_number"] < 0):
        reject()
    if "response_id" in event and (not isinstance(event["response_id"], str) or
            not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", event["response_id"])):
        reject()
    metadata = event.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"openai_chatgpt_moderation_metadata"}:
        reject()
    stages = metadata["openai_chatgpt_moderation_metadata"]
    if not isinstance(stages, dict) or set(stages) - {"prompt", "generation", "tool_call", "tool_response"}:
        reject()
    for stage in stages.values():
        if not isinstance(stage, dict) or set(stage) - {"omnimod"}:
            reject()
        if not stage:
            continue
        moderation = stage["omnimod"]
        if (not isinstance(moderation, dict) or set(moderation) != {"protection_type", "outputs"} or
                not isinstance(moderation["protection_type"], str) or
                not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", moderation["protection_type"]) or
                not isinstance(moderation["outputs"], list)):
            reject()
        for output in moderation["outputs"]:
            if (not isinstance(output, dict) or set(output) != {"results", "is_blocked"} or
                    not isinstance(output["is_blocked"], bool) or not isinstance(output["results"], list)):
                reject()
            if output["is_blocked"]:
                raise TransportError("unsupported_output", "Responses 审核标明输出被阻止，未采纳结果。", status, request_id)
            for result in output["results"]:
                if not isinstance(result, dict) or set(result) != {"category_flags"}:
                    reject()
                flags = result["category_flags"]
                if (not isinstance(flags, dict) or any(not re.fullmatch(r"[A-Z0-9_+-]{1,80}", key) or
                        not isinstance(value, bool) for key, value in flags.items())):
                    reject()
                if flags.get("REFUSED"):
                    raise TransportError("unsupported_output", "Responses 审核标明拒绝输出，未采纳结果。", status, request_id)


class _TextConsistency:
    """Partial text is never actionable and must agree with completed output."""

    def __init__(self, status, request_id):
        self.parts = {}
        self.done = {}
        self.items = {}
        self.started_items = set()
        self.item_ids = {}
        self.item_types = {}
        self.response_ids = set()
        self.invalid_identity = False
        self.status, self.request_id = status, request_id

    def reject(self):
        raise TransportError("inconsistent_output", "Responses 增量文本与完成文本不一致。", self.status, self.request_id)

    def key(self, event):
        indexes = tuple(event.get(field, 0) for field in ("output_index", "content_index"))
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indexes):
            self.reject()
        return indexes

    def finish(self, key, text):
        if not isinstance(text, str):
            self.reject()
        if key in self.parts and "".join(self.parts[key]) != text:
            self.reject()
        if key in self.done and self.done[key] != text:
            self.reject()
        self.done[key] = text

    def observe_identity(self, event):
        response = event.get("response")
        response_id = response.get("id") if isinstance(response, dict) else event.get("response_id")
        if response_id is not None:
            if not isinstance(response_id, str) or not response_id:
                self.invalid_identity = True
            else:
                self.response_ids.add(response_id)
        if "item" not in event and "item_id" not in event:
            return
        index = self.key(event)[0]
        item = event.get("item")
        identity = item.get("id") if isinstance(item, dict) else event.get("item_id")
        if not isinstance(identity, str) or not identity:
            self.invalid_identity = True
        else:
            self.item_ids.setdefault(index, set()).add(identity)
        if isinstance(item, dict):
            self.item_types.setdefault(index, set()).add(item["type"])
            if event["type"] == "response.output_item.added":
                self.started_items.add(index)
            elif event["type"] == "response.output_item.done":
                if index in self.items and self.items[index] != item:
                    self.invalid_identity = True
                self.items[index] = item

    def completed_event(self, event):
        """Some ChatGPT streams finish with output=[] after full item.done events.

        Keep the successful terminal event as the completion gate, but assemble
        its missing aggregate from complete, identity-bound output items. Never
        repair a nonempty aggregate, a delta-only stream or a failed response.
        """
        response = event.get("response")
        if (not isinstance(response, dict) or response.get("status") != "completed" or
                response.get("error") is not None or response.get("incomplete_details") is not None or
                response.get("output") != [] or not self.items):
            return event
        identity = response.get("id")
        if (not isinstance(identity, str) or not identity or self.invalid_identity or
                self.response_ids - {identity}):
            self.reject()
        indexes = set(self.items)
        if (indexes != set(range(len(indexes))) or self.started_items - indexes or
                any(index not in indexes for index in self.item_ids)):
            raise TransportError("incomplete_output", "Responses 流式输出项未完整结束。",
                                 self.status, self.request_id)
        identities = set()
        for index, item in self.items.items():
            item_id = item.get("id")
            if (not isinstance(item_id, str) or not item_id or item_id in identities or
                    self.item_ids.get(index) != {item_id} or len(self.item_types.get(index, ())) != 1):
                self.reject()
            identities.add(item_id)
            if item["type"] == "message" and item.get("status") != "completed":
                raise TransportError("incomplete_output", "Responses 流式助手消息未完整结束。",
                                     self.status, self.request_id)
        output = [self.items[index] for index in range(len(indexes))]
        return dict(event, response=dict(response, output=output))

    def observe(self, event):
        self.observe_identity(event)
        kind = event["type"]
        if kind == "response.output_text.delta":
            key, delta = self.key(event), event.get("delta")
            if not isinstance(delta, str) or key in self.done:
                self.reject()
            self.parts.setdefault(key, []).append(delta)
        elif kind == "response.output_text.done":
            self.finish(self.key(event), event.get("text"))
        elif kind == "response.content_part.done" and event["part"].get("type") == "output_text":
            self.finish(self.key(event), event["part"].get("text"))
        elif kind == "response.output_item.done" and event["item"].get("type") == "message":
            output_index = self.key(event)[0]
            for index, part in enumerate(event["item"]["content"]):
                self.finish((output_index, index), part.get("text"))
        elif kind == "response.completed":
            output = event["response"]["output"]
            for key in set(self.parts) | set(self.done):
                oi, ci = key
                if oi >= len(output) or output[oi].get("type") != "message":
                    self.reject()
                content = output[oi]["content"]
                if ci >= len(content):
                    self.reject()
                self.finish(key, content[ci].get("text"))


def _screen_event(event, status, request_id):
    kind = event["type"]
    if kind not in _ALLOWED_EVENTS:
        # Structural diagnostic only: no event body, model text, IDs or tokens.
        event_name = (kind if isinstance(kind, str) and re.fullmatch(r"response\.[a-z_.]{1,80}", kind)
                      and not any(word in kind for word in ('token', 'bearer', 'secret', 'authorization')) else None)
        raise TransportError("unsupported_output", "Responses 返回了不受支持的事件，未采纳结果。", status, request_id,
                             param=event_name)
    if kind in _REASONING_EVENTS:
        _reasoning_event(event, status, request_id)
        return
    if kind == "response.metadata":
        _metadata_event(event, status, request_id)
        return
    if "item" in event:
        item = event["item"]
        if isinstance(item, dict) and item.get("type") == "reasoning":
            _reasoning_item(item, status, request_id)
        else:
            _text_only_item(item, status, request_id, kind == "response.output_item.done")
    if "part" in event:
        part = event["part"]
        if isinstance(part, dict) and part.get("type") == "reasoning_text":
            _reasoning_part(part, "reasoning_text", status, request_id)
        elif not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
            raise TransportError("unsupported_output", "Responses 包含非文本内容或拒绝信息，未采纳结果。", status, request_id)
    response = event.get("response")
    if isinstance(response, dict):
        output = response.get("output", [])
        if not isinstance(output, list):
            raise TransportError("invalid_output", "Responses 输出结构无效。", status, request_id)
        for item in output:
            if isinstance(item, dict) and item.get("type") == "reasoning":
                _reasoning_item(item, status, request_id)
            else:
                _text_only_item(item, status, request_id)
        if response.get("tools"):
            raise TransportError("unsupported_output", "Responses 包含未授权工具配置，未采纳结果。", status, request_id)
    if event.get("tool_calls") or event.get("function_call") or event.get("refusal"):
        raise TransportError("unsupported_output", "Responses 包含工具调用或拒绝信息，未采纳结果。", status, request_id)


def _completed(event, status, request_id):
    response = event.get("response")
    if (not isinstance(response, dict) or response.get("status") != "completed" or
            response.get("error") is not None or response.get("incomplete_details") is not None):
        raise TransportError("incomplete_response", "Responses 未提供成功完成的响应。", status, request_id)
    output = response.get("output")
    if not isinstance(output, list) or not output:
        raise TransportError("empty_output", "Responses 已完成但没有助手文本输出。", status, request_id)
    messages = []
    for item in output:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            _reasoning_item(item, status, request_id)
        else:
            messages.append(_text_only_item(item, status, request_id, True))
    text = "\n".join(messages)
    if not text.strip():
        raise TransportError("empty_output", "Responses 已完成但文本为空。", status, request_id)
    result = _json(text, status, request_id)
    if not isinstance(result, dict):
        raise TransportError("invalid_output", "Responses 完成文本必须是 JSON 对象。", status, request_id)
    return result


def _socket_from(response):
    try:
        return response.fp.raw._sock
    except AttributeError:
        return None


def _abort(control):
    control["cancelled"].set()
    response = control.get("response")
    sock = _socket_from(response)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass


def _events(response, deadline, control, status, request_id):
    total = 0
    event_size = 0
    event_name = None
    data = []
    count = 0
    first_line = True
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or control["cancelled"].is_set():
            raise TransportError("timeout", "Responses 请求超过总时限，未采纳任何部分结果。", status, request_id)
        sock = _socket_from(response)
        if sock is not None:
            sock.settimeout(max(0.001, remaining))
        line = response.readline(_MAX_LINE + 1)
        if not isinstance(line, bytes):
            raise TransportError("invalid_stream", "Responses SSE 数据类型无效。", status, request_id)
        if not line:
            raise TransportError("stream_incomplete", "Responses 流结束前未收到 response.completed。", status, request_id)
        total += len(line)
        if len(line) > _MAX_LINE or total > _MAX_STREAM:
            raise TransportError("response_too_large", "Responses 流超过大小限制。", status, request_id)
        if not line.endswith(b"\n"):
            raise TransportError("stream_incomplete", "Responses SSE 事件被截断。", status, request_id)
        if first_line:
            if line.startswith(b"\xef\xbb\xbf"):
                line = line[3:]
            first_line = False
        line = line[:-1]
        if line.endswith(b"\r"):
            line = line[:-1]
        try:
            decoded = line.decode("utf-8")
        except UnicodeDecodeError:
            raise TransportError("invalid_stream", "Responses SSE 不是有效 UTF-8。", status, request_id) from None
        if not decoded:
            if data:
                count += 1
                if count > _MAX_EVENTS:
                    raise TransportError("response_too_large", "Responses 事件数量超过限制。", status, request_id)
                raw = "\n".join(data)
                if raw.strip() == "[DONE]":
                    raise TransportError("stream_incomplete", "Responses 未以 response.completed 完成。", status, request_id)
                event = _json(raw, status, request_id)
                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                    raise TransportError("invalid_stream", "Responses SSE 缺少有效事件类型。", status, request_id)
                if event_name is not None and event_name != event["type"]:
                    raise TransportError("invalid_stream", "Responses SSE 事件名称与数据类型不一致。", status, request_id)
                yield event
            data, event_name, event_size = [], None, 0
            continue
        if decoded.startswith(":"):
            continue
        field, separator, value = decoded.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            event_size += len(line) + 1
            if event_size > _MAX_EVENT:
                raise TransportError("response_too_large", "Responses 单个事件超过大小限制。", status, request_id)
            data.append(value)
        elif field == "event":
            if event_name is not None:
                raise TransportError("invalid_stream", "Responses SSE 事件重复声明类型。", status, request_id)
            event_name = value
        # SSE id/retry/unknown fields never trigger retries or new requests.


def _perform(request, deadline, control, token):
    request_id = None
    status = None
    try:
        opener = urllib.request.build_opener(_NoRedirect())
        remaining = deadline - time.monotonic()
        if remaining <= 0 or control["cancelled"].is_set():
            raise TransportError("timeout", "Responses 请求超过总时限。")
        with opener.open(request, timeout=remaining) as response:
            control["response"] = response
            status = getattr(response, "status", None)
            status = status if isinstance(status, int) else None
            request_id = _request_id(response.headers, token)
            if control["cancelled"].is_set():
                raise TransportError("timeout", "Responses 请求超过总时限。", status, request_id)
            if status != 200:
                raise TransportError("http_error", "Responses 返回非成功 HTTP 状态。", status, request_id)
            raw_content_type = response.headers.get("Content-Type")
            content_type = (raw_content_type or "").split(";", 1)[0].strip().lower()
            events = _events(response, deadline, control, status, request_id)
            # This fixed HTTPS endpoint can omit the header on an otherwise
            # valid SSE response. Never reinterpret an explicitly different type.
            if raw_content_type is not None and content_type != "text/event-stream":
                raise TransportError("invalid_content_type", "Responses 必须返回 SSE 事件流。", status, request_id)
            consistency = _TextConsistency(status, request_id)
            for event in events:
                kind = event["type"]
                if kind in ("response.failed", "response.incomplete", "error", "response.error", "response.cancelled"):
                    _screen_event(dict(event, type="response.in_progress"), status, request_id)
                    payload = event.get("response", event)
                    raise _remote_error(payload, kind.replace(".", "_"), "Responses 未成功完成，部分输出已丢弃。",
                                        status, request_id, token)
                _screen_event(event, status, request_id)
                if kind == "response.completed":
                    event = consistency.completed_event(event)
                    result = _completed(event, status, request_id)
                    consistency.observe(event)
                    return result
                consistency.observe(event)
    except urllib.error.HTTPError as error:
        raise _admission_error(error, token) from None
    except TransportError:
        raise
    except PlanError:
        raise TransportError("redirect_rejected", "Responses 返回重定向，已拒绝转发凭据。", status, request_id) from None
    except (TimeoutError, socket.timeout):
        raise TransportError("timeout", "Responses 请求超时，未采纳部分输出。", status, request_id) from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        raise TransportError("network_error", "Responses 连接失败或中断，未采纳部分输出。", status, request_id) from None


def request_response(model, instructions, text, image_bytes, access_token, timeout=20, reasoning_effort=None,
                     context_images=None):
    """One current JPEG, optionally three explicitly labelled historical views."""
    return _request_response(model, instructions, text, access_token, timeout,
                             image_bytes=image_bytes, with_image=True, reasoning_effort=reasoning_effort,
                             context_images=context_images)


def request_labeled_image_response(model, instructions, text, image_bytes, access_token,
                                   current_image_label, timeout=20, reasoning_effort=None,
                                   context_images=None):
    """Explicit camera provenance for non-photography RGB callers."""
    return _request_response(model, instructions, text, access_token, timeout,
                             image_bytes=image_bytes, with_image=True, reasoning_effort=reasoning_effort,
                             context_images=context_images, current_image_label=current_image_label)


def request_text_response(model, instructions, text, access_token, timeout=20, reasoning_effort=None):
    """Send one text-only request; return strict JSON on completion only."""
    return _request_response(model, instructions, text, access_token, timeout, reasoning_effort=reasoning_effort)


def _request_response(model, instructions, text, access_token, timeout, image_bytes=None, with_image=False,
                      reasoning_effort=None, context_images=None, current_image_label=None):
    """Common fixed-endpoint transport; modality never changes error handling.

    The caller supplies an already-authorized OAuth access token and a model
    selected from that account's catalog. No authentication or fallback happens
    here. A daemon worker bounds caller latency even during DNS or slow reads;
    cancellation shuts down an active socket and discards all late results.
    """
    if (not isinstance(access_token, str) or not access_token or len(access_token) > 32768 or
            not access_token.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in access_token)):
        raise TransportError("invalid_token", "Responses 凭据格式无效。")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", model):
        raise TransportError("invalid_model", "请选择有效的账户模型标识。")
    for value, name in ((instructions, "instructions"), (text, "text")):
        if not isinstance(value, str) or not value.strip() or len(value) > 32768:
            raise TransportError("invalid_input", "Responses {} 必须是 1 到 32768 字符的文字。".format(name))
    if with_image and (not isinstance(image_bytes, (bytes, bytearray)) or not 4 <= len(image_bytes) <= _MAX_IMAGE or
                       not image_bytes.startswith(b"\xff\xd8") or not image_bytes.endswith(b"\xff\xd9")):
        raise TransportError("invalid_image", "Responses 视觉输入必须是不超过 4 MiB 的 JPEG。")
    if context_images is None:
        context_images = []
    if not isinstance(context_images, list) or len(context_images) > 3 or (context_images and not with_image):
        raise TransportError("invalid_image", "历史视觉输入必须附属于当前图片，且最多三张。")
    context_bytes = 0
    for item in context_images:
        if (not isinstance(item, dict) or set(item) != {"label", "data"} or
                not isinstance(item['label'], str) or not 1 <= len(item['label']) <= 250 or
                not isinstance(item['data'], (bytes, bytearray)) or not 4 <= len(item['data']) <= _MAX_IMAGE or
                not item['data'].startswith(b'\xff\xd8') or not item['data'].endswith(b'\xff\xd9')):
            raise TransportError("invalid_image", "历史图片标签或 JPEG 内容无效。")
        context_bytes += len(item['data'])
    if context_bytes > 4*1024*1024:
        raise TransportError("invalid_image", "历史视觉输入总大小超过 4 MiB。")
    if current_image_label is not None and (not with_image or not isinstance(current_image_label, str)
                                           or not 1 <= len(current_image_label) <= 250):
        raise TransportError("invalid_image", "当前图像来源标签必须是 1 到 250 字符的文字。")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TransportError("invalid_timeout", "Responses timeout 必须是 0 到 120 秒之间的有限数值。")
    try:
        valid_timeout = math.isfinite(timeout) and 0 < timeout <= 120
    except (OverflowError, ValueError):
        valid_timeout = False
    if not valid_timeout:
        raise TransportError("invalid_timeout", "Responses timeout 必须是 0 到 120 秒之间的有限数值。")
    if reasoning_effort is not None and (not isinstance(reasoning_effort, str) or
            reasoning_effort not in ("low", "medium", "high", "xhigh", "max")):
        raise TransportError("invalid_reasoning_effort", "Responses reasoning_effort 必须是 low、medium、high、xhigh 或 max。")
    deadline = time.monotonic() + timeout
    content = [{"type": "input_text", "text": text}]
    for item in context_images:
        content.extend([{"type": "input_text", "text": item['label']},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64," +
                         base64.b64encode(bytes(item['data'])).decode('ascii')}])
    if with_image:
        if context_images or current_image_label is not None:
            content.append({"type": "input_text", "text": current_image_label or "CURRENT onboard RGB: only this LAST image may authorize capture. Its id/hash are in the request."})
        content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," +
                        base64.b64encode(bytes(image_bytes)).decode("ascii")})
    payload = {"model": model, "instructions": instructions,
               "input": [{"role": "user", "content": content}], "store": False, "stream": True}
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    try:
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError):
        raise TransportError("invalid_input", "Responses 输入不是有效的 UTF-8 JSON。") from None
    request = urllib.request.Request(_ENDPOINT, data=raw, method="POST", headers={
        "Authorization": "Bearer " + access_token, "Content-Type": "application/json", "Accept": "text/event-stream",
    })
    control = {"cancelled": threading.Event()}
    finished = threading.Event()
    outcome = {}
    def worker():
        try:
            outcome["result"] = _perform(request, deadline, control, access_token)
        except TransportError as error:
            outcome["error"] = error
        except Exception:
            # Never expose exception reprs: request/header errors can contain
            # the OAuth credential, response body, or caller text.
            outcome["error"] = TransportError("transport_error", "Responses 传输失败，未采纳结果。")
        finally:
            finished.set()
    thread = threading.Thread(target=worker, name="responses-transport", daemon=True)
    thread.start()
    if not finished.wait(max(0, deadline - time.monotonic())) or time.monotonic() > deadline:
        _abort(control)
        response = control.get("response")
        status = getattr(response, "status", None)
        identifier = _request_id(getattr(response, "headers", None), access_token)
        raise TransportError("timeout", "Responses 请求超过总时限，未采纳任何部分结果。",
                             status if isinstance(status, int) else None, identifier)
    if "error" in outcome:
        raise outcome["error"] from None
    return outcome["result"]
