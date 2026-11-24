"""Small authenticated HTTP API; Python 3.8 standard library only."""

import hmac
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import BridgeError


class BridgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, address, bridge, token):
        if not isinstance(token, str) or len(token) < 16 or not token.isascii() or token.isspace():
            raise ValueError("Bridge token must contain at least 16 ASCII characters")
        self.bridge, self.token = bridge, token
        self.preview_slots = threading.BoundedSemaphore(2)
        self.connection_slots = threading.BoundedSemaphore(24)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.connection_slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connection_slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "UAVBridge/0.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5.0)

    def log_message(self, fmt, *args):
        # Bodies, tokens, model credentials and natural-language text are never logged.
        logging.info("%s %s", self.client_address[0], fmt % args)

    def _send(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authenticate(self):
        supplied = self.headers.get("Authorization", "")
        expected = "Bearer " + self.server.token
        if not supplied.isascii() or not hmac.compare_digest(supplied, expected):
            raise BridgeError("unauthorized", "手机访问令牌不匹配，请重新填写当前服务的访问令牌。", 401)
        self.server.bridge.note_client()

    def _send_camera(self, evidence_id=None):
        jpeg, metadata = (self.server.bridge.evidence_frame(evidence_id) if evidence_id is not None
                          else self.server.bridge.camera_frame())
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        stamp = "%.6f" % metadata["stamp"]
        self.send_header("X-Frame-Timestamp", stamp)
        self.send_header("X-Camera-Timestamp", stamp)
        if evidence_id is not None:
            self.send_header("X-Evidence-Id", evidence_id)
        for field in ("width", "height"):
            if field in metadata:
                self.send_header("X-Frame-" + field.title(), str(metadata[field]))
        self.end_headers()
        try:
            self.wfile.write(jpeg)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_photo(self, photo_id):
        jpeg, metadata = self.server.bridge.photo_frame(photo_id)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Disposition", 'attachment; filename="photo-' + photo_id + '.jpg"')
        self.send_header("X-Photo-Id", photo_id)
        self.send_header("X-Image-SHA256", metadata["image_sha256"])
        self.send_header("X-Frame-Timestamp", "%.6f" % metadata["stamp"])
        self.end_headers()
        try:
            self.wfile.write(jpeg)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise BridgeError("unsupported_encoding", "Chunked requests are not supported", 400)
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            raise BridgeError("content_type", "Use Content-Type: application/json", 415)
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            raise BridgeError("invalid_length", "A valid Content-Length is required", 411)
        if length < 0:
            raise BridgeError("invalid_length", "Content-Length is required", 411)
        if length > 65536:
            raise BridgeError("body_too_large", "Request body exceeds 64 KiB", 413)
        def reject_constant(_):
            raise ValueError("Non-finite JSON number")
        def unique_object(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("Duplicate JSON key")
                obj[key] = value
            return obj
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"),
                              parse_constant=reject_constant, object_pairs_hook=unique_object)
        except (ValueError, UnicodeError, TimeoutError):
            raise BridgeError("invalid_json", "Body must be valid JSON with unique keys and finite numbers")
        if not isinstance(body, dict):
            raise BridgeError("invalid_body", "Expected a JSON object")
        return body

    def _serve(self, method):
        try:
            self._authenticate()
            if method == "GET" and self.path == "/api/status":
                self._send(200, self.server.bridge.status())
                return
            if method == "GET" and self.path == "/api/missions":
                self._send(200, self.server.bridge.missions())
                return
            if method == "GET" and self.path == "/api/logs":
                self._send(200, self.server.bridge.logs())
                return
            if method == "GET" and self.path == "/api/models":
                self._send(200, self.server.bridge.models())
                return
            if method == "GET" and self.path == "/api/camera.jpg":
                self._send_camera()
                return
            photo_match = re.fullmatch(r"/api/photos/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\.jpg", self.path)
            if method == "GET" and photo_match:
                self._send_photo(photo_match.group(1))
                return
            evidence_match = re.fullmatch(r"/api/evidence/([a-f0-9-]{36})\.jpg", self.path)
            if method == "GET" and evidence_match:
                self._send_camera(evidence_match.group(1))
                return
            if method != "POST":
                raise BridgeError("not_found", "Unknown API route", 404)
            if self.path == "/api/model":
                self._send(200, self.server.bridge.select_model(self._body()))
                return
            routes = {"/api/takeoff": "takeoff", "/api/land": "land", "/api/hold": "hold", "/api/goto": "goto", "/api/mission/execute": "execute"}
            planning_routes = ("/api/mission/preview", "/api/photo/subject/preview", "/api/photo/task/parse")
            if self.path not in routes and self.path not in planning_routes:
                raise BridgeError("not_found", "Unknown API route", 404)
            body = self._body()
            if self.path in planning_routes:
                if not self.server.preview_slots.acquire(blocking=False):
                    raise BridgeError("planner_busy", "Planner is busy; retry shortly", 429)
                try:
                    if self.path == "/api/photo/task/parse":
                        preview = self.server.bridge.parse_photo_task(body)
                    elif self.path == "/api/photo/subject/preview":
                        preview = self.server.bridge.subject_preview(body)
                    else:
                        preview = self.server.bridge.preview(body)
                    self._send(200, preview)
                finally:
                    self.server.preview_slots.release()
            else:
                self._send(202, self.server.bridge.command(routes[self.path], body))
        except BridgeError as exc:
            error = {"code": exc.code, "message": exc.message}
            if exc.details is not None:
                error["details"] = exc.details
            self._send(exc.status, {"error": error})
        except Exception:
            logging.exception("Unhandled API error")
            self._send(500, {"error": {"code": "internal_error", "message": "Bridge internal error; inspect the host log"}})

    def do_GET(self):
        self._serve("GET")

    def do_POST(self):
        self._serve("POST")
