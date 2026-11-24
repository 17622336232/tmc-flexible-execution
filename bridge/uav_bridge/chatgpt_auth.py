"""Official, app-owned Sign in with ChatGPT credentials for local clients.

No import or status call opens a browser, contacts OpenAI, or reads Codex's
credentials. Network operations are limited to explicit authorization,
get_access_token() renewal, and list_models(). See the project's login script.
Protocol: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
"""
import base64
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
import time
from urllib import error, parse, request
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
MODELS_URL = RESOURCE + "/models"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + DIRECT_SCOPE
CALLBACK_PATH = "/auth/callback"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / ".runtime" / "chatgpt"
MAX_RESPONSE_BYTES = 1024 * 1024

# Only these static identifiers/messages may reach browser or saved diagnostics.
# Never interpolate a JWT, claim value, client ID, key ID, or nonce into them.
ID_TOKEN_VALIDATION_MESSAGES = {
    "token_format": "令牌的分段或编码格式无效。",
    "json_format": "令牌头或声明不是有效的唯一字段 JSON 对象。",
    "unsupported_algorithm": "签名算法不在本项目允许列表中。",
    "unsupported_critical_header": "令牌要求了本项目不支持的关键扩展。",
    "kid_missing": "令牌缺少有效的签名密钥标识。",
    "jwks_format": "官方签名密钥集合格式无效。",
    "kid_unknown": "当前官方密钥集合未包含该令牌的签名密钥。",
    "key_not_usable": "匹配密钥的用途、操作权限或算法不允许验证此令牌。",
    "key_ambiguous": "官方密钥集合包含多个可用的同标识密钥。",
    "key_type_mismatch": "签名算法与公钥类型不匹配。",
    "key_invalid": "签名公钥参数无效。",
    "weak_rsa_key": "RSA 公钥长度低于本项目要求。",
    "signature_format": "签名编码或长度无效。",
    "signature_invalid": "令牌签名验证失败。",
    "issuer_mismatch": "令牌签发方与受信任的 OpenAI 签发方不一致。",
    "audience_invalid": "令牌受众字段格式无效。",
    "audience_mismatch": "令牌受众未包含本次授权的客户端。",
    "authorized_party_missing": "多受众令牌缺少授权方声明。",
    "authorized_party_mismatch": "令牌授权方与本次授权客户端不一致。",
    "clock_invalid": "本机用于验证的时间无效。",
    "expiration_missing": "令牌缺少过期时间。",
    "expiration_invalid": "令牌过期时间不是有效数值。",
    "token_expired": "令牌已过期，或本机时间与签发服务相差过大。",
    "issued_at_missing": "令牌缺少签发时间。",
    "issued_at_invalid": "令牌签发时间不是有效数值。",
    "issued_at_future": "令牌签发时间晚于本机允许范围，请核对系统时间。",
    "time_order_invalid": "令牌过期时间没有晚于签发时间。",
    "not_before_invalid": "令牌生效时间不是有效数值。",
    "not_before_future": "令牌尚未生效，或本机时间与签发服务相差过大。",
    "subject_missing": "令牌缺少账户身份声明。",
    "subject_invalid": "令牌账户身份声明格式无效。",
    "subject_mismatch": "重新登录的身份与已保存账户不一致。",
    "nonce_missing": "令牌缺少本次登录随机值声明。",
    "nonce_invalid": "令牌登录随机值声明格式无效。",
    "nonce_mismatch": "令牌登录随机值与本次登录事务不一致。",
    "access_hash_invalid": "令牌中的访问令牌摘要格式无效。",
    "access_hash_missing_token": "无法核对访问令牌摘要，因为缺少有效访问令牌。",
    "access_hash_mismatch": "访问令牌与身份令牌中的摘要不匹配。",
}


class AuthError(RuntimeError):
    """Safe diagnostic: never contains token values, code, or raw server body."""

    def __init__(self, code, message, status=None, request_id=None, validation_reason=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id
        self.validation_reason = (validation_reason if code == "invalid_id_token"
                                  and isinstance(validation_reason, str)
                                  and validation_reason in ID_TOKEN_VALIDATION_MESSAGES else None)


def _id_token_error(reason):
    return AuthError("invalid_id_token", "ID token 校验失败：" + ID_TOKEN_VALIDATION_MESSAGES[reason]
                     + " 未保存该授权。", validation_reason=reason)


def server_error(status, body=None, request_id=None):
    body = body if isinstance(body, dict) else {}
    nested = body.get("error")
    code = nested.get("code") if isinstance(nested, dict) else nested
    if not isinstance(code, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", code):
        code = "http_{}".format(status)
    messages = {
        "subscription_sharing_user_not_eligible": "当前账号、工作区或政策不支持此应用使用 ChatGPT 计划；不会循环重试登录。",
        "subscription_sharing_usage_limit_exceeded": "ChatGPT 计划或此应用达到使用限制，请在 ChatGPT 设置 → Usage 查看。",
        "subscription_sharing_usage_unavailable": "暂时无法核验 ChatGPT 使用额度，请稍后重试。",
        "subscription_sharing_unsupported_capability": "本请求使用了 ChatGPT 计划通道不支持的模型、参数或能力。",
        "subscription_sharing_route_not_supported": "此 ChatGPT 计划通道不支持请求的接口。",
        "invalid_grant": "授权码或刷新凭证已失效；请重新运行本项目的登录脚本。",
        "access_denied": "用户未授予此应用请求的权限。",
        "invalid_client": "本项目客户端注册未被接受，请重新登录核验注册。",
    }
    fallback = {401: "ChatGPT 授权未被接受，请重新登录并核验计划使用权限。",
                403: "当前账号、工作区、地区或政策拒绝了请求；未切换到其他计费方式。",
                429: "请求达到使用限制，请查看 ChatGPT 设置 → Usage。",
                503: "ChatGPT 计划通道暂不可用或尚未开放给此客户端。"}
    safe_id = request_id if isinstance(request_id, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", request_id) else None
    return AuthError(code, messages.get(code, fallback.get(status, "OpenAI 请求失败，HTTP {}（{}）。".format(status, code))), status, safe_id)


def _strict_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON member")
        obj[key] = value
    return obj


def _json_loads(value):
    return json.loads(value, object_pairs_hook=_strict_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_json(method, url, data=None, headers=None):
    """Fixed trusted HTTPS destinations; redirects never receive credentials."""
    parsed = parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ("auth.openai.com", "api.openai.com")
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment):
        raise AuthError("untrusted_endpoint", "拒绝向非官方 HTTPS 地址发送授权数据。")
    req = request.Request(url, data=data, headers=dict(headers or {}), method=method)
    req.add_header("Accept", "application/json")
    opener = request.build_opener(_NoRedirect())
    try:
        with opener.open(req, timeout=20) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except error.HTTPError as exc:
        try:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            body = _json_loads(raw) if len(raw) <= MAX_RESPONSE_BYTES else {}
        except (ValueError, UnicodeError):
            body = {}
        raise server_error(exc.code, body, exc.headers.get("x-request-id") or exc.headers.get("openai-request-id")) from None
    except (error.URLError, OSError, TimeoutError):
        raise AuthError("network_error", "无法连接 OpenAI；未更改已保存授权，也未改用其他计费方式。") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise AuthError("invalid_response", "OpenAI 响应超过允许大小。")
    try:
        result = _json_loads(raw)
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except (ValueError, UnicodeError):
        raise AuthError("invalid_response", "OpenAI 返回的 JSON 无效。") from None


def _b64(data):
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid base64url")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def verify_id_token(token, jwks, client_id, nonce, now, subject=None, access_token=None):
    """Verify RSA/EC SHA-256 signatures and OIDC claims; never trust JWT URLs."""
    try:
        if not isinstance(token, str) or len(token) > 65536:
            raise ValueError()
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError()
        decoded = [_unb64(part) for part in parts[:2]]
    except (ValueError, TypeError, UnicodeError):
        raise _id_token_error("token_format") from None
    try:
        header, claims = (_json_loads(part) for part in decoded)
        if not isinstance(header, dict) or not isinstance(claims, dict):
            raise ValueError()
    except (ValueError, TypeError, UnicodeError):
        raise _id_token_error("json_format") from None
    alg, kid = header.get("alg"), header.get("kid")
    if alg not in ("RS256", "ES256"):
        raise _id_token_error("unsupported_algorithm")
    if header.get("crit"):
        raise _id_token_error("unsupported_critical_header")
    if not isinstance(kid, str) or not kid:
        raise _id_token_error("kid_missing")
    if (not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list)
            or not all(isinstance(key, dict) for key in jwks["keys"])):
        raise _id_token_error("jwks_format")
    candidates = [key for key in jwks["keys"] if key.get("kid") == kid]
    if not candidates:
        raise _id_token_error("kid_unknown")
    try:
        keys = [key for key in candidates if
                key.get("use", "sig") == "sig" and key.get("alg", alg) == alg and
                ("key_ops" not in key or "verify" in key["key_ops"])]
    except TypeError:
        raise _id_token_error("key_not_usable") from None
    if not keys:
        raise _id_token_error("key_not_usable")
    if len(keys) != 1:
        raise _id_token_error("key_ambiguous")
    key = keys[0]
    try:
        signature = _unb64(parts[2])
        signed = (parts[0] + "." + parts[1]).encode("ascii")
    except (ValueError, TypeError, UnicodeError):
        raise _id_token_error("signature_format") from None
    if ((alg == "RS256" and key.get("kty") != "RSA")
            or (alg == "ES256" and (key.get("kty") != "EC" or key.get("crv") != "P-256"))):
        raise _id_token_error("key_type_mismatch")
    try:
        if alg == "RS256":
            public = rsa.RSAPublicNumbers(int.from_bytes(_unb64(key["e"]), "big"),
                                          int.from_bytes(_unb64(key["n"]), "big")).public_key(default_backend())
        else:
            public = ec.EllipticCurvePublicNumbers(int.from_bytes(_unb64(key["x"]), "big"),
                                                   int.from_bytes(_unb64(key["y"]), "big"), ec.SECP256R1()).public_key(default_backend())
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError):
        raise _id_token_error("key_invalid") from None
    if alg == "RS256" and public.key_size < 2048:
        raise _id_token_error("weak_rsa_key")
    if alg == "ES256" and len(signature) != 64:
        raise _id_token_error("signature_format")
    try:
        if alg == "RS256":
            public.verify(signature, signed, padding.PKCS1v15(), hashes.SHA256())
        else:
            der = utils.encode_dss_signature(int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big"))
            public.verify(der, signed, ec.ECDSA(hashes.SHA256()))
    except (ValueError, TypeError, InvalidSignature):
        raise _id_token_error("signature_invalid") from None
    # Claim diagnostics are reached only after verification against a trusted key.
    if claims.get("iss") != ISSUER:
        raise _id_token_error("issuer_mismatch")
    audience = claims.get("aud")
    if isinstance(audience, str):
        audience = [audience]
    if not isinstance(audience, list) or not all(isinstance(item, str) for item in audience):
        raise _id_token_error("audience_invalid")
    if client_id not in audience:
        raise _id_token_error("audience_mismatch")
    if len(audience) > 1 and "azp" not in claims:
        raise _id_token_error("authorized_party_missing")
    if "azp" in claims and claims["azp"] != client_id:
        raise _id_token_error("authorized_party_mismatch")
    if not _number(now):
        raise _id_token_error("clock_invalid")
    if "exp" not in claims:
        raise _id_token_error("expiration_missing")
    if not _number(claims["exp"]):
        raise _id_token_error("expiration_invalid")
    if claims["exp"] <= now - 5:
        raise _id_token_error("token_expired")
    if "iat" not in claims:
        raise _id_token_error("issued_at_missing")
    if not _number(claims["iat"]):
        raise _id_token_error("issued_at_invalid")
    if claims["iat"] > now + 5:
        raise _id_token_error("issued_at_future")
    if claims["exp"] <= claims["iat"]:
        raise _id_token_error("time_order_invalid")
    if "nbf" in claims:
        if not _number(claims["nbf"]):
            raise _id_token_error("not_before_invalid")
        if claims["nbf"] > now + 5:
            raise _id_token_error("not_before_future")
    if "sub" not in claims:
        raise _id_token_error("subject_missing")
    if not isinstance(claims["sub"], str) or not claims["sub"]:
        raise _id_token_error("subject_invalid")
    if subject and claims["sub"] != subject:
        raise _id_token_error("subject_mismatch")
    if nonce is not None:
        if "nonce" not in claims:
            raise _id_token_error("nonce_missing")
        if not isinstance(claims["nonce"], str):
            raise _id_token_error("nonce_invalid")
        try:
            matched = secrets.compare_digest(claims["nonce"], nonce)
        except TypeError:
            raise _id_token_error("nonce_invalid") from None
        if not matched:
            raise _id_token_error("nonce_mismatch")
    if "at_hash" in claims:
        if not isinstance(claims["at_hash"], str):
            raise _id_token_error("access_hash_invalid")
        try:
            expected = _b64(hashlib.sha256(access_token.encode("ascii")).digest()[:16]) if access_token else None
        except (AttributeError, UnicodeError):
            raise _id_token_error("access_hash_missing_token") from None
        if not expected:
            raise _id_token_error("access_hash_missing_token")
        try:
            matched = secrets.compare_digest(claims["at_hash"], expected)
        except TypeError:
            raise _id_token_error("access_hash_invalid") from None
        if not matched:
            raise _id_token_error("access_hash_mismatch")
    return claims


class AuthorizationAttempt:
    """One in-memory, expiring OAuth transaction. repr intentionally hides secrets."""
    def __init__(self, host_id, port, registration=None, now=None, reconsent=False):
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise AuthError("invalid_callback", "回调必须使用本机 127.0.0.1 的有效端口。")
        self.state = secrets.token_urlsafe(32)
        self.nonce = secrets.token_urlsafe(32)
        self.verifier = secrets.token_urlsafe(64)
        self.redirect_uri = "http://127.0.0.1:{}{}".format(port, CALLBACK_PATH)
        self.registration = dict(registration or {})
        self.client_id = self.registration.get("client_id") or "dynamic_agent_client"
        self.host_id = host_id
        self.expires_at = (time.time() if now is None else now) + 600
        self.consumed = False
        self._lock = threading.Lock()
        params = {"client_id": self.client_id, "ext_agent_host_id": host_id,
                  "response_type": "code", "redirect_uri": self.redirect_uri,
                  "scope": SCOPES, "resource": RESOURCE, "state": self.state,
                  "nonce": self.nonce, "code_challenge_method": "S256",
                  "code_challenge": _b64(hashlib.sha256(self.verifier.encode("ascii")).digest())}
        if self.client_id == "dynamic_agent_client":
            params["agent_name_hint"] = "Diff-VLA"
        if reconsent:
            params["prompt"] = "consent"
        # Optional id_token_hint is deliberately omitted: no credential in URLs.
        self.authorization_url = AUTHORIZE_URL + "?" + parse.urlencode(params)

    def consume(self, query, now):
        with self._lock:
            if self.consumed or now >= self.expires_at:
                raise AuthError("expired_attempt", "登录请求已过期或已使用，请重新运行登录脚本。")
            if (not isinstance(query, dict) or not isinstance(query.get("state"), str)
                    or not secrets.compare_digest(query["state"], self.state)):
                raise AuthError("invalid_state", "回调 state 不匹配，已拒绝该回调。")
            self.consumed = True
            if query.get("error"):
                raise server_error(400, {"error": query["error"]})
            code = query.get("code")
            issued = query.get("client_id", self.client_id)
            if not isinstance(code, str) or not code or len(code) > 8192:
                raise AuthError("invalid_callback", "回调缺少有效授权码。")
            if not isinstance(issued, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{3,200}", issued) or issued == "dynamic_agent_client":
                raise AuthError("incomplete_registration", "回调未提供签发的客户端 ID，注册未完成。")
            if self.client_id != "dynamic_agent_client" and issued != self.client_id:
                raise AuthError("client_mismatch", "重新登录返回了不同客户端，已拒绝替换现有账户。")
            return code, issued


class CredentialStore:
    """Local profile storage; constructing it and status() never use the network."""
    def __init__(self, directory=None, profile="default", transport=None, clock=None):
        if not isinstance(profile, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", profile):
            raise AuthError("invalid_profile", "账户配置名称只能含字母、数字、下划线和短横线。")
        self.directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
        self.profile = profile
        self.path = self.directory / (profile + ".json")
        self.transport = transport or http_json
        self.clock = clock or time.time

    def _prepare_directory(self):
        if self.directory.is_symlink():
            raise AuthError("unsafe_storage", "凭证目录不能是符号链接。")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.stat()
        if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode):
            raise AuthError("unsafe_storage", "凭证目录必须归当前用户所有。")
        os.chmod(str(self.directory), 0o700)

    @contextlib.contextmanager
    def _locked(self):
        self._prepare_directory()
        fd = os.open(str(self.directory / ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise AuthError("unsafe_storage", "凭证锁文件不安全。")
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read(self, path):
        try:
            fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError:
            raise AuthError("unsafe_storage", "无法安全读取本项目凭证文件。") from None
        try:
            info = os.fstat(fd)
            if (info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077):
                raise AuthError("unsafe_storage", "凭证必须是当前用户独占的普通文件（权限 600）。")
            with os.fdopen(fd, "rb") as stream:
                fd = None
                raw = stream.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError()
            value = _json_loads(raw)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, UnicodeError):
            raise AuthError("invalid_storage", "本项目凭证记录损坏，请重新登录。") from None
        finally:
            if fd is not None:
                os.close(fd)

    def _write(self, path, value):
        if path.is_symlink():
            raise AuthError("unsafe_storage", "凭证文件不能是符号链接。")
        fd, temp = tempfile.mkstemp(prefix=".auth-", dir=str(self.directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, str(path))
            directory_fd = os.open(str(self.directory), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def _host_id(self):
        path = self.directory / "host.json"
        host = self._read(path)
        if host is None:
            host = {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())}
            self._write(path, host)
        value = host.get("ext_agent_host_id", "")
        try:
            if not value.startswith("urn:uuid:") or uuid.UUID(value[9:]).version != 4:
                raise ValueError()
        except (ValueError, AttributeError, TypeError):
            raise AuthError("invalid_host", "本项目的持久化 host ID 无效，请检查凭证目录。") from None
        return value

    def _jwks(self):
        discovery = self.transport("GET", DISCOVERY_URL)
        jwks_uri = discovery.get("jwks_uri")
        if (discovery.get("issuer") != ISSUER or discovery.get("authorization_endpoint") != AUTHORIZE_URL
                or discovery.get("token_endpoint") != TOKEN_URL or not isinstance(jwks_uri, str)):
            raise AuthError("invalid_discovery", "OpenAI 身份发现文档与受信任端点不符。")
        parsed = parse.urlsplit(jwks_uri)
        if (parsed.scheme != "https" or parsed.netloc != "auth.openai.com"
                or parsed.query or parsed.fragment or not parsed.path.startswith("/")):
            raise AuthError("invalid_discovery", "OpenAI JWKS 地址不在受信任域名。")
        return self.transport("GET", jwks_uri)

    def begin_authorization(self, port, reconsent=False):
        with self._locked():
            record = self._read(self.path) or self._read(self.directory / (self.profile + ".registration.json"))
            host = self._host_id()
            return AuthorizationAttempt(host, port, record, self.clock(), reconsent)

    def _token_request(self, fields):
        return self.transport("POST", TOKEN_URL, parse.urlencode(fields).encode("ascii"),
                              {"Content-Type": "application/x-www-form-urlencoded"})

    def _token_record(self, response, client_id, host_id, nonce=None, previous=None, jwks=None):
        previous = previous or {}
        scope = response.get("scope")
        if not isinstance(scope, str):
            raise AuthError("invalid_token_response", "OAuth 未返回实际授予的权限。")
        granted = scope.split()
        direct = DIRECT_SCOPE in granted
        if response.get("token_type", "").lower() != "bearer" and (direct or response.get("access_token")):
            raise AuthError("invalid_token_response", "OAuth 未返回 Bearer 类型授权。")
        access = response.get("access_token")
        refresh = response.get("refresh_token")
        expires = response.get("expires_in", 0)
        if ((direct and not access) or (access is not None and (not isinstance(access, str) or not re.fullmatch(r"[\x21-\x7e]{1,65536}", access)))
                or (refresh is not None and (not isinstance(refresh, str) or not re.fullmatch(r"[\x21-\x7e]{1,65536}", refresh)))
                or ("offline_access" in granted and not refresh)
                or not _number(expires) or not 0 <= expires <= 86400 or (access and expires <= 0)):
            raise AuthError("invalid_token_response", "OAuth 缺少有效 token、轮换刷新凭证、权限或有效期。")
        now = self.clock()
        id_token = response.get("id_token")
        if id_token is not None:
            identity = verify_id_token(id_token, jwks if jwks is not None else self._jwks(), client_id, nonce, now,
                                       previous.get("subject"), access)
        elif nonce is not None or not previous.get("subject"):
            raise AuthError("invalid_token_response", "初次授权缺少 ID token，不能确认账户身份。")
        else:
            identity = {"sub": previous["subject"], "email": previous.get("email")}
            id_token = previous.get("id_token")
        if "openid" not in granted:
            raise AuthError("invalid_token_response", "OAuth 未授予所需身份权限。")
        earliest = response.get("earliest_refresh_at", 0)
        if not _number(earliest) or earliest < 0:
            raise AuthError("invalid_token_response", "OAuth 刷新时间无效。")
        return {"issuer": ISSUER, "subject": identity["sub"], "email": identity.get("email"),
                "client_id": client_id, "ext_agent_host_id": host_id, "id_token": id_token,
                "access_token": access, "refresh_token": refresh, "scopes": granted,
                "expires_at": now + expires, "earliest_refresh_at": earliest,
                "saved_at": now, "refresh_expires_at": now + 30 * 86400 if refresh else 0,
                "redirect_path": CALLBACK_PATH}

    def finish_authorization(self, attempt, query):
        code, client_id = attempt.consume(query, self.clock())
        with self._locked():
            pending_path = self.directory / (self.profile + ".registration.json")
            current = self._read(self.path) or self._read(pending_path)
            expected = attempt.registration
            if (current or {}).get("client_id") != expected.get("client_id"):
                raise AuthError("profile_changed", "登录过程中此账户配置发生变更，请重新登录。")
            # Retain an issued registration if exchange fails, without activating identity.
            if not current:
                self._write(pending_path, {"client_id": client_id, "ext_agent_host_id": attempt.host_id})
            jwks = self._jwks()
            response = self._token_request({"grant_type": "authorization_code", "client_id": client_id,
                                           "code": code, "code_verifier": attempt.verifier,
                                           "redirect_uri": attempt.redirect_uri, "resource": RESOURCE})
            record = self._token_record(response, client_id, attempt.host_id, attempt.nonce, expected, jwks)
            # A valid identity without direct scope remains signed in, but cannot infer.
            self._write(self.path, record)
            if pending_path.exists():
                pending_path.unlink()
        return self.status()

    def status(self):
        result = {"profile": self.profile, "signed_in": False, "plan_usage_enabled": False,
                  "access_token_valid": False, "refresh_available": False}
        try:
            if self.directory.is_symlink():
                raise AuthError("unsafe_storage", "凭证目录不能是符号链接。")
            record = self._read(self.path)
            if not record:
                return result
            now = self.clock()
            result.update(signed_in=bool(record.get("subject") and record.get("id_token")),
                          plan_usage_enabled=DIRECT_SCOPE in record.get("scopes", []),
                          access_token_valid=bool(record.get("access_token") and _number(record.get("expires_at")) and record["expires_at"] > now),
                          refresh_available=bool(record.get("refresh_token") and not record.get("refresh_in_progress") and _number(record.get("refresh_expires_at")) and record["refresh_expires_at"] > now),
                          email=record.get("email"), client_id=record.get("client_id"))
            if record.get("refresh_in_progress"):
                result.update(access_token_valid=False, error="refresh_interrupted", message="上次刷新未确认完成，请重新登录，避免重复使用轮换凭证。")
        except AuthError as exc:
            result.update(error=exc.code, message=str(exc))
        except (TypeError, KeyError, ValueError):
            result.update(error="invalid_storage", message="本项目授权记录格式无效，请重新登录。")
        return result

    def get_access_token(self):
        with self._locked():
            record = self._read(self.path)
            if not record or not record.get("subject"):
                raise AuthError("not_signed_in", "请先运行 python3 scripts/login_chatgpt.py 登录本项目。")
            if record.get("refresh_in_progress"):
                raise AuthError("refresh_interrupted", "上次刷新未确认完成，请重新登录，避免重复使用轮换凭证。")
            if DIRECT_SCOPE not in record.get("scopes", []) or "resource.invoke" not in record.get("scopes", []):
                raise AuthError("plan_usage_disabled", "当前登录未授权使用 ChatGPT 计划；请显式重新授权。")
            now = self.clock()
            expires = record.get("expires_at", 0)
            earliest = record.get("earliest_refresh_at", 0)
            if not _number(expires) or not _number(earliest):
                raise AuthError("invalid_storage", "凭证时间字段无效，请重新登录。")
            if record.get("access_token") and expires > now + 60:
                return record["access_token"]
            if earliest > now:
                if expires > now:
                    return record["access_token"]
                raise AuthError("refresh_not_ready", "授权已过期但服务端尚不允许刷新，请稍后重试。")
            if not record.get("refresh_token") or record.get("refresh_expires_at", 0) <= now:
                raise AuthError("refresh_expired", "本项目刷新授权已过期，请重新登录。")
            # Cross-process flock covers the request and atomic replacement: one rotation.
            jwks = self._jwks()  # Fetch before rotating, so a discovery outage loses no replacement.
            self._write(self.path, dict(record, refresh_in_progress=True))
            try:
                response = self._token_request({"grant_type": "refresh_token", "client_id": record["client_id"],
                                               "refresh_token": record["refresh_token"], "resource": RESOURCE})
            except AuthError as exc:
                if exc.code in {"network_error", "invalid_response"}:
                    # The server may already have rotated before the response was lost.
                    # Keep tokens protected, but prohibit automatic reuse of this session.
                    raise AuthError("refresh_interrupted", "刷新响应未确认完整，可能已轮换；请重新登录，旧刷新凭证不会再次发送。") from None
                if exc.code in {"invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused"}:
                    record = {key: value for key, value in record.items() if key not in {"access_token", "refresh_token", "id_token"}}
                    record.update(expires_at=0, refresh_expires_at=0)
                # Explicit service failures preserve the token set; no automatic retry loop.
                self._write(self.path, record)
                raise
            renewed = self._token_record(response, record["client_id"], record["ext_agent_host_id"], previous=record, jwks=jwks)
            self._write(self.path, renewed)
            if DIRECT_SCOPE not in renewed["scopes"] or "resource.invoke" not in renewed["scopes"]:
                raise AuthError("plan_usage_disabled", "刷新后的授权未包含 ChatGPT 计划使用权限，已停止请求。")
            return renewed["access_token"]

    def list_models(self):
        """Explicit user action only; no inference request is made here."""
        token = self.get_access_token()
        with self._locked():
            before = self._read(self.path)
            if not before or before.get("access_token") != token:
                raise AuthError("profile_changed", "获取模型目录前授权发生变化，请重试。")
        result = self.transport("GET", MODELS_URL, headers={"Authorization": "Bearer " + token})
        models = result.get("models")
        if not isinstance(models, list):
            raise AuthError("invalid_catalog", "ChatGPT 模型目录格式不符，未保存目录。")
        visible = [{"slug": item["slug"], "display_name": item.get("display_name", item["slug"])}
                   for item in models if isinstance(item, dict) and item.get("visibility") == "list"
                   and isinstance(item.get("slug"), str) and item["slug"]]
        with self._locked():
            current = self._read(self.path)
            if not current or (current.get("client_id"), current.get("subject")) != (before.get("client_id"), before.get("subject")):
                raise AuthError("profile_changed", "获取模型目录时账户发生变化，请重试。")
            self._write(self.directory / (self.profile + ".models.json"),
                        {"client_id": current["client_id"], "subject": current["subject"],
                         "fetched_at": self.clock(), "models": visible})
        return visible

    def cached_models(self):
        try:
            if self.directory.is_symlink():
                return []
            record = self._read(self.path)
            catalog = self._read(self.directory / (self.profile + ".models.json"))
            if not record or not catalog or (record.get("client_id"), record.get("subject")) != (catalog.get("client_id"), catalog.get("subject")):
                return []
            models = catalog.get("models", [])
            return models if isinstance(models, list) else []
        except AuthError:
            return []

    def is_model_available(self, model):
        return any(item.get("slug") == model for item in self.cached_models() if isinstance(item, dict))
