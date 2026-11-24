#!/usr/bin/env python3
"""User-operated OAuth sign-in. --help/--status never contact OpenAI."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bridge"))

from uav_bridge.chatgpt_auth import (AuthError, CALLBACK_PATH, CredentialStore,
                                   ID_TOKEN_VALIDATION_MESSAGES)


def public_login_error(error):
    """Only fixed diagnostic text can reach the browser or a local report."""
    messages = {
        "invalid_id_token": "身份令牌校验未通过；没有保存本次授权。",
        "invalid_state": "回调不属于当前登录请求，请使用本次脚本打开的新页面。",
        "expired_attempt": "登录请求已过期或已使用，请重新运行登录脚本。",
        "invalid_callback": "回调缺少有效授权码。",
        "incomplete_registration": "回调未提供签发的客户端 ID，注册未完成。",
        "client_mismatch": "返回的客户端与当前注册不一致。",
        "profile_changed": "登录期间本机账户配置发生变化，请重新登录。",
        "network_error": "连接 OpenAI 失败，请检查网络后重新登录。",
        "invalid_discovery": "官方身份配置校验失败。",
        "invalid_response": "官方接口响应格式无效。",
        "invalid_token_response": "官方授权响应缺少必要字段或权限。",
        "invalid_grant": "授权码或刷新凭证已失效，请重新登录。",
        "access_denied": "本次请求的授权没有获得批准。",
        "invalid_client": "客户端注册未被接受。",
        "unsafe_storage": "本地授权目录或文件的权限检查未通过。",
        "invalid_storage": "本地授权记录格式无效。",
        "login_timeout": "等待浏览器授权超时，请重新运行登录脚本。",
        "subscription_sharing_user_not_eligible": "当前账户或工作区不支持此应用使用计划额度。",
        "subscription_sharing_usage_limit_exceeded": "账户或此应用达到使用限制。",
    }
    code = error.code if error.code in messages else "authorization_failed"
    result = {"code": code, "message": messages.get(code, "登录未完成，请查看终端中的错误码。")}
    reason = getattr(error, "validation_reason", None)
    if code == "invalid_id_token" and reason in ID_TOKEN_VALIDATION_MESSAGES:
        result["validation_reason"] = reason
        result["message"] = ID_TOKEN_VALIDATION_MESSAGES[reason] + " 未保存本次授权。"
    return result


def save_login_error(store, error):
    """Save no URL, code, token, claims, account identity, or server body."""
    diagnostic = dict(public_login_error(error), recorded_at=time.time())
    with store._locked():
        store._write(store.directory / (store.profile + ".last-login-error.json"), diagnostic)


def clear_login_error(store):
    with store._locked():
        path = store.directory / (store.profile + ".last-login-error.json")
        if path.exists() or path.is_symlink():
            path.unlink()


def receive_callback(store, port=0, open_browser=True, reconsent=False):
    """Bind first; accept one valid state on a fixed loopback path, then close."""
    result = {}
    holder = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Authorization codes and state must never appear in HTTP logs.
            pass

        def do_GET(self):
            expected_host = "127.0.0.1:{}".format(self.server.server_port)
            parsed = urlsplit(self.path)
            if (self.client_address[0] != "127.0.0.1" or self.headers.get("Host") != expected_host
                    or parsed.scheme or parsed.netloc or parsed.path != CALLBACK_PATH
                    or len(self.path) > 16384 or parsed.fragment):
                self.reply(400, "Invalid local callback.")
                return
            try:
                values = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=12)
                if any(len(value) != 1 for value in values.values()):
                    raise ValueError()
                query = {key: value[0] for key, value in values.items()}
            except ValueError:
                self.reply(400, "Invalid callback parameters.")
                return
            try:
                result["status"] = store.finish_authorization(holder["attempt"], query)
                self.reply(200, "Authorization received. Return to the terminal for the verified result. You may close this window.")
            except AuthError as exc:
                # Random/mismatched state cannot consume the legitimate attempt.
                if exc.code != "invalid_state":
                    result["error"] = exc
                diagnostic = public_login_error(exc)
                label = diagnostic["code"]
                if "validation_reason" in diagnostic:
                    label += "/" + diagnostic["validation_reason"]
                self.reply(400, "登录未完成 [{}]\n{}\n请返回终端。本页不显示任何登录凭据。".format(
                    label, diagnostic["message"]))

        def reply(self, code, message):
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(message.encode("utf-8"))

    class LoopbackServer(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(3)
            return connection, address

    with LoopbackServer(("127.0.0.1", port), Handler) as server:
        server.timeout = 1
        attempt = store.begin_authorization(server.server_port, reconsent=reconsent)
        holder["attempt"] = attempt
        print("Continue with ChatGPT：本项目将请求使用你的 ChatGPT 计划／积分。")
        print("仅在你完成浏览器授权后保存项目凭证；本次登录不会发送无人机图像或执行推理。")
        print("请在这台 Ubuntu 电脑的浏览器打开以下地址（不要转发该一次性地址）：")
        print(attempt.authorization_url, flush=True)
        if open_browser:
            webbrowser.open(attempt.authorization_url, new=2)
        deadline = time.monotonic() + 600
        while not result and time.monotonic() < deadline:
            server.handle_request()
        if not result:
            raise AuthError("login_timeout", "等待授权超过 10 分钟；未完成登录，请重新运行脚本。")
        if "error" in result:
            raise result["error"]
        return result["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="本项目独立的官方 ChatGPT 登录；不读取 Codex 凭证，也不发送推理请求。")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="仅检查本地授权状态，不联网、不刷新")
    action.add_argument("--list-models", action="store_true", help="使用已授权账号显式获取模型目录；可能刷新 token，不做推理")
    parser.add_argument("--profile", default="default", help="独立账户配置名称，默认 default；新名称可添加另一账号")
    parser.add_argument("--port", type=int, default=0, help="127.0.0.1 回调端口；默认自动分配可用端口")
    parser.add_argument("--no-browser", action="store_true", help="显示登录地址，手动在同一 Ubuntu 主机浏览器打开")
    parser.add_argument("--reconsent", action="store_true", help="显式再次请求计划使用权限，仅用户主动选择时使用")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port 必须在 0 到 65535 之间")
    store = None
    try:
        store = CredentialStore(profile=args.profile)
        if args.status:
            print(json.dumps(store.status(), ensure_ascii=False, indent=2))
            return 0
        if args.list_models:
            models = store.list_models()
            print(json.dumps(models, ensure_ascii=False, indent=2))
            print("请将选定 slug 配置为视觉模型；获取目录成功不代表图像推理已验收。")
            return 0
        status = receive_callback(store, args.port, not args.no_browser, args.reconsent)
        try:
            clear_login_error(store)
        except (AuthError, OSError):
            print("授权已保存，但旧诊断未能清理；以当前授权状态为准。", file=sys.stderr)
        if status["plan_usage_enabled"]:
            print("已核验账户身份并保存本项目授权；使用的是你的 ChatGPT 计划／积分。")
            print("该账号的最终资格和模型可用性仍由服务端决定，尚未发送推理请求。")
            print("下一步显式获取目录：python3 scripts/login_chatgpt.py --profile {} --list-models".format(args.profile))
        else:
            print("身份已登录，但未授予 ChatGPT 计划使用权限；推理已禁用。")
            print("如需授权，请由你主动重新运行并增加 --reconsent。")
        print("管理应用权限和额度：https://chatgpt.com/#settings/Usage")
        return 0 if status["plan_usage_enabled"] else 2
    except AuthError as exc:
        print("登录未完成 [{}]：{}".format(exc.code, exc), file=sys.stderr)
        if exc.validation_reason:
            print("校验失败项：" + exc.validation_reason, file=sys.stderr)
        if store is not None:
            try:
                save_login_error(store, exc)
                print("不含凭据的诊断已保存在本项目授权目录。", file=sys.stderr)
            except (AuthError, OSError):
                print("无法保存诊断；原始登录错误如上。", file=sys.stderr)
        if exc.request_id:
            print("诊断请求 ID：" + exc.request_id, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已取消本次登录。", file=sys.stderr)
        return 130
    except OSError:
        print("本机回调端口或凭证目录不可用；请检查权限或换用 --port 0。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
