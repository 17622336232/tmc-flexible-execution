"""python3 -m uav_bridge --mock | --adapter bridge_ros.ros_adapter:create_adapter"""

import argparse
import importlib
import logging
import os
import signal
import threading
import time
from pathlib import Path

from .config import load_config
from .core import Bridge
from .mock_adapter import MockAdapter
from .server import BridgeHTTPServer
from .mobile_settings import MobileState, model_name
from .vision import VisionService


def main():
    parser = argparse.ArgumentParser(description="Diff-VLA authenticated phone bridge")
    parser.add_argument("--config", help="Shared scene / bridge JSON configuration")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token-file", help="Read the bridge token from a file; otherwise UAV_BRIDGE_TOKEN")
    parser.add_argument("--connection-card", help="Save a local phone connection page after startup succeeds")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Explicitly use the non-ROS demonstration vehicle")
    group.add_argument("--adapter", help="Import module:factory; factory(config) returns the vehicle adapter")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    for variable, field in (("UAV_LLM_BASE_URL", "base_url"), ("UAV_LLM_MODEL", "model")):
        if os.environ.get(variable):
            config["llm"][field] = os.environ[variable]
    for variable, field in (("UAV_VLM_BASE_URL", "base_url"), ("UAV_VLM_MODEL", "model"), ("UAV_VLM_PROVIDER", "provider")):
        if os.environ.get(variable):
            config["vision"][field] = os.environ[variable]
    config['mobile_state_directory'] = str(Path(__file__).resolve().parents[2]/'.runtime/mobile')
    config['model_settings_file'] = ('model.gazebo.json' if os.environ.get('VLA_MODE', config.get('mode')) == 'gazebo' else 'model.json')
    saved = MobileState(config['mobile_state_directory']).read(config['model_settings_file'], {})
    if (config['vision'].get('provider') == 'chatgpt_plan' and isinstance(saved, dict)
            and model_name(saved.get('model'))
            and VisionService._chatgpt_store(config['vision']).is_model_available(saved['model'])):
        config['vision']['model'] = saved['model']
    token = os.environ.get("UAV_BRIDGE_TOKEN", "")
    if args.token_file:
        with open(args.token_file, "r", encoding="ascii") as handle:
            token = handle.read().strip()
    if len(token) < 16 or not token.isascii() or token.isspace():
        parser.error("Set UAV_BRIDGE_TOKEN (16+ ASCII characters) or supply --token-file")
    # Install before adapter construction: background Python may inherit SIGINT
    # ignored, while the ROS adapter waits for the planner with signals disabled.
    def stop_initializing(signum, frame):
        raise SystemExit(128 + signum)
    adapter = None
    bridge = None
    server = None
    initializing = True
    initialization_failed = False
    stage = "创建车辆适配器"
    try:
        signal.signal(signal.SIGINT, stop_initializing)
        signal.signal(signal.SIGTERM, stop_initializing)
        if args.mock:
            config["mode"] = "mock"
            adapter = MockAdapter(config)
        else:
            if ":" not in args.adapter:
                parser.error("--adapter must be module:factory")
            module, factory = args.adapter.rsplit(":", 1)
            if config.get("mode") == "mock":
                config["mode"] = "gazebo"
            adapter = getattr(importlib.import_module(module), factory)(config)
        stage = "创建任务服务"
        bridge = Bridge(adapter, config)
        stage = "加载本地检测模型"
        warmup = getattr(bridge.photo.detector, 'warmup', None)
        if config['photo']['enabled'] and warmup:
            started = time.monotonic()
            logging.info("正在初始化本地人物检测模型；机载首次加载可能较慢，完成后开放手机连接。")
            warmup()  # Local model load/zero input only, before accepting tasks.
            logging.info("本地检测模型已就绪，用时 %.1f 秒。", time.monotonic() - started)
        stage = "绑定手机服务端口"
        server = BridgeHTTPServer((args.host, args.port), bridge, token)
        stage = "启动任务服务"
        bridge.start()
        if args.connection_card:
            stage = "生成手机连接页"
            try:
                from scripts.phone_connection import create_connection_card
                card = create_connection_card(server.server_address[1], output=args.connection_card, token=token)
                for address in card['addresses']:
                    logging.info("手机连接地址（%s）：%s", address['interface'], address['url'])
                logging.info("手机连接页已保存：%s；可用文件管理器打开。", card['path'])
                if card['warnings']:
                    logging.warning("本机网络地址采集有提示，请查看手机连接页说明。")
            except Exception:
                # Connection help is optional. Do not include exception text:
                # it may contain the pairing token supplied to the helper.
                logging.warning("手机连接页生成失败，手机服务继续启动。")

        def shutdown(signum, frame):
            logging.info("Stopping bridge; active missions will request hold")
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        initializing = False
        logging.info("手机服务已监听 %s:%d（%s）；App 连接后仍需等待飞行状态就绪。", args.host, server.server_address[1], config.get("mode"))
        server.serve_forever(poll_interval=0.2)
    except BaseException as error:
        if initializing:
            initialization_failed = True
            if isinstance(error, Exception):
                logging.error("手机服务初始化失败（阶段：%s）；请查看下方具体原因，无需因此重新授权 GPT。", stage)
            else:
                logging.info("手机服务初始化已中止（阶段：%s），正在释放已创建的资源。", stage)
        raise
    finally:
        try:
            try:
                if server is not None:
                    server.server_close()
            finally:
                if bridge is not None:
                    bridge.close()
                elif adapter is not None:
                    close = getattr(adapter, "close", None)
                    if close:
                        close()
        except Exception:
            if not initialization_failed:
                raise
            # A not-yet-ready executor can reject hold during cleanup. Keep
            # the original startup failure or cancellation, without logging
            # cleanup exception details that could contain private values.
            logging.warning("初始化资源清理遇到异常；保留原始失败或取消结果。")


if __name__ == "__main__":
    main()
