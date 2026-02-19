"""
微信公众号测试号机器人 — Flask 回调服务

功能:
  1. 响应微信 URL 验证（GET 请求）
  2. 接收用户文本消息 -> 调用 DeepSeek API -> 客服消息逐条回复
  3. 收到图片/表情包/语音 -> 自然回应

使用方式:
  python3 mp_bot.py                      # 默认端口 8080
  python3 mp_bot.py --port 9000          # 自定义端口

公众号测试号配置:
  URL: http://你的公网地址:端口/wx/callback
  Token: 与 .env 中 MP_TOKEN 一致
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import xml.etree.ElementTree as ET

import requests
from flask import Flask, request, make_response

from bot_core import load_dotenv, QingqingBot

# ─── 初始化 ────────────────────────────────────────────────────

load_dotenv()

MP_TOKEN = os.getenv("MP_TOKEN", "qingqing_bot_token")
MP_APP_ID = os.getenv("MP_APP_ID", "")
MP_APP_SECRET = os.getenv("MP_APP_SECRET", "")

app = Flask(__name__)
bot: QingqingBot = None

# Access token 缓存
_access_token = ""
_token_expires_at = 0

# 每个用户一把锁，保证同一用户的消息按顺序处理，避免并发导致上下文矛盾
_user_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def get_user_lock(user_id: str) -> threading.Lock:
    """获取指定用户的处理锁（线程安全）"""
    with _locks_lock:
        if user_id not in _user_locks:
            _user_locks[user_id] = threading.Lock()
        return _user_locks[user_id]


# 收到图片/表情包时的随机回应（像真人一样反应）
STICKER_REPLIES = [
    "哈哈哈哈哈哈",
    "哈哈哈哈哈哈哈",
    "笑死",
    "哈哈哈",
    "嘿嘿",
    "哈哈哈哈哈好好笑",
]


# ─── 微信 API ─────────────────────────────────────────────────


def get_access_token() -> str:
    """获取公众号 access_token（带缓存）"""
    global _access_token, _token_expires_at
    now = time.time()
    if _access_token and now < _token_expires_at - 60:
        return _access_token

    url = "https://api.weixin.qq.com/cgi-bin/token"
    resp = requests.get(url, params={
        "grant_type": "client_credential",
        "appid": MP_APP_ID,
        "secret": MP_APP_SECRET,
    }, timeout=10)
    data = resp.json()

    if "access_token" not in data:
        print(f"[mp] 获取 access_token 失败: {data}", file=sys.stderr)
        return ""

    _access_token = data["access_token"]
    _token_expires_at = now + data.get("expires_in", 7200)
    print(f"[mp] access_token 已刷新")
    return _access_token


def send_custom_message(to_user: str, content: str) -> bool:
    """通过客服消息接口发送文本消息"""
    token = get_access_token()
    if not token:
        print(f"[mp] 客服消息失败: 无 access_token", file=sys.stderr)
        return False

    url = f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={token}"
    payload = {
        "touser": to_user,
        "msgtype": "text",
        "text": {"content": content},
    }
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        resp = requests.post(
            url, data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=10,
        )
        result = resp.json()
        if result.get("errcode", 0) != 0:
            print(f"[mp] 客服消息失败: {result}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[mp] 客服消息异常: {e}", file=sys.stderr)
        return False


# ─── 微信签名验证 ──────────────────────────────────────────────


def check_signature(signature: str, timestamp: str, nonce: str) -> bool:
    """验证消息来自微信服务器"""
    items = sorted([MP_TOKEN, timestamp, nonce])
    sha1 = hashlib.sha1("".join(items).encode("utf-8")).hexdigest()
    return sha1 == signature


# ─── XML 解析/构造 ─────────────────────────────────────────────


def parse_xml_message(xml_str: str) -> dict:
    """解析微信推送的 XML 消息"""
    root = ET.fromstring(xml_str)
    return {
        "to_user": (root.findtext("ToUserName") or "").strip(),
        "from_user": (root.findtext("FromUserName") or "").strip(),
        "create_time": (root.findtext("CreateTime") or "").strip(),
        "msg_type": (root.findtext("MsgType") or "").strip(),
        "content": (root.findtext("Content") or "").strip(),
        "msg_id": (root.findtext("MsgId") or "").strip(),
    }


def build_text_reply(from_user: str, to_user: str, content: str) -> str:
    """构造被动回复 XML（兜底用）"""
    timestamp = str(int(time.time()))
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{timestamp}</CreateTime>"
        f"<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content>"
        "</xml>"
    )


def make_xml_response(xml_str: str):
    """返回 XML 响应"""
    resp = make_response(xml_str)
    resp.headers["Content-Type"] = "application/xml; charset=utf-8"
    return resp


# ─── Flask 路由 ────────────────────────────────────────────────


@app.route("/wx/callback", methods=["GET"])
def verify_url():
    """微信 URL 验证"""
    signature = request.args.get("signature", "")
    timestamp = request.args.get("timestamp", "")
    nonce = request.args.get("nonce", "")
    echostr = request.args.get("echostr", "")

    if check_signature(signature, timestamp, nonce):
        print(f"[mp] URL 验证成功")
        resp = make_response(echostr)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    else:
        print(f"[mp] URL 验证失败: 签名不匹配")
        return "signature mismatch", 403


@app.route("/wx/callback", methods=["POST"])
def receive_message():
    """
    接收微信消息。
    先返回 success（避免 5 秒超时重试），然后异步用客服消息接口逐条发送。
    如果客服接口失败，用被动回复兜底。
    """
    signature = request.args.get("signature", "")
    timestamp = request.args.get("timestamp", "")
    nonce = request.args.get("nonce", "")
    if not check_signature(signature, timestamp, nonce):
        return "signature mismatch", 403

    xml_data = request.data.decode("utf-8")
    msg = parse_xml_message(xml_data)

    msg_type = msg.get("msg_type", "")
    content = msg.get("content", "")
    from_user = msg.get("from_user", "")
    to_user = msg.get("to_user", "")

    print(f"[mp] 收到消息 [{msg_type}] from {from_user}: {content[:50]}", flush=True)

    # 非文本消息（图片/表情包/语音/视频等）→ 被动回复
    if msg_type != "text":
        reply_text = random.choice(STICKER_REPLIES)
        print(f"[mp] 非文本消息 [{msg_type}]，回复: {reply_text}", flush=True)
        return make_xml_response(
            build_text_reply(to_user, from_user, reply_text)
        )

    if not content.strip():
        return "success"

    # 表情包/不支持的消息类型（微信可能发英文或中文提示）
    unsupported_markers = [
        "[Unsupported Message]",
        "[收到不支持的消息类型",
        "暂无法显示",
    ]
    if any(marker in content.strip() for marker in unsupported_markers):
        reply_text = random.choice(STICKER_REPLIES)
        print(f"[mp] 不支持的消息，回复: {reply_text}", flush=True)
        return make_xml_response(
            build_text_reply(to_user, from_user, reply_text)
        )

    # 特殊指令 → 被动回复
    if content.strip().lower() in {"清除记录", "reset", "清空"}:
        bot.clear_history(from_user)
        return make_xml_response(
            build_text_reply(to_user, from_user, "记忆已清除~")
        )

    # 异步调 DeepSeek + 客服消息逐条发送（同一用户排队处理）
    def async_reply():
        user_lock = get_user_lock(from_user)
        with user_lock:
            try:
                reply = bot.reply(content, user_id=from_user)
                lines = [l.strip() for l in reply.split("\n") if l.strip()]

                for i, line in enumerate(lines):
                    ok = send_custom_message(from_user, line)
                    if not ok:
                        remaining = "\n".join(lines[i:])
                        send_custom_message(from_user, remaining)
                        break
                    if i < len(lines) - 1:
                        time.sleep(0.6)

                print(f"[mp] 回复 {from_user} ({len(lines)}条): {reply}", flush=True)
            except Exception as e:
                print(f"[mp] 生成回复失败: {e}", file=sys.stderr, flush=True)
                send_custom_message(from_user, "emmm 我脑子卡了一下")

    threading.Thread(target=async_reply, daemon=True).start()

    # 先返回空响应，避免微信超时重试
    return "success"


@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok", "bot": "晴晴", "platform": "mp_test"}


# ─── 压力测试 / 直接调用接口 ──────────────────────────────────


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    直接聊天接口（不走微信），用于脚本测试 / 压力测试。

    请求:
      POST /api/chat
      Content-Type: application/json
      {
        "message": "你好呀",
        "user_id": "test_user_001"   // 可选，默认 "test"
      }

    响应:
      {
        "reply": "干嘛呀",
        "lines": ["干嘛呀"],
        "user_id": "test_user_001",
        "latency_ms": 1234
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    user_id = data.get("user_id", "test")

    if not message:
        return {"error": "message 不能为空"}, 400

    print(f"[api] 收到测试消息 from {user_id}: {message[:50]}", flush=True)

    t0 = time.time()
    user_lock = get_user_lock(user_id)
    with user_lock:
        reply = bot.reply(message, user_id=user_id)
    latency_ms = int((time.time() - t0) * 1000)

    lines = [l.strip() for l in reply.split("\n") if l.strip()]

    print(f"[api] 回复 {user_id} ({latency_ms}ms): {reply}", flush=True)

    return {
        "reply": reply,
        "lines": lines,
        "user_id": user_id,
        "latency_ms": latency_ms,
    }


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """清除指定用户的对话历史"""
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id", "test")
    bot.clear_history(user_id)
    return {"status": "ok", "user_id": user_id, "message": "历史已清除"}


# ─── 启动 ──────────────────────────────────────────────────────


def main():
    global bot

    parser = argparse.ArgumentParser(description="晴晴微信公众号机器人")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument(
        "--config", default="./config/styles.json", help="风格配置文件路径"
    )
    parser.add_argument("--chat-samples", default=None, help="聊天样本文件路径")
    parser.add_argument("--debug", action="store_true", help="Flask debug 模式")
    args = parser.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    if not MP_APP_ID or not MP_APP_SECRET:
        print("缺少 MP_APP_ID 或 MP_APP_SECRET 环境变量", file=sys.stderr)
        sys.exit(1)

    # 初始化机器人
    bot = QingqingBot(
        config_path=args.config,
        chat_samples_path=args.chat_samples,
    )
    print(f"[mp] 晴晴机器人初始化完成")

    # 预热 access_token
    get_access_token()

    print()
    print("=" * 50)
    print(f"  晴晴公众号机器人已启动（客服消息模式）")
    print(f"  回调 URL: http://YOUR_HOST:{args.port}/wx/callback")
    print(f"  健康检查: http://localhost:{args.port}/health")
    print(f"  测试接口: POST http://localhost:{args.port}/api/chat")
    print(f"  清除历史: POST http://localhost:{args.port}/api/clear")
    print("=" * 50)
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
