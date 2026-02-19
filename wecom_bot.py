"""
企业微信机器人 — Flask 回调服务

功能:
  1. 响应企业微信 URL 验证（GET 请求）
  2. 接收用户消息（POST 请求） -> 调用 DeepSeek API -> 主动发送回复

使用方式:
  python3 wecom_bot.py                      # 默认端口 5000
  python3 wecom_bot.py --port 8080          # 自定义端口
  python3 wecom_bot.py --port 8080 --ngrok  # 自动启动 ngrok 隧道

企业微信后台配置:
  回调 URL: http://你的公网地址:端口/wecom/callback
  Token: 与 .env 中 WECOM_TOKEN 一致
  EncodingAESKey: 与 .env 中 WECOM_ENCODING_AES_KEY 一致
"""
import argparse
import os
import sys
import threading
import time

import requests
from flask import Flask, request, abort, make_response

from bot_core import load_dotenv, QingqingBot
from wecom_crypto import WeComCrypto, parse_text_message

# ─── 初始化 ────────────────────────────────────────────────────

load_dotenv()

CORP_ID = os.getenv("WECOM_CORP_ID", "")
CORP_SECRET = os.getenv("WECOM_CORP_SECRET", "")
AGENT_ID = os.getenv("WECOM_AGENT_ID", "")
TOKEN = os.getenv("WECOM_TOKEN", "")
ENCODING_AES_KEY = os.getenv("WECOM_ENCODING_AES_KEY", "")

app = Flask(__name__)
crypto: WeComCrypto = None
bot: QingqingBot = None

# Access token 缓存
_access_token = ""
_token_expires_at = 0


# ─── 企业微信 API ──────────────────────────────────────────────


def get_access_token() -> str:
    """获取企业微信 access_token（带缓存）"""
    global _access_token, _token_expires_at
    now = time.time()
    if _access_token and now < _token_expires_at - 60:
        return _access_token

    url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
    resp = requests.get(url, params={
        "corpid": CORP_ID,
        "corpsecret": CORP_SECRET,
    }, timeout=10)
    data = resp.json()

    if data.get("errcode", 0) != 0:
        print(f"[wecom] 获取 access_token 失败: {data}", file=sys.stderr)
        return ""

    _access_token = data["access_token"]
    _token_expires_at = now + data.get("expires_in", 7200)
    print(f"[wecom] access_token 已刷新，有效期 {data.get('expires_in', 7200)}s")
    return _access_token


def send_text_message(user_id: str, content: str) -> dict:
    """通过企业微信 API 主动发送文本消息"""
    token = get_access_token()
    if not token:
        return {"errcode": -1, "errmsg": "no access_token"}

    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    payload = {
        "touser": user_id,
        "msgtype": "text",
        "agentid": int(AGENT_ID) if AGENT_ID else 0,
        "text": {"content": content},
    }
    resp = requests.post(url, json=payload, timeout=10)
    result = resp.json()
    if result.get("errcode", 0) != 0:
        print(f"[wecom] 发送消息失败: {result}", file=sys.stderr)
    return result


# ─── Flask 路由 ────────────────────────────────────────────────


@app.route("/wecom/callback", methods=["GET"])
def verify_url():
    """
    企业微信 URL 验证。
    管理后台配置回调时，企业微信服务器会发 GET 请求验证。
    需要解密 echostr 并原样返回。
    """
    msg_signature = request.args.get("msg_signature", "")
    timestamp = request.args.get("timestamp", "")
    nonce = request.args.get("nonce", "")
    echostr = request.args.get("echostr", "")

    if not all([msg_signature, timestamp, nonce, echostr]):
        abort(400, "缺少参数")

    try:
        # 验证签名并解密 echostr
        if not crypto.verify_signature(msg_signature, timestamp, nonce, echostr):
            abort(403, "签名验证失败")
        decrypted = crypto.decrypt(echostr)
        print(f"[wecom] URL 验证成功")
        # 必须返回 text/html，不能带引号、BOM头、换行符
        resp = make_response(decrypted)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    except Exception as e:
        print(f"[wecom] URL 验证失败: {e}", file=sys.stderr)
        abort(403, str(e))


@app.route("/wecom/callback", methods=["POST"])
def receive_message():
    """
    接收企业微信消息。
    企业微信推送消息时发 POST 请求，body 是加密的 XML。
    我们解密后提取内容，异步调用 DeepSeek 生成回复，再主动发送。
    """
    msg_signature = request.args.get("msg_signature", "")
    timestamp = request.args.get("timestamp", "")
    nonce = request.args.get("nonce", "")
    post_body = request.data.decode("utf-8")

    if not all([msg_signature, timestamp, nonce, post_body]):
        abort(400, "缺少参数")

    try:
        # 解密消息
        plain_xml = crypto.decrypt_message(
            post_body, msg_signature, timestamp, nonce
        )
        msg = parse_text_message(plain_xml)
    except Exception as e:
        print(f"[wecom] 消息解密失败: {e}", file=sys.stderr)
        abort(403, str(e))

    msg_type = msg.get("msg_type", "")
    content = msg.get("content", "")
    from_user = msg.get("from_user", "")

    print(f"[wecom] 收到消息 [{msg_type}] from {from_user}: {content}")

    if msg_type != "text" or not content.strip():
        # 非文本消息，返回空响应（企业微信要求 5 秒内响应）
        return "success"

    # 特殊指令
    if content.strip().lower() in {"清除记录", "reset", "清空"}:
        bot.clear_history(from_user)
        send_text_message(from_user, "记忆已清除~")
        return "success"

    # 异步处理：先响应企业微信（避免 5 秒超时），再异步生成回复
    def async_reply():
        try:
            reply = bot.reply(content, user_id=from_user)
            # 模拟微信多条消息：每行单独发送
            lines = [l.strip() for l in reply.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                send_text_message(from_user, line)
                if i < len(lines) - 1:
                    time.sleep(0.5)  # 模拟打字间隔
            print(f"[wecom] 回复 {from_user}: {reply}")
        except Exception as e:
            print(f"[wecom] 生成回复失败: {e}", file=sys.stderr)
            send_text_message(from_user, "emmm 我脑子卡了一下[捂脸]")

    thread = threading.Thread(target=async_reply, daemon=True)
    thread.start()

    return "success"


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return {"status": "ok", "bot": "晴晴"}


# ─── 启动 ──────────────────────────────────────────────────────


def validate_config():
    """启动前检查必要配置"""
    missing = []
    if not CORP_ID:
        missing.append("WECOM_CORP_ID")
    if not CORP_SECRET:
        missing.append("WECOM_CORP_SECRET")
    if not AGENT_ID:
        missing.append("WECOM_AGENT_ID")
    if not TOKEN:
        missing.append("WECOM_TOKEN")
    if not ENCODING_AES_KEY:
        missing.append("WECOM_ENCODING_AES_KEY")
    if not os.getenv("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")

    if missing:
        print("=" * 50)
        print("缺少以下配置（请在 .env 文件中设置）：")
        for key in missing:
            print(f"  - {key}")
        print()
        print("配置说明：")
        print("  1. 登录企业微信管理后台: https://work.weixin.qq.com")
        print("  2. 创建自建应用，获取 CORP_ID、CORP_SECRET、AGENT_ID")
        print("  3. 在应用的「接收消息」中配置回调 URL、Token、EncodingAESKey")
        print("  4. 将这些值写入 .env 文件")
        print("=" * 50)
        sys.exit(1)


def main():
    global crypto, bot

    parser = argparse.ArgumentParser(description="晴晴企业微信机器人")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument(
        "--config", default="./config/styles.json", help="风格配置文件路径"
    )
    parser.add_argument("--chat-samples", default=None, help="聊天样本文件路径")
    parser.add_argument("--debug", action="store_true", help="Flask debug 模式")
    args = parser.parse_args()

    validate_config()

    # 初始化加解密
    crypto = WeComCrypto(TOKEN, ENCODING_AES_KEY, CORP_ID)
    print(f"[wecom] 加解密模块初始化完成 (CorpID: {CORP_ID})")

    # 初始化机器人
    bot = QingqingBot(
        config_path=args.config,
        chat_samples_path=args.chat_samples,
    )
    print(f"[wecom] 晴晴机器人初始化完成")

    # 预热 access_token
    get_access_token()

    print()
    print("=" * 50)
    print(f"  晴晴企业微信机器人已启动")
    print(f"  回调 URL: http://YOUR_HOST:{args.port}/wecom/callback")
    print(f"  健康检查: http://localhost:{args.port}/health")
    print("=" * 50)
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
