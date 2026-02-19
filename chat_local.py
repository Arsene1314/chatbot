#!/usr/bin/env python3
"""
本地交互聊天客户端 - 通过 SSH 连接 AutoDL 上的 Joker 模型
用法: python chat_local.py
"""
import subprocess, json, base64, sys, os

SSH_CMD = "ssh -p 57584 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com"
REMOTE_PYTHON = "/root/miniconda3/bin/python"
REMOTE_SCRIPT = "/root/chat_server.py"

def main():
    print("=" * 50)
    print("  Joker Chat (Qwen2.5-14B + LoRA)")
    print("=" * 50)
    print("\n输入消息开始聊天，输入 q 退出")
    print("正在连接服务器并加载模型，请稍等...")

    cmd = f"{SSH_CMD} {REMOTE_PYTHON} -u {REMOTE_SCRIPT}"
    proc = subprocess.Popen(
        cmd, shell=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    while True:
        line = proc.stdout.readline().decode("utf-8").strip()
        if "MODEL_READY" in line:
            break
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8")
            print(f"启动失败: {err}")
            return

    print("模型加载完成！开始聊天吧~\n")
    history = []

    try:
        while True:
            user_input = input("\033[36m你: \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ("q", "quit", "exit"):
                break

            history.append({"role": "user", "content": user_input})
            payload = base64.b64encode(json.dumps(history, ensure_ascii=False).encode("utf-8")).decode("ascii")
            proc.stdin.write((payload + "\n").encode("utf-8"))
            proc.stdin.flush()

            while True:
                line = proc.stdout.readline().decode("utf-8").strip()
                if line.startswith("REPLY:"):
                    encoded = line[6:]
                    reply = base64.b64decode(encoded).decode("utf-8")
                    break

            history.append({"role": "assistant", "content": reply})
            print(f"\033[33mJoker: {reply}\033[0m\n")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        proc.stdin.write(b"EXIT\n")
        proc.stdin.flush()
        proc.terminate()
        print("\n再见！")

if __name__ == "__main__":
    main()
