#!/usr/bin/env python3
"""与 fine-tuned Joker 模型交互式聊天"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("fine_tuned_model.txt") as f:
    MODEL = f.read().strip()

with open("training_data/openai-finetune.jsonl") as f:
    first = json.loads(f.readline())
    SYSTEM_PROMPT = next(m["content"] for m in first["messages"] if m["role"] == "system")

print(f"模型: {MODEL}")
print(f"输入 'quit' 退出，'clear' 清空对话历史\n")
print("=" * 50)

history = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    try:
        user_input = input("\n你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("再见！")
        break
    if user_input.lower() == "clear":
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("[对话历史已清空]")
        continue

    history.append({"role": "user", "content": user_input})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=history,
            temperature=0.9,
            max_tokens=512,
        )
        reply = resp.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        print(f"\nJoker: {reply}")
    except Exception as e:
        print(f"\n[错误] {e}")
        history.pop()
