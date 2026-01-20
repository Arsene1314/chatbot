import argparse
import json
import os
import sys
from typing import List, Dict

from openai import OpenAI

from prompt_builder import load_styles, build_messages


def load_history(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("history 文件必须是 role/content 组成的数组")
    return data


def call_deepseek(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str,
    api_key: str
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def cap_history(history: List[Dict], max_rounds: int) -> List[Dict]:
    if max_rounds <= 0:
        return []
    max_items = max_rounds * 2
    return history[-max_items:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek 语气模仿聊天 CLI")
    parser.add_argument("--tag", default="classmate", help="语气标签: classmate/ambiguous/parent/roommate")
    parser.add_argument("--input", help="用户输入内容（不传则进入交互模式）")
    parser.add_argument("--history", help="历史对话 JSON 文件路径")
    parser.add_argument("--config", default="./config/styles.json", help="风格配置文件路径")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=6, help="交互模式保留的历史轮数")
    return parser.parse_args()


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    styles = load_styles(args.config)

    history: List[Dict] = []
    if args.history:
        history = load_history(args.history)

    def respond(user_input: str, history_state: List[Dict]) -> str:
        messages = build_messages(
            user_input=user_input,
            styles=styles,
            tag_key=args.tag,
            history=history_state,
            max_examples=args.max_examples
        )
        return call_deepseek(
            messages=messages,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=base_url,
            api_key=api_key
        )

    if args.input:
        reply = respond(args.input, history)
        print(reply)
        return

    print("进入交互模式（输入 exit/quit 结束）")
    while True:
        user_input = input("对方：").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = respond(user_input, history)
        print(f"我：{reply}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        history = cap_history(history, args.max_rounds)


if __name__ == "__main__":
    main()
