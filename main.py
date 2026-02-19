"""
DeepSeek 语气模仿聊天 CLI — Many-Shot Prompting 增强版
将全部真实聊天记录嵌入 system prompt，配合详细人格描述，
让 DeepSeek 精准模仿指定人物的说话风格。
"""
import argparse
import os
import sys

from bot_core import load_dotenv, QingqingBot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek 语气模仿聊天 CLI (Many-Shot 增强版)"
    )
    parser.add_argument(
        "--tag", default="ambiguous",
        help="语气标签: classmate/ambiguous/parent/roommate"
    )
    parser.add_argument("--input", help="用户输入内容（不传则进入交互模式）")
    parser.add_argument(
        "--config", default="./config/styles.json", help="风格配置文件路径"
    )
    parser.add_argument(
        "--chat-samples",
        default=None,
        help="真实聊天记录文件路径（默认自动查找）",
    )
    parser.add_argument(
        "--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--max-rounds", type=int, default=8,
        help="交互模式保留的历史轮数"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    bot = QingqingBot(
        config_path=args.config,
        chat_samples_path=args.chat_samples,
        tag=args.tag,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_rounds=args.max_rounds,
    )

    # 单次模式
    if args.input:
        reply = bot.reply(args.input)
        print(reply)
        return

    # 交互模式
    print("\n进入交互模式（输入 exit/quit 结束）")
    print("=" * 40)
    while True:
        try:
            user_input = input("\n对方：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue

        reply = bot.reply(user_input)
        for line in reply.split("\n"):
            line = line.strip()
            if line:
                print(f"晴晴：{line}")

    print("\n再见！")


if __name__ == "__main__":
    main()
