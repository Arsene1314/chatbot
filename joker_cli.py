"""
Joker 数字分身 CLI — 多风格聊天测试
用法:
  python joker_cli.py                        # 默认 default 风格
  python joker_cli.py --style brother        # 兄弟风格
  python joker_cli.py --style crush          # 追求对象风格
  python joker_cli.py --style female_friend  # 女生朋友风格
  python joker_cli.py --style ex             # 前任/亲密异性风格

交互中可以输入:
  /style brother    切换风格
  /clear            清空对话历史
  /debug            查看当前 system prompt
  exit / quit / q   退出
"""
import argparse
import os
import sys

from bot_core import load_dotenv, JokerBot


VALID_STYLES = ["brother", "female_friend", "crush", "ex", "default"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joker 数字分身 CLI（多风格聊天测试）"
    )
    parser.add_argument(
        "--style", default="default", choices=VALID_STYLES,
        help="对话风格: brother/female_friend/crush/ex/default"
    )
    parser.add_argument(
        "--profile-dir", default="./joker_profile",
        help="人格资料目录"
    )
    parser.add_argument(
        "--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument(
        "--max-rounds", type=int, default=10,
        help="保留的历史轮数"
    )
    parser.add_argument("--input", help="单次输入（不传则进入交互模式）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    bot = JokerBot(
        style_tag=args.style,
        profile_dir=args.profile_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_rounds=args.max_rounds,
    )

    if args.input:
        reply = bot.reply(args.input)
        print(reply)
        return

    print(f"\n进入 Joker 交互模式 | 当前风格: {args.style}")
    print(f"  /style <tag>  切换风格 ({', '.join(VALID_STYLES)})")
    print(f"  /clear        清空对话历史")
    print(f"  /debug        查看 system prompt")
    print(f"  exit/quit/q   退出")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n对方：").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/style"):
            parts = user_input.split()
            if len(parts) >= 2 and parts[1] in VALID_STYLES:
                bot.switch_style(parts[1])
                bot.clear_history("cli")
                print(f"  → 已切换至 {parts[1]} 风格，对话已清空")
            else:
                print(f"  → 可选风格: {', '.join(VALID_STYLES)}")
            continue

        if user_input == "/clear":
            bot.clear_history("cli")
            print("  → 对话历史已清空")
            continue

        if user_input == "/debug":
            from joker_prompt_builder import build_joker_system_prompt
            examples = bot._examples_cache.get(bot.style_tag, "")
            prompt = build_joker_system_prompt(bot.style_tag, examples)
            print(f"\n{'='*50}")
            print(f"System Prompt ({len(prompt)} 字):")
            print(f"{'='*50}")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print(f"\n... (共 {len(prompt)} 字, 仅显示前 2000)")
            continue

        reply = bot.reply(user_input, user_id="cli")
        for line in reply.split("\n"):
            line = line.strip()
            if line:
                print(f"Joker：{line}")

    print("\n再见！")


if __name__ == "__main__":
    main()
