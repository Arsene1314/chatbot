"""
Phase 2: 合并真实 + 合成训练数据为最终 SFT 数据集。

输入：
  - training_data/sft-joker-chat.json     (真实聊天 + transcript)
  - training_data/sft-joker-synthetic.json (合成数据)

输出：
  - training_data/sft-joker-final.json    (合并 + 去重 + 打乱)

用法：
  python merge_sft_data.py
"""
import json
import os
import random
import sys


INPUT_FILES = [
    "./training_data/sft-joker-chat.json",
    "./training_data/sft-joker-synthetic.json",
    "./training_data/sft-joker-synthetic-large.json",
]
OUTPUT_FILE = "./training_data/sft-joker-final.json"


def load_json(path: str) -> list:
    if not os.path.exists(path):
        print(f"[跳过] 文件不存在: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_entry(entry: dict) -> bool:
    """验证一条训练样本的基本格式"""
    if "conversations" not in entry:
        return False
    conv = entry["conversations"]
    if not isinstance(conv, list) or len(conv) < 2:
        return False
    has_human = any(m.get("from") == "human" for m in conv)
    has_gpt = any(m.get("from") == "gpt" for m in conv)
    return has_human and has_gpt


def main():
    all_data = []

    for path in INPUT_FILES:
        data = load_json(path)
        valid = [d for d in data if validate_entry(d)]
        invalid = len(data) - len(valid)
        all_data.extend(valid)
        print(f"[加载] {path}: {len(valid)} 条有效" + (f" ({invalid} 条无效已过滤)" if invalid else ""))

    if not all_data:
        print("没有可用的训练数据!", file=sys.stderr)
        sys.exit(1)

    # 打乱顺序
    random.seed(42)
    random.shuffle(all_data)

    # 统计
    style_counts = {}
    source_counts = {}
    for item in all_data:
        s = item.get("style", "unknown")
        src = item.get("source", "unknown")
        style_counts[s] = style_counts.get(s, 0) + 1
        src_key = src[:30] if len(src) > 30 else src
        source_counts[src_key] = source_counts.get(src_key, 0) + 1

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n合并完毕!")
    print(f"总计: {len(all_data)} 条")
    print(f"输出: {OUTPUT_FILE}")
    print(f"\n风格分布: {json.dumps(style_counts, ensure_ascii=False)}")
    print(f"来源分布: {json.dumps(source_counts, ensure_ascii=False, indent=2)}")

    # 计算平均对话长度
    total_turns = sum(len(d["conversations"]) for d in all_data)
    avg_turns = total_turns / len(all_data) if all_data else 0
    print(f"平均对话轮次: {avg_turns:.1f}")


if __name__ == "__main__":
    main()
