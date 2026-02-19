"""
将 ShareGPT 格式的训练数据转换为 OpenAI fine-tuning JSONL 格式。
精选均衡、高质量的样本。

OpenAI 格式:
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
"""
import json
import random
import sys
import tiktoken
from collections import Counter

random.seed(42)

INPUT = "./training_data/sft-joker-clean.json"
OUTPUT = "./training_data/openai-finetune.jsonl"
SAMPLES_PER_TYPE = 60
MAX_TOKENS_PER_EXAMPLE = 4096


def sharegpt_to_openai(conv: dict) -> dict:
    """ShareGPT → OpenAI messages 格式"""
    messages = []
    for msg in conv["conversations"]:
        if msg["from"] == "system":
            messages.append({"role": "system", "content": msg["value"]})
        elif msg["from"] == "human":
            messages.append({"role": "user", "content": msg["value"]})
        elif msg["from"] == "gpt":
            messages.append({"role": "assistant", "content": msg["value"]})
    # OpenAI 要求最后一条必须是 assistant
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return {"messages": messages}


def count_tokens(messages: list, encoding) -> int:
    total = 0
    for msg in messages:
        total += 4  # <im_start>, role, \n, <im_end>
        total += len(encoding.encode(msg["content"]))
    total += 2  # <im_start>assistant prefix
    return total


def quality_score(conv: dict) -> float:
    """简单评分：优先选多轮、长度适中、有实质内容的对话"""
    msgs = conv["conversations"]
    gpt_msgs = [m for m in msgs if m["from"] == "gpt"]
    human_msgs = [m for m in msgs if m["from"] == "human"]

    n_turns = len(gpt_msgs)
    avg_len = sum(len(m["value"]) for m in gpt_msgs) / max(n_turns, 1)

    score = 0.0
    # 3-8 轮最佳
    if 3 <= n_turns <= 8:
        score += 2.0
    elif n_turns <= 2:
        score += 1.0  # 短对话也要，但权重低一些
    else:
        score += 1.5

    # 回复平均长度 10-80 字最自然
    if 10 <= avg_len <= 80:
        score += 2.0
    elif avg_len < 10:
        score += 0.5
    else:
        score += 1.0

    # 有实际内容的加分
    all_gpt = " ".join(m["value"] for m in gpt_msgs)
    if any(w in all_gpt for w in ["MF", "数学", "INFP", "写歌", "看番", "炒股", "骑车"]):
        score += 1.0

    # 有自然口语化表达的加分
    if any(w in all_gpt for w in ["素", "笑死", "我勒个豆", "emmmm", "好好好", "🉑"]):
        score += 1.0

    return score


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"原始数据: {len(data)} 条")

    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    # 按类型分组
    by_type: dict[str, list] = {}
    for conv in data:
        style = conv.get("style", "default")
        by_type.setdefault(style, []).append(conv)

    print("\n各类型数量:")
    for k, v in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {k}: {len(v)}")

    # 每种类型按质量评分排序，选 top N
    selected = []
    for style, convs in by_type.items():
        scored = [(quality_score(c), random.random(), c) for c in convs]
        scored.sort(key=lambda x: (-x[0], x[1]))

        count = 0
        for _, _, conv in scored:
            if count >= SAMPLES_PER_TYPE:
                break
            openai_fmt = sharegpt_to_openai(conv)
            tokens = count_tokens(openai_fmt["messages"], encoding)
            if tokens <= MAX_TOKENS_PER_EXAMPLE:
                selected.append(openai_fmt)
                count += 1
        print(f"  {style}: 选取 {count} 条")

    random.shuffle(selected)

    # 统计 token
    total_tokens = sum(count_tokens(s["messages"], encoding) for s in selected)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n输出: {OUTPUT}")
    print(f"总条数: {len(selected)}")
    print(f"总 token: {total_tokens:,}")
    print(f"预估训练费用 (GPT-4o-mini): ~${total_tokens * 8 / 1_000_000:.2f}")
    print(f"预估训练费用 (GPT-4o):      ~${total_tokens * 25 / 1_000_000:.2f}")

    # 验证分布
    role_dist = Counter()
    for item in selected:
        for msg in item["messages"]:
            if msg["role"] == "system":
                content = msg["content"]
                if "兄弟" in content and "暗恋" not in content and "前任" not in content:
                    role_dist["brother"] += 1
                elif "暗恋" in content:
                    role_dist["crush"] += 1
                elif "前任" in content:
                    role_dist["ex"] += 1
                elif "女生朋友" in content:
                    role_dist["female_friend"] += 1
                else:
                    role_dist["default"] += 1
                break

    print(f"\n关系类型分布: {dict(role_dist)}")


if __name__ == "__main__":
    main()
