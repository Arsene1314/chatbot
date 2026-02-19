"""
清洗训练数据：修复匿名化问题。

1. 修复 system prompt 中未匿名的学校名（滑铁卢→[学校]）
2. 清除 GPT 回复中的城市泄露（多伦多→[城市]）
"""
import json
import re
from collections import Counter

INPUT = "./training_data/sft-joker-safe.json"
OUTPUT = "./training_data/sft-joker-clean.json"


def fix_anonymization(conv: dict) -> bool:
    """修复匿名化问题，返回是否修改过"""
    changed = False
    for msg in conv["conversations"]:
        old = msg["value"]
        if msg["from"] == "system":
            msg["value"] = msg["value"].replace("滑铁卢大学", "[学校]")
            msg["value"] = msg["value"].replace("滑铁卢", "[学校]")
        if msg["from"] == "gpt":
            msg["value"] = re.sub(r"多伦多", "[城市]", msg["value"])
            msg["value"] = re.sub(r"滑铁卢", "[学校所在地]", msg["value"])
        if msg["value"] != old:
            changed = True
    return changed


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"原始数据: {len(data)} 条")

    dist = Counter(c.get("style", "unknown") for c in data)
    print("\n分布:")
    for k, v in dist.most_common():
        print(f"  {k}: {v} ({v/len(data)*100:.1f}%)")

    fixed = sum(1 for conv in data if fix_anonymization(conv))
    print(f"\n修复匿名化: {fixed} 条")

    # 验证
    anon_issue = sum(1 for c in data for m in c["conversations"]
                     if m["from"] == "system" and "滑铁卢" in m["value"])
    city_leak = sum(1 for c in data for m in c["conversations"]
                    if m["from"] == "gpt" and "多伦多" in m["value"])
    print(f"残留问题: 未匿名={anon_issue}, 城市泄露={city_leak}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n输出: {OUTPUT}")


if __name__ == "__main__":
    main()
