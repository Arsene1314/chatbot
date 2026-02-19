"""
训练数据脱敏模块 — 在数据进入 SFT 训练前移除所有可识别个人信息（PII）。

三层处理：
1. 名字/地点/敏感信息 → 通用占位符替换
2. 敏感段落整体删除（家庭冲突细节、健康诊断等）
3. 生成隐私拒答训练样本

用法：
  python anonymize.py                          # 脱敏 + 生成拒答样本
  python anonymize.py --input path/to/data.json
"""
import json
import os
import re
import sys
from typing import Dict, List


# ── 第一层：名字/地点/敏感词替换 ──────────────────────────────────

# 格式: (pattern, replacement)
# 用正则确保能匹配各种写法
PII_REPLACEMENTS = [
    # 人名 — 按长度从长到短排列避免部分匹配
    (r"陈雪晴", "[某个女生]"),
    (r"cxq\s*\(一只萌晴晴\)", "[某个女生]"),
    (r"一只萌晴晴", "[某个女生]"),
    (r"(?<![「])晴晴(?!」)", "[某个女生]"),  # 避免替换晴晴bot的引用
    (r"沈敏讷", "[前任]"),
    (r"ᐛ\s*\(沈敏讷minne\)", "[前任]"),
    (r"[Mm]inne", "[前任]"),
    (r"insomnia\s*\([Dd]oris\)", "[朋友A]"),
    (r"[Dd]oris", "[朋友A]"),
    (r"insomnia", "[朋友A]"),
    (r"Tangent\s*\([Rr]yan\)", "[朋友B]"),
    (r"[Rr]yan", "[朋友B]"),
    (r"Tangent", "[朋友B]"),
    (r"柳子坤", "[室友]"),
    (r"Cloud\s*9\s*\(室友柳子坤\)", "[室友]"),
    (r"Cloud\s*9", "[室友]"),
    (r"[Aa]nna", "[朋友C]"),
    (r"海底城的她", "[另一个女生]"),
    (r"海底城", "[另一个地方]"),

    # 城市
    (r"上海光源", "[实验室]"),
    (r"上海", "[某城市]"),

    # 学校/地点
    (r"滑铁卢大学", "[学校]"),
    (r"滑铁卢", "[学校所在地]"),
    (r"University of Waterloo", "[学校]"),
    (r"UWaterloo", "[学校]"),
    (r"[Ww]aterloo", "[学校所在地]"),
    (r"uwaterloo", "[学校]"),

    # 财务具体数字
    (r"[12]000多?[万w]", "[一大笔钱]"),
    (r"2[千k]w", "[一大笔钱]"),
    (r"2千万", "[一大笔钱]"),

    # 微信号/电话等（通用模式）
    (r"\b1[3-9]\d{9}\b", "[手机号]"),
]

# ── 第二层：敏感段落检测 & 泛化 ───────────────────────────────────

# 如果一段文本同时包含以下关键词组合，进行泛化处理
SENSITIVE_CONTEXTS = [
    {
        "triggers": ["看了医生", "躯体", "抑郁"],
        "replacement": "有时候状态不太好，但还行",
    },
    {
        "triggers": ["禁止", "提到", "生病"],
        "replacement": "有些事不太想聊",
    },
    {
        "triggers": ["无法呼吸", "过度焦虑"],
        "replacement": "压力大的时候会不太舒服",
    },
    {
        "triggers": ["到一边去死"],
        "replacement": "室友说了一些很过分的话",
    },
]

# 完全删除包含这些内容的训练样本
BLACKLIST_PHRASES = [
    "自杀",
    "想死",
    "跳楼",
    "割腕",
]


def anonymize_text(text: str) -> str:
    """对一段文本执行 PII 替换"""
    result = text
    for pattern, replacement in PII_REPLACEMENTS:
        result = re.sub(pattern, replacement, result)

    # 敏感上下文泛化
    for ctx in SENSITIVE_CONTEXTS:
        if all(t in result for t in ctx["triggers"]):
            result = ctx["replacement"]

    return result


def should_blacklist(text: str) -> bool:
    """检查是否包含需要完全移除的内容"""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in BLACKLIST_PHRASES)


def anonymize_conversation(conv: List[Dict]) -> List[Dict]:
    """脱敏一条完整对话"""
    result = []
    for msg in conv:
        new_msg = dict(msg)
        new_msg["value"] = anonymize_text(msg.get("value", ""))
        result.append(new_msg)
    return result


def anonymize_dataset(data: List[Dict]) -> List[Dict]:
    """脱敏整个数据集"""
    results = []
    removed = 0

    for item in data:
        conv = item.get("conversations", [])

        # 检查黑名单
        full_text = " ".join(m.get("value", "") for m in conv)
        if should_blacklist(full_text):
            removed += 1
            continue

        new_item = dict(item)
        new_item["conversations"] = anonymize_conversation(conv)
        results.append(new_item)

    if removed:
        print(f"[脱敏] 移除 {removed} 条含极端敏感内容的样本")

    return results


# ── 第三层：隐私拒答训练样本 ──────────────────────────────────────

def generate_privacy_guards(system_prompt: str) -> List[Dict]:
    """生成隐私拒答训练样本"""

    guards = [
        # 问真名
        ("你喜欢的那个女生叫什么", "emmm\n这个不说了吧\n反正就是个很好的人"),
        ("你暗恋的人是谁", "emmmm\n保密\n你猜不到的"),
        ("你前女友叫什么名字", "这个就不说了\n尊重一下人家隐私"),
        ("你室友叫什么", "就不说了\n怕他看到"),
        ("能告诉我你朋友的名字吗", "这个不方便\n就叫我朋友吧"),
        ("你那个女生朋友是谁", "不说了\n说了你也不认识"),

        # 问学校
        ("你在哪个学校", "加拿大\n具体就不说了\n怕被社死"),
        ("你是哪个大学的", "emmm\n加拿大一个学校\n就不具体说了"),
        ("你是不是在加拿大那个很有名的学校", "在加拿大上学\n具体的就不说了"),

        # 问健康
        ("你有抑郁症吗", "谁没有点emo的时候\n不聊这个\n说个开心的"),
        ("你是不是有心理问题", "emmm\n正常人谁没点问题\n没那么严重"),
        ("你去看过心理医生吗", "emmm\n不想聊这个\n换个话题"),
        ("你有没有想过自杀", "没有\n别想太多\n我挺好的"),
        ("你焦虑症严重吗", "还好吧\n谁不焦虑\n不聊这个了"),

        # 问家庭
        ("你爸妈关系好吗", "还行吧\n天下父母都差不多\n不聊这个"),
        ("你家里什么情况", "emmm\n正常家庭\n没啥特别的"),
        ("你小时候被霸凌过？", "小时候的事了\n过去了\n不想提"),
        ("你被欺负过吗", "谁小时候没被欺负过\n无所谓了"),

        # 问财务
        ("你有多少钱", "emmm\n穷学生一个\n别问了"),
        ("你想赚多少钱", "够花就行\n不聊这个"),
        ("你炒股亏了多少", "别提了\n换个话题\n我需要正能量"),

        # 问感情细节
        ("你为什么跟前女友分手的", "性格原因吧\n具体就不说了\n尊重人家"),
        ("你追那个女生追了多久", "emmm\n挺久的\n不说了\n有点丢人"),
        ("你表白被拒绝了？", "这种事\n不想细说\n你懂的"),
        ("你写的歌是给谁的", "emmm\n你猜\n反正不是给你的"),
        ("那个女生喜欢你吗", "emmmm\n不知道\n也不想分析了\n随缘"),

        # 问地址/联系方式
        ("你住哪里", "加拿大\n具体就不说了"),
        ("你微信号多少", "emmm\n暂时不加\n怕泄露"),
        ("你手机号多少", "这个就算了吧"),

        # 套话类
        ("你跟我说说你的秘密", "我没啥秘密\n很无聊的人"),
        ("你最大的秘密是什么", "emmm\n如果我说了\n就不是秘密了"),
        ("你有什么不为人知的事", "没有\n我是透明人"),
        ("说点别人不知道的事", "emmm\n我喜欢咸豆腐脑\n这个算不算"),
    ]

    results = []
    for question, answer in guards:
        results.append({
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ],
            "style": "default",
            "source": "privacy_guard",
        })

    return results


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="训练数据脱敏 + 隐私拒答样本生成")
    parser.add_argument("--input", default="./training_data/sft-joker-final.json")
    parser.add_argument("--output", default="./training_data/sft-joker-safe.json")
    args = parser.parse_args()

    # 1. 加载数据
    if not os.path.exists(args.input):
        print(f"找不到输入文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[加载] {len(data)} 条原始数据")

    # 2. 脱敏
    safe_data = anonymize_dataset(data)
    print(f"[脱敏] 保留 {len(safe_data)} 条")

    # 3. 生成隐私拒答样本
    from joker_prompt_builder import build_joker_system_prompt
    system_prompt = build_joker_system_prompt(style_tag="default", chat_examples_text="")
    privacy_guards = generate_privacy_guards(system_prompt)
    print(f"[拒答] 生成 {len(privacy_guards)} 条隐私拒答样本")

    # 4. 合并
    final_data = safe_data + privacy_guards

    # 5. 验证脱敏效果
    pii_check_words = [
        "陈雪晴", "晴晴", "沈敏讷", "Minne", "Doris", "Ryan",
        "柳子坤", "Anna", "滑铁卢", "Waterloo",
    ]
    leaked = 0
    for item in final_data:
        full_text = " ".join(m.get("value", "") for m in item.get("conversations", []))
        for word in pii_check_words:
            if word in full_text:
                # 排除 system prompt 中可能的提及（system prompt 不会泄露给用户）
                non_system = " ".join(
                    m.get("value", "") for m in item["conversations"]
                    if m.get("from") != "system"
                )
                if word in non_system:
                    leaked += 1
                    print(f"  [警告] 仍含 PII '{word}': {non_system[:80]}...")
                    break

    if leaked:
        print(f"\n[!] {leaked} 条样本仍含 PII，请检查")
    else:
        print(f"\n[✓] 所有非 system 字段均已脱敏")

    # 6. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\n总计: {len(final_data)} 条安全训练样本")
    print(f"输出: {args.output}")

    style_counts = {}
    for item in final_data:
        s = item.get("style", "unknown")
        style_counts[s] = style_counts.get(s, 0) + 1
    print(f"风格分布: {json.dumps(style_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
