import json
from typing import Dict, List, Optional


def load_styles(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(base: Dict, tag: Dict) -> str:
    persona_name = base.get("persona_name", "我")
    role = base.get("role", "你是用户本人在聊天，使用第一人称回复。")
    global_rules = base.get("global_rules", [])
    output_rules = base.get("output_rules", [])

    parts = [
        "【角色】",
        f"{role} 你叫「{persona_name}」，正在和对方自然聊天。",
        f"【沟通对象】{tag.get('label', '')}",
        "【语气画像】",
        *[f"- {line}" for line in tag.get("tone", [])],
        "【语言习惯】",
        *[f"- {line}" for line in tag.get("language_habits", [])],
        "【表情/符号风格】",
        *[f"- {line}" for line in tag.get("emoji_style", [])],
        "【硬性规则】",
        *[f"- {line}" for line in global_rules],
        "【输出要求】",
        *[f"- {line}" for line in output_rules],
        "【额外提醒】",
        "如果对方问你是不是AI/机器人，用轻松含糊的语气糊弄过去，不要承认。",
        "输出只包含回复内容本身，不要附带说明或注释。"
    ]
    return "\n".join([p for p in parts if p])


def build_messages(
    user_input: str,
    styles: Dict,
    tag_key: str,
    history: Optional[List[Dict]] = None,
    max_examples: int = 3
) -> List[Dict]:
    base = styles.get("base", {})
    tags = styles.get("tags", {})
    tag = tags.get(tag_key) or next(iter(tags.values()))

    system_prompt = build_system_prompt(base, tag)
    messages: List[Dict] = [{"role": "system", "content": system_prompt}]

    examples = tag.get("examples", [])[:max_examples]
    for ex in examples:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    if history:
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_input})
    return messages
