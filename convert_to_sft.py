"""
Phase 2: 将所有聊天记录 + Cursor transcript 转换为 ShareGPT 格式的 SFT 训练数据。

输出格式 (ShareGPT):
[
  {
    "conversations": [
      {"from": "system", "value": "你是Joker的数字分身..."},
      {"from": "human", "value": "用户消息"},
      {"from": "gpt", "value": "Joker的回复"},
      ...
    ],
    "style": "brother"
  },
  ...
]

数据来源:
1. joker_profile/Cloud 9 (室友柳子坤).txt  → brother + crush 风格
2. joker_profile/和Doris的聊天记录.txt       → female_friend + ex 风格
3. Cursor agent transcript                    → default 风格（自我表达）
"""
import json
import os
import re
import sys
from typing import Dict, List, Optional

from bot_core import parse_joker_chat_file
from joker_prompt_builder import build_joker_system_prompt


# ── 配置 ─────────────────────────────────────────────────────────

PROFILE_DIR = "./joker_profile"
OUTPUT_PATH = "./training_data/sft-joker-chat.json"

# 聊天记录 → 风格映射
CHAT_SOURCES = [
    {
        "file": "Cloud 9 (室友柳子坤).txt",
        "style": "brother",
        "description": "与室友/兄弟的日常对话",
    },
    {
        "file": "和Doris的聊天记录.txt",
        "style": "female_friend",
        "description": "与女生朋友的日常对话",
    },
]


def chat_to_sharegpt(
    conversations: List[List[Dict[str, str]]],
    style: str,
    description: str,
) -> List[Dict]:
    """将解析好的对话列表转为 ShareGPT 格式"""
    system_prompt = build_joker_system_prompt(style_tag=style, chat_examples_text="")
    results = []

    for conv in conversations:
        if len(conv) < 2:
            continue

        sharegpt_conv = [
            {"from": "system", "value": system_prompt},
        ]

        for msg in conv:
            if msg["role"] == "user":
                sharegpt_conv.append({"from": "human", "value": msg["content"]})
            elif msg["role"] == "assistant":
                sharegpt_conv.append({"from": "gpt", "value": msg["content"]})

        # 确保对话至少有一轮有效交互（system + human + gpt）
        has_human = any(m["from"] == "human" for m in sharegpt_conv)
        has_gpt = any(m["from"] == "gpt" for m in sharegpt_conv)
        if has_human and has_gpt:
            results.append({
                "conversations": sharegpt_conv,
                "style": style,
                "source": description,
            })

    return results


def parse_transcript_to_sharegpt(transcript_path: str) -> List[Dict]:
    """
    从 Cursor agent transcript 中提取用户的个人叙述内容，
    转为 ShareGPT 格式的 default 风格训练数据。
    只提取 <user_query> 标签内、长度足够、非技术内容的自述。
    """
    if not os.path.exists(transcript_path):
        print(f"[transcript] 未找到: {transcript_path}")
        return []

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()

    results = []
    system_prompt = build_joker_system_prompt(style_tag="default", chat_examples_text="")

    # 提取 <user_query> 标签内的内容
    query_pattern = re.compile(r"<user_query>\s*(.+?)\s*</user_query>", re.DOTALL)
    raw_messages = query_pattern.findall(content)

    # 也提取没有 user_query 标签的纯 user: 消息（用户直接输入的长段自述）
    # 这些在 transcript 里是 user: 后面直接跟内容（非 tool/image 相关）
    bare_user_pattern = re.compile(
        r"^user:\s*\n(.+?)(?=^(?:assistant:|user:|\[Tool|\[Thinking))",
        re.MULTILINE | re.DOTALL,
    )
    bare_messages = bare_user_pattern.findall(content)
    for bm in bare_messages:
        cleaned = bm.strip()
        # 去掉 XML 标签残留
        cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()
        if cleaned and len(cleaned) > 30:
            raw_messages.append(cleaned)

    # 过滤条件：技术/指令类
    tech_keywords = [
        "cpolar", "python", "npm", "git", "terminal", "pip", "error",
        "traceback", ".py", "bug", "flask", "wechaty", "import", "def ",
        "class ", "curl", "http", "json", "xml", "apt", "brew",
        "$", "authtoken", "mkdir", "chmod", "ssh", "lora", "微调",
        "模型", "api", "部署", "训练", "prompt", "deepseek", "qwen",
        "聊天记录", "chat_sample", "config", "样本", "生成更多",
        "token", "gpu", "autodl", "远程", "操控", "租", "配置",
        "fine-tune", "finetune", "sharegpt", "sft", "你先",
        "你能不能", "你看看", "你检查", "你试试", "你搜搜",
        "发你", "放进去", "txt", "文件", "pages", "实现",
        "功能", "代码", "phase", "convert", "parse",
        "回复", "表情包", "看不到", "撤回", "发个", "聊天框",
        "两个问题", "分行", "你把", "提取", "核实", "表情",
        "人机", "规则", "测试", "更新", "修改", "添加",
    ]
    meta_keywords = [
        "image_files", "user_query", "open_and_recently", "system_reminder",
        "git_status", "user_info", "mcp_instructions", "screenshot",
        "@/users", "untited", "cpolar", "weclone",
    ]
    # 正面关键词：包含这些才保留（至少命中一个）
    personal_keywords = [
        "我", "喜欢", "觉得", "感觉", "以前", "小时候", "父母",
        "朋友", "她", "他", "恋爱", "分手", "焦虑", "抑郁",
        "音乐", "说唱", "写歌", "看书", "推理", "哲学",
        "霸凌", "高中", "大学", "初中", "小学", "出国",
        "性格", "infp", "enfj", "mbti", "健身", "减肥",
        "晴晴", "ryan", "minne", "doris", "anna",
        "挣", "赚", "努力", "摆烂", "上海", "滑铁卢",
        "mf", "stat", "退化", "华丽", "文字",
    ]

    personal_messages = []
    seen = set()

    for msg in raw_messages:
        msg = msg.strip()

        # 跳过太短的（个人叙述至少需要一定长度才有训练价值）
        if len(msg) < 50:
            continue

        # 跳过纯技术消息
        msg_lower = msg.lower()
        if any(kw in msg_lower for kw in tech_keywords):
            continue

        # 跳过元数据/系统标签
        if any(kw in msg_lower for kw in meta_keywords):
            continue

        # 必须包含至少一个个人关键词
        if not any(kw in msg_lower for kw in personal_keywords):
            continue

        # 去重（基于前50字）
        key = msg[:50]
        if key in seen:
            continue
        seen.add(key)

        personal_messages.append(msg)

    # 根据内容类型匹配引导问题
    prompts_map = {
        "感情": "你的感情经历是什么样的",
        "喜欢": "你喜欢什么样的人",
        "分手": "后来怎么了",
        "音乐": "你平时听什么音乐",
        "说唱": "跟我聊聊你的说唱",
        "父母": "跟我说说你的家庭",
        "霸凌": "你小时候经历了什么",
        "焦虑": "你的状态怎么样",
        "抑郁": "你最近还好吗",
        "mbti": "你觉得你是什么性格",
        "看书": "你平时看什么书",
    }
    default_prompts = [
        "跟我说说你的想法",
        "你怎么看这件事",
        "继续聊",
        "然后呢",
        "你觉得呢",
        "展开说说",
    ]

    for i, msg in enumerate(personal_messages):
        # 尝试匹配话题
        prompt = None
        for keyword, p in prompts_map.items():
            if keyword in msg:
                prompt = p
                break
        if not prompt:
            prompt = default_prompts[i % len(default_prompts)]

        results.append({
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": msg},
            ],
            "style": "default",
            "source": "cursor_transcript_self_narration",
        })

    return results


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_data: List[Dict] = []

    # 1. 聊天记录
    for source in CHAT_SOURCES:
        filepath = os.path.join(PROFILE_DIR, source["file"])
        if not os.path.exists(filepath):
            print(f"[跳过] 找不到: {filepath}")
            continue

        convs = parse_joker_chat_file(filepath)
        sharegpt = chat_to_sharegpt(convs, source["style"], source["description"])
        all_data.extend(sharegpt)
        print(f"[聊天记录] {source['file']} → {len(sharegpt)} 条 ({source['style']})")

    # 2. Cursor transcript
    transcript_path = os.path.expanduser(
        "~/.cursor/projects/Users-joker-Desktop-deepseek-style-bot/"
        "agent-transcripts/09daa8fb-b41a-4d61-9783-0bce0721e399.txt"
    )
    transcript_data = parse_transcript_to_sharegpt(transcript_path)
    all_data.extend(transcript_data)
    print(f"[Transcript] → {len(transcript_data)} 条 (default)")

    # 3. 输出
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n总计: {len(all_data)} 条训练样本")
    print(f"输出: {OUTPUT_PATH}")

    # 统计
    style_counts = {}
    for item in all_data:
        s = item["style"]
        style_counts[s] = style_counts.get(s, 0) + 1
    print("风格分布:", style_counts)


if __name__ == "__main__":
    main()
