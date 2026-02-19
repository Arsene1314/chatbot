"""
解析真实聊天记录文件，提取所有对话为结构化格式。
每条对话包含多轮 user/assistant 交替。
连续的 Q: 行合并为一条 user 消息，连续的 A: 行合并为一条 assistant 消息。
"""
import re
from typing import List, Dict


EXCLUDE_MARKERS = [
    "gol'lanzhou",
    "我mf的",
    "mathematical finance",
    "我现在正在",
    "试图用云端跑chat模型",
    "做音乐那个软件占了68个G",
    "我训练出来一个和我说话一样的模型",
    "尼采说的超人",
    "我陪她溜一溜",
    "最好有个篝火",
]


def _should_exclude(conv: List[Dict[str, str]]) -> bool:
    """检查对话是否包含已知的假数据标记"""
    full_text = " ".join(m["content"] for m in conv).lower()
    for marker in EXCLUDE_MARKERS:
        if marker.lower() in full_text:
            return True
    return False


def parse_chat_file(filepath: str) -> List[List[Dict[str, str]]]:
    """
    解析 chat_samples 文件，返回对话列表。
    每条对话是 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    conversations: List[List[Dict[str, str]]] = []
    current_conv: List[Dict[str, str]] = []
    current_role = None
    current_lines: List[str] = []

    def flush():
        nonlocal current_role, current_lines
        if current_role and current_lines:
            role = "user" if current_role == "Q" else "assistant"
            content = "\n".join(current_lines)
            current_conv.append({"role": role, "content": content})
        current_role = None
        current_lines = []

    def flush_conv():
        nonlocal current_conv
        flush()
        if current_conv:
            conversations.append(current_conv)
            current_conv = []

    for raw_line in lines:
        line = raw_line.rstrip("\n").strip()

        # 跳过空行
        if not line:
            continue

        # 跳过注释行
        if line.startswith("//"):
            continue

        # 新对话编号（如 "1." "2." "45."）
        if re.match(r"^\d+\.\s*$", line):
            flush_conv()
            continue

        # Q: 或 A: 开头的行
        m = re.match(r"^([QA]):(.*)$", line)
        if m:
            role_char = m.group(1)
            text = m.group(2).strip()
            if role_char != current_role:
                flush()
                current_role = role_char
            if text:
                current_lines.append(text)
            continue

    # 处理最后一条对话
    flush_conv()

    # 过滤掉包含假数据的对话
    filtered = [c for c in conversations if not _should_exclude(c)]
    removed = len(conversations) - len(filtered)
    if removed > 0:
        print(f"  （已过滤 {removed} 条含假数据的对话）")

    return filtered


def conversations_to_example_text(
    conversations: List[List[Dict[str, str]]],
    filter_words: List[str] = None,
) -> str:
    """
    将对话列表转换为可嵌入 system prompt 的文本格式。
    filter_words: 需要从助手回复中替换掉的词列表
    """
    if filter_words is None:
        filter_words = ["哥哥"]

    parts = []
    for i, conv in enumerate(conversations, 1):
        conv_lines = []
        for msg in conv:
            prefix = "对方" if msg["role"] == "user" else "晴晴"
            content = msg["content"]
            # 过滤所有消息中的指定词
            for word in filter_words:
                content = content.replace(word, "")
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    conv_lines.append(f"{prefix}：{line}")
        parts.append(f"【对话{i}】\n" + "\n".join(conv_lines))
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "chat_samples_副本.txt"
    convs = parse_chat_file(filepath)
    print(f"解析到 {len(convs)} 条对话")

    total_turns = sum(len(c) for c in convs)
    print(f"总轮次: {total_turns}")

    # 预览前3条
    for i, conv in enumerate(convs[:3], 1):
        print(f"\n--- 对话 {i} ---")
        for msg in conv:
            print(f"  [{msg['role']}] {msg['content'][:60]}...")

    # 生成示例文本并统计字符数
    text = conversations_to_example_text(convs)
    print(f"\n示例文本总字符数: {len(text)}")
    print(f"预估 token 数: ~{int(len(text) * 1.5)}")
