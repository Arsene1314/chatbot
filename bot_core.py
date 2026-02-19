"""
核心机器人逻辑 — 供 CLI (main.py) 和企业微信 (wecom_bot.py) 共用。
负责加载聊天样本、构建 prompt、调用 DeepSeek API。
支持两种人格：QingqingBot（晴晴）和 JokerBot（数字分身）。
"""
import os
from typing import Dict, List, Optional

from openai import OpenAI

from chat_parser import parse_chat_file, conversations_to_example_text
from prompt_builder import load_styles, build_messages, build_system_prompt
from joker_prompt_builder import build_joker_messages


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


def call_deepseek(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str,
    api_key: str,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def find_chat_samples(base_dir: str = ".") -> str:
    candidates = [
        os.path.join(base_dir, "chat_samples_副本.txt"),
        os.path.join(base_dir, "../Downloads/chat_samples_副本.txt"),
        os.path.expanduser("~/Downloads/chat_samples_副本.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


class QingqingBot:
    """晴晴机器人核心：加载样本 + 管理会话历史 + 生成回复"""

    def __init__(
        self,
        config_path: str = "./config/styles.json",
        chat_samples_path: Optional[str] = None,
        tag: str = "ambiguous",
        model: str = "deepseek-chat",
        temperature: float = 0.85,
        max_tokens: int = 100,
        max_rounds: int = 8,
    ):
        self.tag = tag
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds

        # 环境变量
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        # 加载风格配置
        self.styles = load_styles(config_path)

        # 加载聊天样本
        base_dir = os.path.dirname(os.path.abspath(__file__))
        samples_path = chat_samples_path or find_chat_samples(base_dir)
        all_conversations = []

        if samples_path and os.path.exists(samples_path):
            real_convs = parse_chat_file(samples_path)
            all_conversations.extend(real_convs)
            print(f"[bot_core] 已加载 {len(real_convs)} 条真实对话")

        generated_path = os.path.join(base_dir, "chat_samples_generated.txt")
        if os.path.exists(generated_path):
            gen_convs = parse_chat_file(generated_path)
            all_conversations.extend(gen_convs)
            print(f"[bot_core] 已加载 {len(gen_convs)} 条生成对话")

        if all_conversations:
            self.chat_examples_text = conversations_to_example_text(all_conversations)
            total_chars = len(self.chat_examples_text)
            print(
                f"[bot_core] 共 {len(all_conversations)} 条对话"
                f"（约 {total_chars} 字 / ~{int(total_chars * 1.5)} tokens）"
            )
        else:
            self.chat_examples_text = ""
            print("[bot_core] 未找到聊天记录文件，将使用纯 prompt 模式")

        # 每个用户独立的对话历史 {user_id: [messages]}
        self._histories: Dict[str, List[Dict]] = {}

    def _cap_history(self, history: List[Dict]) -> List[Dict]:
        if self.max_rounds <= 0:
            return []
        max_items = self.max_rounds * 2
        return history[-max_items:]

    def get_history(self, user_id: str) -> List[Dict]:
        return self._histories.get(user_id, [])

    def clear_history(self, user_id: str) -> None:
        self._histories.pop(user_id, None)

    def reply(self, user_input: str, user_id: str = "default") -> str:
        """生成回复并自动维护会话历史"""
        history = self.get_history(user_id)

        messages = build_messages(
            user_input=user_input,
            styles=self.styles,
            tag_key=self.tag,
            chat_examples_text=self.chat_examples_text,
            history=history,
        )

        answer = call_deepseek(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # 更新历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        self._histories[user_id] = self._cap_history(history)

        return answer


# ──────────────────────────────────────────────────────────────────
# JokerBot — 数字分身
# ──────────────────────────────────────────────────────────────────

# 每种风格对应的聊天记录文件（相对于 joker_profile/ 目录）
JOKER_CHAT_SOURCES = {
    "brother": "Cloud 9 (室友柳子坤).txt",
    "female_friend": "和Doris的聊天记录.txt",
    "crush": "Cloud 9 (室友柳子坤).txt",
    "ex": "和Doris的聊天记录.txt",
    "default": None,
}


class JokerBot:
    """Joker 数字分身：多风格聊天 + 会话历史管理"""

    def __init__(
        self,
        style_tag: str = "default",
        profile_dir: str = "./joker_profile",
        model: str = "deepseek-chat",
        temperature: float = 0.90,
        max_tokens: int = 150,
        max_rounds: int = 10,
    ):
        self.style_tag = style_tag
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds
        self.profile_dir = os.path.abspath(profile_dir)

        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        # 按风格加载聊天样本
        self._examples_cache: Dict[str, str] = {}
        self._load_examples(style_tag)

        self._histories: Dict[str, List[Dict]] = {}
        print(f"[JokerBot] 初始化完成 | 风格={style_tag} | 模型={model}")

    def _load_examples(self, tag: str) -> None:
        """加载指定风格的聊天样本"""
        if tag in self._examples_cache:
            return

        source_file = JOKER_CHAT_SOURCES.get(tag)
        if not source_file:
            self._examples_cache[tag] = ""
            return

        path = os.path.join(self.profile_dir, source_file)
        if not os.path.exists(path):
            print(f"[JokerBot] 警告：找不到聊天记录 {path}")
            self._examples_cache[tag] = ""
            return

        convs = parse_joker_chat_file(path)
        if convs:
            text = joker_conversations_to_text(convs)
            self._examples_cache[tag] = text
            print(f"[JokerBot] 已加载 {len(convs)} 条对话 ({tag})")
        else:
            self._examples_cache[tag] = ""
            print(f"[JokerBot] {source_file} 中未解析到对话")

    def switch_style(self, new_tag: str) -> None:
        """切换对话风格"""
        if new_tag not in JOKER_CHAT_SOURCES:
            print(f"[JokerBot] 未知风格 {new_tag}，可选: {list(JOKER_CHAT_SOURCES)}")
            return
        self.style_tag = new_tag
        self._load_examples(new_tag)
        print(f"[JokerBot] 风格切换为: {new_tag}")

    def _cap_history(self, history: List[Dict]) -> List[Dict]:
        if self.max_rounds <= 0:
            return []
        max_items = self.max_rounds * 2
        return history[-max_items:]

    def get_history(self, user_id: str) -> List[Dict]:
        return self._histories.get(user_id, [])

    def clear_history(self, user_id: str) -> None:
        self._histories.pop(user_id, None)

    def reply(self, user_input: str, user_id: str = "default") -> str:
        """生成 Joker 风格的回复"""
        history = self.get_history(user_id)
        examples = self._examples_cache.get(self.style_tag, "")

        messages = build_joker_messages(
            user_input=user_input,
            style_tag=self.style_tag,
            chat_examples_text=examples,
            history=history,
        )

        answer = call_deepseek(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        self._histories[user_id] = self._cap_history(history)

        return answer


# ── Joker 聊天记录解析 ────────────────────────────────────────────

import re


def parse_joker_chat_file(filepath: str) -> List[List[Dict[str, str]]]:
    """
    解析 Joker 的微信聊天导出记录。
    支持两种格式：
      名字: 内容        （内容在同一行）
      名字:             （内容在下一行）
      内容
    自动识别 Joker 侧为 assistant，其他人为 user。
    按固定窗口（CHUNK_SIZE 轮）切分为多段对话用于 many-shot。
    """
    joker_names = {"雨中的马孔多", "joker", "我", "Joker", "JOKER"}
    CHUNK_SIZE = 10  # 每段对话的最大 turn 数

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 第一遍：解析成一个平坦的 turn 列表
    all_turns: List[Dict[str, str]] = []
    pending_name: Optional[str] = None  # 上一行是 "名字:" 但没内容

    # 匹配 "名字: 内容" 或 "名字:" （内容可能为空）
    msg_pattern = re.compile(r"^(.+?)\s*[：:]\s*(.*)$")

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            pending_name = None
            continue

        m = msg_pattern.match(line)
        if m:
            name = m.group(1).strip()
            content = m.group(2).strip()

            # 确认是已知的发言人名字（避免把普通含冒号的句子误解析）
            is_known = name in joker_names or len(name) <= 30

            if is_known and (name in joker_names or not content or len(name) < 20):
                role = "assistant" if name in joker_names else "user"

                if content:
                    # 同一 role 连续发言，合并
                    if all_turns and all_turns[-1]["role"] == role:
                        all_turns[-1]["content"] += "\n" + content
                    else:
                        all_turns.append({"role": role, "content": content})
                    pending_name = None
                else:
                    pending_name = name
                continue

        # 没匹配到格式头：可能是上一个 "名字:" 的内容，或者是续行
        if pending_name is not None:
            role = "assistant" if pending_name in joker_names else "user"
            if all_turns and all_turns[-1]["role"] == role:
                all_turns[-1]["content"] += "\n" + line
            else:
                all_turns.append({"role": role, "content": line})
            pending_name = None
        elif all_turns:
            all_turns[-1]["content"] += "\n" + line

    # 第二遍：按窗口切分为多段对话
    conversations: List[List[Dict[str, str]]] = []
    i = 0
    while i < len(all_turns):
        chunk = all_turns[i: i + CHUNK_SIZE]
        if len(chunk) >= 2:
            conversations.append(chunk)
        i += CHUNK_SIZE

    print(f"[joker_parser] 从 {filepath} 解析到 {len(all_turns)} 个 turn，切分为 {len(conversations)} 段对话")
    return conversations


def joker_conversations_to_text(
    conversations: List[List[Dict[str, str]]],
) -> str:
    """将 Joker 的对话列表转为 prompt 嵌入格式"""
    parts = []
    for i, conv in enumerate(conversations, 1):
        conv_lines = []
        for msg in conv:
            prefix = "对方" if msg["role"] == "user" else "雨中的马孔多"
            for line in msg["content"].split("\n"):
                line = line.strip()
                if line:
                    conv_lines.append(f"{prefix}：{line}")
        if conv_lines:
            parts.append(f"【对话{i}】\n" + "\n".join(conv_lines))
    return "\n\n".join(parts)
