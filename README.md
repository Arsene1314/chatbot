# DeepSeek 语气模仿聊天

一个简单的 CLI 程序：根据不同标签（同学/暧昧/家长/室友），调用 DeepSeek API 生成“像你”的聊天回复。

## 1) 安装

```bash
cd /Users/joker/Desktop/deepseek_style_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 配置

设置环境变量：

```bash
export DEEPSEEK_API_KEY="你的key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
```

或者在项目根目录新建 `.env`（推荐，任务会自动读取）：

```bash
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## 3) 使用

单次调用：

```bash
python main.py --tag classmate --input "第一题最后一问你会吗"
```

交互模式：

```bash
python main.py --tag ambiguous
```

带历史对话（JSON 数组）：

```bash
python main.py --tag roommate --history ./history.json
```

`history.json` 示例：

```json
[
  {"role": "user", "content": "你现在写到哪啦"},
  {"role": "assistant", "content": "刚写完第一题"}
]
```

## 4) 标签与风格配置

所有风格配置在 `config/styles.json`，你可以：
- 修改 `base.global_rules` 和 `base.output_rules`
- 为每个标签添加/替换 `tone`、`language_habits`、`emoji_style`
- 增加 `examples`（会作为 few-shot 示例注入）

标签键名示例：
- `classmate` 同学/课友
- `ambiguous` 暧昧对象
- `parent` 家长
- `roommate` 室友

## 5) 常用参数

- `--tag` 选择语气标签
- `--max-examples` few-shot 示例条数（默认 3）
- `--temperature` 随机度（建议 0.7-0.9）
- `--max-tokens` 回复长度上限
- `--max-rounds` 交互模式保留的历史轮数

## 6) 注意

该程序只是“语气模仿器”，不会自动读取微信，也不会替你发送消息。
如果你要接入真实聊天平台，请确保合规与对方知情。

## 7) 在 Cursor 里一键运行

已添加 `.vscode/tasks.json`，可在 Cursor 里用任务启动：

1. `Cmd+Shift+P` → 输入 `Tasks: Run Task`
2. 选择 `Style Bot: 单次回复` 或 `Style Bot: 交互模式`
3. 选择标签并输入内容

注意：任务会使用当前终端环境变量中的 `DEEPSEEK_API_KEY`。
