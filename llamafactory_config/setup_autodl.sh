#!/bin/bash
# ── AutoDL 一键部署脚本（R1-Distill-Qwen-14B + LoRA）──
# 在 AutoDL 实例的终端运行此脚本

set -e

echo "=========================================="
echo "  Joker LoRA 训练环境部署"
echo "  模型: DeepSeek-R1-Distill-Qwen-14B"
echo "=========================================="

echo ""
echo "=== Step 1: 安装 LLaMA Factory ==="
pip install llamafactory[torch,metrics] -q
pip install deepspeed -q

echo ""
echo "=== Step 2: 创建目录结构 ==="
mkdir -p ~/data
mkdir -p ~/output

echo ""
echo "=== Step 3: 检查数据文件 ==="
if [ ! -f ~/data/sft-joker-safe.json ]; then
    echo "[!] 缺少 ~/data/sft-joker-safe.json"
    echo "    请先上传数据文件"
    exit 1
fi

if [ ! -f ~/data/dataset_info.json ]; then
    echo "[!] 缺少 ~/data/dataset_info.json"
    exit 1
fi

if [ ! -f ~/joker_lora.yaml ]; then
    echo "[!] 缺少 ~/joker_lora.yaml"
    exit 1
fi

echo "[✓] 所有文件就绪"

# 统计数据
python3 -c "
import json
with open('data/sft-joker-safe.json') as f:
    data = json.load(f)
print(f'训练样本: {len(data)} 条')
styles = {}
for d in data:
    s = d.get('style','?')
    styles[s] = styles.get(s,0)+1
for s,c in sorted(styles.items()):
    print(f'  {s}: {c}')
"

echo ""
echo "=== Step 4: 下载模型（如果本地没有）==="
# AutoDL 可能已经缓存了模型，先检查
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# 检查 AutoDL 公共模型缓存
if [ -d "/root/autodl-tmp/models/${MODEL_PATH}" ]; then
    echo "[✓] 模型已在 autodl-tmp 缓存中"
elif [ -d "/root/share/model_repos/${MODEL_PATH}" ]; then
    echo "[✓] 模型已在公共 share 目录中"
else
    echo "[↓] 开始下载模型..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${MODEL_PATH}',
    local_dir='/root/autodl-tmp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    resume_download=True
)
print('[✓] 模型下载完成')
"
fi

echo ""
echo "=== Step 5: 开始训练 ==="
echo "运行以下命令开始训练:"
echo ""
echo "  cd ~ && llamafactory-cli train joker_lora.yaml"
echo ""
echo "训练完成后，LoRA 权重保存在 ~/output/joker-lora-r1-14b/"
echo ""
echo "=== Step 6: 测试推理 ==="
echo "  llamafactory-cli chat joker_lora.yaml"
echo ""
echo "=== Step 7: 导出合并模型（可选）==="
echo "  llamafactory-cli export joker_lora.yaml \\"
echo "    --export_dir ~/output/joker-merged/ \\"
echo "    --export_size 2"
