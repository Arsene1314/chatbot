# Joker 模型训练文件与参数清单

更新时间：2026-02-19

## 1) 本地项目训练数据（`training_data/`）

- `training_data/sft-joker-safe.json`
- `training_data/sft-joker-clean.json`
- `training_data/sft-joker-chat.json`
- `training_data/sft-joker-balanced.json`
- `training_data/sft-joker-final.json`
- `training_data/sft-joker-synthetic.json`
- `training_data/sft-joker-synthetic-large.json`
- `training_data/sft-joker-synthetic-daily.json`
- `training_data/sft-daily.json`
- `training_data/openai-finetune.jsonl`
- `training_data/openai-finetune-real.jsonl`
- `training_data/preview.txt`

## 2) 本地训练/转换脚本

- `generate_joker.py`
- `balance_data.py`
- `prepare_openai_finetune.py`
- `run_finetune.py`
- `chat_finetune.py`
- `chat_local.py`

## 3) 本地 LLaMA Factory 配置

- `llamafactory_config/joker_lora.yaml`
- `llamafactory_config/dataset_info.json`
- `llamafactory_config/setup_autodl.sh`

`llamafactory_config/dataset_info.json` 中的核心映射：

- dataset 名称：`joker_sft`
- 源文件：`sft-joker-safe.json`
- 格式：`sharegpt`
- role 映射：`human/gpt/system`

## 4) AutoDL 远端训练配置（真实跑过）

- `/root/qwen25_lora.yaml`
- `/root/qwen3_lora.yaml`
- `/root/joker_lora.yaml`

### 共同 LoRA 训练超参（qwen25/qwen3）

- `stage: sft`
- `finetuning_type: lora`
- `lora_rank: 64`
- `lora_alpha: 128`
- `lora_dropout: 0.05`
- `lora_target: all`
- `dataset: joker_sft`
- `template: qwen`
- `cutoff_len: 1024`
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 8`
- `learning_rate: 2.0e-5`
- `num_train_epochs: 5`
- `lr_scheduler_type: cosine`
- `warmup_ratio: 0.1`
- `weight_decay: 0.01`
- `max_grad_norm: 1.0`
- `bf16: true`
- `seed: 42`
- `report_to: none`

### 模型与输出目录

- Qwen2.5：
  - `model_name_or_path: /root/autodl-tmp/Qwen2.5-14B-Instruct`
  - `output_dir: /root/autodl-tmp/output-qwen25`
  - `resume_from_checkpoint: /root/autodl-tmp/output-qwen25/checkpoint-400`
- Qwen3：
  - `model_name_or_path: /root/autodl-tmp/Qwen3-14B`
  - `output_dir: /root/autodl-tmp/output-qwen3`
  - `resume_from_checkpoint: /root/autodl-tmp/output-qwen3/checkpoint-400`
- DeepSeek-R1-Distill：
  - `model_name_or_path: /root/autodl-tmp/DeepSeek-R1-Distill-Qwen-14B`
  - `output_dir: ./output/joker-lora-r1-14b`

## 5) AutoDL 远端 LoRA 产物目录

- `/root/autodl-tmp/output-qwen25`
- `/root/autodl-tmp/output-qwen3`

每个目录都包含：

- `adapter_model.safetensors`
- `adapter_config.json`
- `train_results.json`
- `trainer_state.json`
- `trainer_log.jsonl`
- `checkpoint-*`

## 6) 关键训练结果（远端）

来自 `/root/autodl-tmp/output-qwen25/train_results.json`：

- `train_loss: 0.5183690153600599`
- `epoch: 5.0`
- `train_runtime: 9303.9419`

来自 `/root/autodl-tmp/output-qwen3/train_results.json`：

- `train_loss: 0.5331358918672279`
- `epoch: 5.0`
- `train_runtime: 9263.4777`

## 7) OpenAI Fine-tune 产物（本地）

- `finetune_job_id.txt`：`ftjob-MRiQc24tneB6j3eWThMU5yDh`
- `fine_tuned_model.txt`：`ft:gpt-4o-mini-2024-07-18:love-qing:joker-real:DAg7x5g9`

