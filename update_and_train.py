import pyjson5, json, os

config_path = "settings.jsonc"

with open(config_path, "r") as f:
    cfg = pyjson5.load(f)

# 更新训练参数
cfg["train_sft_args"]["num_train_epochs"] = 15
cfg["train_sft_args"]["learning_rate"] = 2e-5
cfg["train_sft_args"]["warmup_ratio"] = 0.05
cfg["train_sft_args"]["save_steps"] = 200
cfg["train_sft_args"]["logging_steps"] = 5

# 清空旧的输出目录内容
output_dir = cfg["common_args"]["adapter_name_or_path"]

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=4, ensure_ascii=False)

print(f"CONFIG_UPDATED")
print(f"  epochs: {cfg['train_sft_args']['num_train_epochs']}")
print(f"  learning_rate: {cfg['train_sft_args']['learning_rate']}")
print(f"  warmup_ratio: {cfg['train_sft_args']['warmup_ratio']}")
print(f"  save_steps: {cfg['train_sft_args']['save_steps']}")
print(f"  output_dir: {output_dir}")
