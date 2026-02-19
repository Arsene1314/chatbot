import pyjson5, json

with open("settings.jsonc", "r") as f:
    cfg = pyjson5.load(f)

# LoRA 大幅增强
cfg["train_sft_args"]["lora_target"] = "all"  # 训练所有linear层
cfg["train_sft_args"]["lora_rank"] = 32  # 从8提升到32
cfg["train_sft_args"]["lora_dropout"] = 0.1  # 降低dropout
cfg["train_sft_args"]["num_train_epochs"] = 20  # 20个epoch
cfg["train_sft_args"]["learning_rate"] = 5e-5  # 稍高一点的lr
cfg["train_sft_args"]["warmup_ratio"] = 0.1
cfg["train_sft_args"]["save_steps"] = 500
cfg["train_sft_args"]["logging_steps"] = 5
cfg["train_sft_args"]["weight_decay"] = 0.01  # 降低weight decay

with open("settings.jsonc", "w") as f:
    json.dump(cfg, f, indent=4, ensure_ascii=False)

print("CONFIG_V2_UPDATED")
print(f"  lora_target: {cfg['train_sft_args']['lora_target']}")
print(f"  lora_rank: {cfg['train_sft_args']['lora_rank']}")
print(f"  lora_dropout: {cfg['train_sft_args']['lora_dropout']}")
print(f"  epochs: {cfg['train_sft_args']['num_train_epochs']}")
print(f"  learning_rate: {cfg['train_sft_args']['learning_rate']}")
print(f"  weight_decay: {cfg['train_sft_args']['weight_decay']}")
