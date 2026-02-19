import pyjson5, json

with open("settings.jsonc", "r") as f:
    cfg = pyjson5.load(f)

cfg["make_dataset_args"]["telegram_args"]["my_id"] = "1234567890"

with open("settings.jsonc", "w") as f:
    json.dump(cfg, f, indent=4, ensure_ascii=False)

print("CONFIG_FIXED: my_id updated to 1234567890")
