import torch, re, json, sys, base64
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_PATH = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
LORA_PATH = "/root/autodl-tmp/output-qwen25"

_role_arg = sys.argv[1] if len(sys.argv) > 1 else "casual"
_role_keyword = {
    "brother": "兄弟/好哥们",
    "girl_friend": "女生朋友（纯友谊）",
    "casual": "一般朋友/泛交",
    "crush": "暗恋/追求对象",
    "ex": "前任/很亲密的异性朋友",
}
_target = _role_keyword.get(_role_arg, "一般朋友/泛交")

import json as _json
with open("/root/data/sft-joker-safe.json", "r", encoding="utf-8") as _f:
    _data = _json.load(_f)
_base_system = ""
for _conv in _data:
    for _msg in _conv["conversations"]:
        if _msg["from"] == "system" and _target in _msg["value"]:
            _base_system = _msg["value"]
            break
    if _base_system:
        break

SYSTEM = _base_system + "\n\n【最最重要的规则】你只能根据对方实际发的消息来回复。绝对禁止编造对方没说过的事情、人物、场景。如果对方只是打招呼，你就正常回应打招呼，不要凭空生成话题。"

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, torch_dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cuda:0")
model.eval()
print("MODEL_READY", flush=True)

while True:
    line = sys.stdin.readline().strip()
    if not line or line == "EXIT":
        break
    history = json.loads(base64.b64decode(line).decode("utf-8"))

    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(history)

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "<think>" in response:
        response = response.split("</think>")[-1].strip() if "</think>" in response else response.split("<think>")[0].strip()

    encoded = base64.b64encode(response.encode("utf-8")).decode("ascii")
    print(f"REPLY:{encoded}", flush=True)
