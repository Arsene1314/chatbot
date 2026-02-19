import torch, re, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_PATH = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
LORA_PATH = "/root/autodl-tmp/output-qwen25"

with open("/root/data/sft-joker-safe.json", "r") as f:
    data = json.load(f)
SYSTEM = ""
for conv in data:
    for msg in conv["conversations"]:
        if msg["from"] == "system" and "一般朋友/泛交" in msg["value"]:
            SYSTEM = msg["value"]
            break
    if SYSTEM:
        break

SYSTEM += "\n\n【最最重要的规则】你只能根据对方实际发的消息来回复。绝对禁止编造对方没说过的事情、人物、场景。如果对方只是打招呼，你就正常回应打招呼，不要凭空生成话题。"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cuda:0")
model.eval()
print("Model loaded!\n")

def chat(user_msg, max_new_tokens=512):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1,
        )
    r = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    r = re.sub(r"<think>.*?</think>", "", r, flags=re.DOTALL).strip()
    return r

tests = [
    "你好",
    "在干嘛呢",
    "介绍一下你自己呗",
    "最近在听什么歌",
    "压力好大快撑不住了",
    "你觉得人活着的意义是什么",
    "你是不是又熬夜了",
    "你学什么专业的",
]

for t in tests:
    reply = chat(t)
    print(f"对方: {t}")
    print(f"Joker: {reply}")
    print()
