import torch, re, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_path, lora_path, device):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path, device_map={"": device})
    model.eval()
    return tokenizer, model

def chat(tokenizer, model, system, user_msg, max_new_tokens=512):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "<think>" in response:
        response = response.split("</think>")[-1].strip() if "</think>" in response else response.split("<think>")[0].strip()
    return response

SYSTEM = (
    "你是雨中的马孔多（Joker），一个在加拿大留学的中国男生，INFP性格。"
    "说话风格：微信聊天风格，简短随意，经常用省略号和换行，偶尔用拼音缩写，"
    "会用哈哈哈、嗯嗯、啊这等口语化表达。"
    "性格特点：内向但对熟人很话多，喜欢深度思考，有点丧但本质温柔，对感兴趣的话题会突然变得很热情。"
    "不要输出思考过程，直接用微信聊天的方式回复。"
)

TESTS = [
    ("日常问候", "在干嘛呢"),
    ("情感倾诉", "最近压力好大，感觉快撑不住了"),
    ("兴趣话题", "你最近在听什么歌"),
    ("深度话题", "你觉得人活着的意义是什么"),
    ("调侃", "你是不是又熬夜了"),
    ("留学生活", "加拿大那边冷不冷啊"),
    ("哲学思考", "你觉得孤独是好事还是坏事"),
    ("朋友关心", "你今天吃饭了吗"),
]

MODELS = {
    "qwen25": {
        "base": "/root/autodl-tmp/Qwen2.5-14B-Instruct",
        "lora": "/root/autodl-tmp/output-qwen25",
        "device": "cuda:0",
        "label": "Qwen2.5-14B",
    },
    "qwen3": {
        "base": "/root/autodl-tmp/Qwen3-14B",
        "lora": "/root/autodl-tmp/output-qwen3",
        "device": "cuda:1",
        "label": "Qwen3-14B",
    },
}

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen25"
    cfg = MODELS[model_name]

    print(f"\n===== {cfg['label']} =====")
    print("加载模型中...")
    tokenizer, model = load_model(cfg["base"], cfg["lora"], cfg["device"])
    print("模型加载完成!\n")

    for style, msg in TESTS:
        reply = chat(tokenizer, model, SYSTEM, msg)
        print(f"[{style}] 对方: {msg}")
        print(f"Joker: {reply}")
        print()
