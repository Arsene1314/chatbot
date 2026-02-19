from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen3-14B")
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/models/Qwen3-14B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/model_output")
model.eval()

test_messages = [
    "今天好无聊啊 你在干嘛呀",
    "哈哈哈哈 我给你发了个搞笑视频 你看了吗",
    "晚安啦 明天见",
    "你今天吃了什么呀",
    "我想你了",
]

system_prompt = "请模仿我的说话风格和习惯来回复消息，不要说你是人工智能"

for msg in test_messages:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=True
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"\n{'='*50}")
    print(f"用户: {msg}")
    print(f"晴晴: {reply}")

print(f"\n{'='*50}")
print("===ALL_TESTS_DONE===")
