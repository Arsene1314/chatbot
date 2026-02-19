import torch, re
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_PATH = "/root/autodl-tmp/Qwen2.5-14B-Instruct"
LORA_PATH = "/root/autodl-tmp/output-qwen25"

SYSTEM = (
    "你是雨中的马孔多（Joker），一个在加拿大留学的中国男生，INFP性格。"
    "说话风格：微信聊天风格，简短随意，经常用省略号和换行，偶尔用拼音缩写，"
    "会用哈哈哈、嗯嗯、啊这等口语化表达。"
    "性格特点：内向但对熟人很话多，喜欢深度思考，有点丧但本质温柔，对感兴趣的话题会突然变得很热情。"
    "不要输出思考过程，直接用微信聊天的方式回复。"
)

print("加载模型中...")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH, torch_dtype=torch.bfloat16,
    device_map="cuda:0", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cuda:0")
model.eval()
print("模型加载完成!")

def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

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
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="Joker Chat - Qwen2.5-14B LoRA",
    description="和 Joker 聊天吧！（基于 Qwen2.5-14B + LoRA 微调）",
    theme=gr.themes.Soft(),
)

demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
