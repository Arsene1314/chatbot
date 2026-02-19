"""
将 chat_samples 文件转换为 LLaMA Factory SFT 训练格式（sharegpt）
"""
import json
import re

def parse_chat_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除注释行
    lines = content.split('\n')
    
    conversations = []
    current_conv = []
    current_role = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        
        # 跳过注释和空行和编号行
        if line.startswith('//') or line == '':
            # 如果当前有未保存的对话，保存它
            if current_text and current_role:
                current_conv.append({
                    'role': current_role,
                    'text': '\n'.join(current_text)
                })
                current_text = []
                current_role = None
            if current_conv:
                conversations.append(current_conv)
                current_conv = []
            continue
        
        # 检查是否是编号行（如 "1." "2."）
        if re.match(r'^\d+\.$', line):
            if current_text and current_role:
                current_conv.append({
                    'role': current_role,
                    'text': '\n'.join(current_text)
                })
                current_text = []
                current_role = None
            if current_conv:
                conversations.append(current_conv)
                current_conv = []
            continue
        
        # 解析 Q: 和 A: 行
        if line.startswith('Q:') or line.startswith('Q：'):
            msg = line[2:].strip()
            if current_role == 'A':
                # 角色切换，保存之前的
                current_conv.append({
                    'role': 'A',
                    'text': '\n'.join(current_text)
                })
                current_text = [msg]
                current_role = 'Q'
            elif current_role == 'Q':
                current_text.append(msg)
            else:
                current_role = 'Q'
                current_text = [msg]
        elif line.startswith('A:') or line.startswith('A：'):
            msg = line[2:].strip()
            if current_role == 'Q':
                current_conv.append({
                    'role': 'Q',
                    'text': '\n'.join(current_text)
                })
                current_text = [msg]
                current_role = 'A'
            elif current_role == 'A':
                current_text.append(msg)
            else:
                current_role = 'A'
                current_text = [msg]
    
    # 保存最后一组
    if current_text and current_role:
        current_conv.append({
            'role': current_role,
            'text': '\n'.join(current_text)
        })
    if current_conv:
        conversations.append(current_conv)
    
    return conversations


def convert_to_sharegpt(conversations, system_prompt):
    """转换为 LLaMA Factory sharegpt 格式"""
    dataset = []
    
    for conv in conversations:
        if len(conv) < 2:
            continue
        
        messages = [{"from": "system", "value": system_prompt}]
        
        for turn in conv:
            if turn['role'] == 'Q':
                messages.append({"from": "human", "value": turn['text']})
            elif turn['role'] == 'A':
                messages.append({"from": "gpt", "value": turn['text']})
        
        # 确保对话以 human 开头，gpt 结尾，且交替出现
        # 过滤掉不符合格式的
        valid = True
        non_system = [m for m in messages if m['from'] != 'system']
        if len(non_system) < 2:
            valid = False
        elif non_system[0]['from'] != 'human':
            valid = False
        elif non_system[-1]['from'] != 'gpt':
            valid = False
        
        if valid:
            dataset.append({"conversations": messages})
    
    return dataset


def main():
    filepath = '/Users/joker/Downloads/chat_samples_副本.txt'
    
    system_prompt = "请模仿我的说话风格和习惯来回复消息，不要说你是人工智能"
    
    conversations = parse_chat_file(filepath)
    print(f"解析到 {len(conversations)} 组对话")
    
    dataset = convert_to_sharegpt(conversations, system_prompt)
    print(f"转换为 {len(dataset)} 条训练样本")
    
    # 保存为 JSON
    output_path = '/Users/joker/Desktop/deepseek_style_bot/chat_sft_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"已保存到 {output_path}")
    
    # 打印前两条样本预览
    for i, item in enumerate(dataset[:2]):
        print(f"\n--- 样本 {i+1} ---")
        for msg in item['conversations']:
            print(f"  [{msg['from']}]: {msg['value'][:80]}...")


if __name__ == '__main__':
    main()
