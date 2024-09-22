import json
import copy
from tqdm import tqdm
from templates import QuestionTemplateGenerator

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def generate_formatted_prompt(message: str):
    question_template = QuestionTemplateGenerator().generate()
    if message.startswith("<image>"):
        prompt = "<image>\n" + question_template.format(question=message.replace("<image>", "").strip())
    elif message.endswith("<image>"):
        prompt = question_template.format(question=message.replace("<image>", "").strip()) + "\n<image>"
    else:
        prompt = question_template.format(question=message.strip())
    return prompt

def process_conversations(data):
    for sample in tqdm(data):
        assert len(sample["conversations"]) % 2 == 0, "The conversation is incomplete."
        for i, item in enumerate(sample["conversations"]):
            if i % 2 == 0:
                item["value"] = generate_formatted_prompt(item["value"])
    return data

if __name__ == "__main__":
    input_path = './playground/data/llava_v1_5_mix665k.json'
    output_path = './playground/data/templated_llava_v1_5_mix665k.json'
    conversation_data = load_json(input_path)
    processed_data = process_conversations(copy.deepcopy(conversation_data))
    save_json(processed_data, output_path)
