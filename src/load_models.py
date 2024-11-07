import os
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from llava import LlavaNextProcessor, LlavaNextForConditionalGeneration

def load_vlm_model(vlm_model_name, device):
    global vl_model, vl_processor
    if vl_model is None or vl_processor is None:
        print("Loading VLM model and processor...")
        vl_processor = LlavaNextProcessor.from_pretrained(vlm_model_name)
        # Set required attributes
        vl_processor.patch_size = 14  # Replace with the correct value
        vl_processor.vision_feature_select_strategy = 'patch'  # Replace with the correct value
        vl_model = LlavaNextForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        print("VLM model loaded.")
    else:
        print("VLM model is already loaded.")

    return vl_model, vl_processor



def load_text_model(text_model_name, device):
    global txt_model, txt_tokenizer
    if txt_model is None or txt_tokenizer is None:
        print("Loading text model and tokenizer...")
        txt_tokenizer = T5Tokenizer.from_pretrained(text_model_name)
        txt_model = T5ForConditionalGeneration.from_pretrained(
            text_model_name
        ).to(device)
        print("Text model loaded.")
    else:
        print("Text model is already loaded.")

    return txt_tokenizer, txt_model


def load_instructions(instructions_dir, instructions_name='instructions_0'):
    instructions_file = os.path.join(instructions_dir, instructions_name)
    if not os.path.exists(instructions_file):
        raise FileNotFoundError(f"Instructions file not found: {instructions_file}")

    if instructions_file.lower().endswith('.json'):
        with open(instructions_file, 'r', encoding='utf-8') as f:
            instructions = json.load(f)
        vision_instruction = instructions.get('vision_instruction', '')
        text_instruction = instructions.get('text_instruction', '')
    elif instructions_file.lower().endswith('.txt'):
        with open(instructions_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # Assuming the instructions are separated by '===VISION===' and '===TEXT==='
        vision_marker = '===VISION==='
        text_marker = '===TEXT==='
        if vision_marker in content and text_marker in content:
            vision_instruction = content.split(vision_marker)[1].split(text_marker)[0].strip()
            text_instruction = content.split(text_marker)[1].strip()
        else:
            raise ValueError("Instructions file format is incorrect.")
    else:
        raise ValueError("Instructions file must be a .json or .txt file.")

    return vision_instruction, text_instruction