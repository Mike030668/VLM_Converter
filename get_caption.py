# get_caption.py
import os
import json
from src.load_models import load_vlm_model, load_text_model

# Import your captioner class (adjust the import path as necessary)
from src.vlm_captioners import Llava_Flan_captioner

# Global variables to hold models and tokenizers
vl_model = None
vl_processor = None
txt_model = None
txt_tokenizer = None


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

def generate_caption(input_text=None, image_path=None, character_name='sbercat',
                     instructions_dir='instructions/inference', instructions_name='instructions_0',
                     special_corrections=None, device='cuda'):
    # Ensure appropriate models are loaded
    if image_path is not None:
        load_vlm_model('llava-hf/llava-v1.6-mistral-7b-hf', device)
        load_text_model('google/flan-t5-large', device)
    elif input_text is not None:
        load_text_model('google/flan-t5-large', device)
    else:
        raise ValueError("Either 'input_text' or 'image_path' must be provided.")

    # Load instructions
    vision_instruction, text_instruction = load_instructions(instructions_dir, instructions_name)

    # Instantiate the captioner
    captioner = Llava_Flan_captioner(
        vlm_model=vl_model,
        processor=vl_processor,
        text_model=txt_model,
        tokenizer=txt_tokenizer,
        device=device
    )

    # Set instructions and character name
    captioner.vision_instruction = vision_instruction
    captioner.text_instruction = text_instruction
    captioner.main_object_replacement = character_name

    # Generate the caption
    final_caption = captioner.prepare_and_convert(
        input_prompt=input_text,
        image_path=image_path,
        main_object_replacement=character_name,
        special_corrections=special_corrections
    )

    return final_caption
