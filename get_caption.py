# get_caption.py
import os
import json
from src.utils import load_vlm_model, load_text_model

# Import your captioner class (adjust the import path as necessary)
from src.vlm_captioners import Llava_Flan_captioner

# Global variables to hold models and tokenizers
vl_model = None
vl_processor = None
txt_model = None
txt_tokenizer = None

# define ide
ide_type = ''
if 'google.colab' in str(get_ipython()):
  ide_type += 'CoLab'
elif 'vscode' in str(get_ipython()):
  ide_type += 'VS Code'
else:
  ide_type += 'Jupyter Notebook'
print(f"Running on {ide_type}")

def apply_special_corrections(final_caption, special_corrections):
    for pattern, replacement in special_corrections:
        final_caption = final_caption.replace(pattern, replacement)
    return final_caption

def load_instructions(instructions_dir, instructions_name='instructions_0.txt'):
    instructions_file = os.path.join(instructions_dir, instructions_name)
    if ide_type=="CoLab": instructions_file = "/content/VLM_Converter/"+instructions_file
    #print(instructions_file)
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
                     instructions_dir='instructions/inference', 
                     instructions_name='instructions_0.txt',
                     special_corrections=None, device='cuda'):
    
    global txt_model, txt_tokenizer, vl_model, vl_processor
    # Ensure appropriate models are loaded
    if image_path is not None:
        vl_model, vl_processor = load_vlm_model('llava-hf/llava-v1.6-mistral-7b-hf', device)
        txt_tokenizer, txt_model = load_text_model('google/flan-t5-large', device)
    elif input_text is not None:
        txt_tokenizer, txt_model = load_text_model('google/flan-t5-large', device)
    else:
        raise ValueError("Either 'input_text' or 'image_path' must be provided.")

    # Load instructions
    vision_instruction, text_instruction = load_instructions(instructions_dir, instructions_name)

    # Instantiate the captioner
    captioner = Llava_Flan_captioner(
        vlm_model=vl_model,
        processor=vl_processor,
        text_model=txt_model,
        text_tokenizer=txt_tokenizer,
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

    )
    
    # Apply special corrections
    if special_corrections:
        final_caption = apply_special_corrections(final_caption, special_corrections)

    return final_caption
