##!/usr/bin/env python

import os
import sys
import argparse
import random
from PIL import Image
from tqdm import tqdm
import torch
import gc
import json
import shutil  # Import shutil for file operations

# Import your models and classes
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import your captioner class (ensure it's properly imported or defined)
from src.vlm_captioners import Llava_Flan_captioner  # Update this import based on your actual file structure

def parse_special_corrections(corrections_str):
    corrections = []
    # Split the string by semicolons to get individual corrections
    pairs = corrections_str.split(';')
    for pair in pairs:
        if '=>' in pair:
            pattern, replacement = pair.split('=>', 1)
            corrections.append((pattern.strip(), replacement.strip()))
        else:
            print(f"Warning: Invalid correction format '{pair}'. Expected 'pattern=>replacement'.")
    return corrections


def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using VLMCaptioner.')
    parser.add_argument('--image_directory', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--output_directory', type=str, required=True, help='Directory to save caption text files.')
    parser.add_argument('--num_images', type=int, default=None, help='Number of random images to process for testing.')
    parser.add_argument('--character', '-c', type=str, required=True, help='Character name to use as main object replacement.')
    parser.add_argument('--instructions_file', type=str, required=True, help='Path to the file containing the instructions (JSON or TXT).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the models on (e.g., "cuda" or "cpu").')
    parser.add_argument('--special_corrections', type=str, default=None,
                        help='Special corrections to apply to the final caption. Format: "pattern1=>replacement1;pattern2=>replacement2"')

    args = parser.parse_args()

    image_directory = args.image_directory
    output_directory = args.output_directory
    num_images = args.num_images
    character = args.character
    instructions_file = args.instructions_file
    device = args.device

    # Parse special corrections
    special_corrections = None
    if args.special_corrections:
        special_corrections = parse_special_corrections(args.special_corrections)
        if special_corrections:
            print(f"Applying special corrections: {special_corrections}")
        else:
            print("No valid special corrections found.")


    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # List of image extensions to consider
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # Get the list of image files
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(image_extensions)]

    # If num_images is specified, select a random subset
    if num_images is not None and num_images < len(image_files):
        image_files = random.sample(image_files, num_images)

    # Load the models and tokenizer
    print("Loading models...")

    # Define your model names https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
    VLM_MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
    TEXT_MODEL_NAME = "google/flan-t5-large"

    #vl_model = AutoModelForCausalLM.from_pretrained(VLM_MODEL_NAME, torch_dtype=torch.float16).to(device)
    vl_model = LlavaNextForConditionalGeneration.from_pretrained(
    VLM_MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)
    
    #vl_processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME)
    vl_processor = LlavaNextProcessor.from_pretrained(VLM_MODEL_NAME)

    #txt_model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_NAME, torch_dtype=torch.float16).to(device)
    txt_model = T5ForConditionalGeneration.from_pretrained(TEXT_MODEL_NAME).to(device)

    #txt_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    txt_tokenizer = T5Tokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Instantiate the captioner with models and tokenizer
    captioner = Llava_Flan_captioner(
        vlm_model=vl_model,
        processor=vl_processor,
        text_model=txt_model,
        text_tokenizer=txt_tokenizer,
        device=device
    )

    # Set the main object replacement from the command-line argument
    captioner.main_object_replacement = character

    # Read the instructions from the file
    if instructions_file.lower().endswith('.json'):
        with open(instructions_file, 'r', encoding='utf-8') as f:
            instructions = json.load(f)
        captioner.vision_instruction = instructions.get('vision_instruction', '')
        captioner.text_instruction = instructions.get('text_instruction', '')
    elif instructions_file.lower().endswith('.txt'):
        with open(instructions_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # Assuming the instructions are separated by a special delimiter
        # e.g., ===VISION=== and ===TEXT===
        vision_marker = '===VISION==='
        text_marker = '===TEXT==='
        if vision_marker in content and text_marker in content:
            vision_instruction = content.split(vision_marker)[1].split(text_marker)[0].strip()
            text_instruction = content.split(text_marker)[1].strip()
            captioner.vision_instruction = vision_instruction
            captioner.text_instruction = text_instruction
        else:
            print("Error: Instructions file format is incorrect.")
            sys.exit(1)
    else:
        print("Error: Instructions file must be a .json or .txt file.")
        sys.exit(1)

    # Process each image
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_directory, filename)
        base_name, _ = os.path.splitext(filename)
        output_text_file = os.path.join(output_directory, base_name + '.txt')
        output_image_file = os.path.join(output_directory, filename)  # Copy the image file with the same name

        try:
            # Generate the caption
            caption = captioner.prepare_and_convert(
                image_path=image_path
            )

            # Save the caption to a text file
            with open(output_text_file, 'w', encoding='utf-8') as text_file:
                text_file.write(caption)

            # Copy the image file to the output directory
            shutil.copy(image_path, output_image_file)

            # Clear memory after each image
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    main()