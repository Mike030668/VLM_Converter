import torch
# Import your models and classes
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_vlm_model(vlm_model_name, device):
    global vlm_model_loaded, vl_model, vl_processor
    if not vlm_model_loaded:
        user_input = input(f"Do you want to load the VLM model '{vlm_model_name}'? (Y/N): ")
        if user_input.strip().lower() == 'y':
            # Load VLM model and processor
            print("Loading VLM model and processor...")
            vl_processor = LlavaNextProcessor.from_pretrained(vlm_model_name)
            vl_model = LlavaNextForConditionalGeneration.from_pretrained(
                vlm_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
            vlm_model_loaded = True
            print("VLM model loaded.")
        else:
            print("VLM model loading skipped.")
    else:
        print("VLM model is already loaded.")

def load_text_model(text_model_name, device):
    global text_model_loaded, txt_model, txt_tokenizer
    if not text_model_loaded:
        user_input = input(f"Do you want to load the text model '{text_model_name}'? (Y/N): ")
        if user_input.strip().lower() == 'y':
            # Load text model and tokenizer
            print("Loading text model and tokenizer...")
            txt_tokenizer = T5Tokenizer.from_pretrained(text_model_name)
            txt_model = T5ForConditionalGeneration.from_pretrained(
                text_model_name
            ).to(device)
            text_model_loaded = True
            print("Text model loaded.")
        else:
            print("Text model loading skipped.")
    else:
        print("Text model is already loaded.")
