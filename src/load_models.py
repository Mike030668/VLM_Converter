
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Global variables to hold models and tokenizers
vl_model = None
vl_processor = None
txt_model = None
txt_tokenizer = None

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
        txt_tokenizer = T5Tokenizer.from_pretrained(text_model_name, legacy=False)
        txt_model = T5ForConditionalGeneration.from_pretrained(
            text_model_name
        ).to(device)
        print("Text model loaded.")
    else:
        print("Text model is already loaded.")

    return txt_tokenizer, txt_model
