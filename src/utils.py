from PIL import Image
import matplotlib.pyplot as plt
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


def display_plot(image, text, plot_size=(12, 6), max_font_size=20, min_font_size=12):
    """
    Displays the image with the caption text side by side.
    """
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=plot_size, gridspec_kw={'width_ratios': [1, 1]})

    # Display image on the left
    ax_img.imshow(image)
    ax_img.axis('off')  # Hide image axes

    # Display caption on the right with dynamic font sizing
    font_size = max_font_size
    text_box = ax_txt.text(0, 0.5, text, fontsize=font_size, ha='left', va='center', wrap=True)
    ax_txt.axis('off')  # Hide text axes

    # Adjust font size to fit within the subplot area
    while font_size > min_font_size:
        renderer = fig.canvas.get_renderer()
        bbox = text_box.get_window_extent(renderer=renderer)
        if bbox.width < ax_txt.get_window_extent().width and bbox.height < ax_txt.get_window_extent().height:
            break  # Text fits; exit loop
        font_size -= 1
        text_box.set_fontsize(font_size)  # Reduce font size if text is too large

    # Adjust spacing and show plot
    plt.subplots_adjust(wspace=0.1)  # Adjust space between image and text
    plt.show()

def show_imagecaption(image_path, caption_text, plot_size=(12, 6), max_font_size=20, min_font_size=12):
    """
    Uses the display_plot function to show the image with the provided caption.
    """
    # Load and resize image
    image = Image.open(image_path)
    display_plot(image, caption_text, plot_size, max_font_size, min_font_size)
