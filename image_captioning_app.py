#image_captioning_app.py
!pip install gradio transformers torch pillow
# Import required libraries
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr
import numpy as np
import os

# Load processor and model from Hugging Face model hub
print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(input_image: np.ndarray):
    """
    Takes a numpy image input, generates and returns a caption.
    """
    try:
        image = Image.fromarray(input_image).convert("RGB")  # Convert to PIL image
        inputs = processor(images=image, return_tensors="pt")  # Preprocess
        outputs = model.generate(**inputs)  # Generate caption tokens
        caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode tokens to text
        return f"Caption: {caption}"
    except Exception as e:
        return f"Error generating caption: {e}"

def caption_dataset(folder_path):
    """
    Takes a folder path, generates captions for all images inside, and returns them.
    """
    if not os.path.exists(folder_path):
        return "Error: Folder not found."

    captions = ""
    valid_formats = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_formats):
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                outputs = model.generate(**inputs)
                caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                captions += f"{filename}: {caption}\n"
            except Exception as e:
                captions += f"{filename}: Error - {str(e)}\n"

    return captions or "No valid images found in the folder."

# Input components
image_input = gr.Image(type="numpy", label="Upload an image")
folder_input = gr.Textbox(
    label="Enter path to dataset folder (e.g., ./dataset)",
    placeholder="./dataset",
    lines=1
)

# Gradio layout with two tabs
with gr.Blocks() as demo:
    gr.Markdown("# *Image Captioning with Generative AI*")
    gr.Markdown("Upload a single image or process an entire folder using the BLIP model.")

    with gr.Tab("Single Image Captioning"):
        gr.Interface(fn=caption_image, inputs=image_input, outputs="text").render()

    with gr.Tab("Dataset Folder Captioning"):
        gr.Interface(fn=caption_dataset, inputs=folder_input, outputs="text").render()

# Launch the app
if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch()
