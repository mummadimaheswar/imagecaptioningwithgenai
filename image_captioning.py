!pip install gradio transformers torch pillow requests

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr
import numpy as np
import os
import requests

# Load processor and model from Hugging Face model hub
print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Gemini API details
GEMINI_API_ENDPOINT = "http://127.0.0.1:7860/"
GEMINI_API_KEY = ""

def ask_gemini(user_message, chat_history):
    """
    Calls Google Gemini API for chat responses.
    """
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {"parts": [{"text": user_message}]}
        ]
    }
    url = f"{GEMINI_API_ENDPOINT}?key={GEMINI_API_KEY}"
    try:
        response = requests.post(url, headers=headers, json=data)
        gemini_response = response.json()
        # Parse Gemini's response (adapt this if the response structure changes)
        return gemini_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error contacting Gemini API: {e}"

def caption_image(input_image: np.ndarray):
    try:
        image = Image.fromarray(input_image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return f"Caption: {caption}"
    except Exception as e:
        return f"Error generating caption: {e}"

def caption_dataset(folder_path):
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

def process_message(message, history, uploaded_file):
    """
    Use Gemini for general chat, BLIP for image/folder captioning.
    """
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)
            caption = caption_image(image_array)
            response = f"I can see you've uploaded an image! {caption}"
        except Exception as e:
            response = f"Sorry, I couldn't process the uploaded image. Error: {str(e)}"
    elif message.lower().startswith("/caption_folder"):
        parts = message.split(" ", 1)
        if len(parts) > 1:
            folder_path = parts[1].strip()
            response = f"Processing folder: {folder_path}\n\n{caption_dataset(folder_path)}"
        else:
            response = "Please provide a folder path. Usage: /caption_folder <path>"
    elif message.strip():  # For other chat messages, use Gemini
        response = ask_gemini(message, history)
    else:
        response = "Just upload an image or type a message to get started!"
    return response

with gr.Blocks(title="Image Captioning Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Image Captioning Chatbot")
    gr.Markdown("Chat with me, upload images for captions, or ask me anything!")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                placeholder="Hi! I'm your image captioning assistant. Upload an image or type 'help' to get started!"
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here or upload an image...",
                    container=False,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Image",
                file_types=["image"],
                type="filepath"
            )
            gr.Markdown("""
            ### 💡 Tips:
            - Upload any image to get a caption
            - Type `/caption_folder <path>` to process multiple images
            - Ask me about image captioning or anything!
            """)

    def respond(message, chat_history, uploaded_file):
        bot_message = process_message(message, chat_history, uploaded_file)
        chat_history.append((message, bot_message))
        return chat_history, ""

    def handle_file_upload(uploaded_file, chat_history):
        if uploaded_file is not None:
            bot_message = process_message("", chat_history, uploaded_file)
            user_message = f"[Uploaded image: {os.path.basename(uploaded_file)}]"
            chat_history.append((user_message, bot_message))
        return chat_history, None

    submit_btn.click(
        respond,
        inputs=[msg, chatbot, file_upload],
        outputs=[chatbot, msg]
    )
    msg.submit(
        respond,
        inputs=[msg, chatbot, file_upload],
        outputs=[chatbot, msg]
    )
    file_upload.upload(
        handle_file_upload,
        inputs=[file_upload, chatbot],
        outputs=[chatbot, file_upload]
    )

if __name__ == "__main__":
    print("Launching Image Captioning Chatbot with Gemini...")
    demo.launch()
