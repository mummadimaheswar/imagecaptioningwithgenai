#!/usr/bin/env python3
"""
Flask API for Image Captioning Chatbot
Provides REST API endpoints for image captioning functionality
"""

import os
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import base64
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='static', static_folder='static')
CORS(app)

# Global variables for model caching
processor = None
model = None
device = None

@app.route('/style.css')
def serve_css():
    return send_from_directory('static', 'style.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('static', 'script.js')

def load_blip_model(model_name: str = "Salesforce/blip-image-captioning-base"):
    """Load BLIP processor and model once."""
    global processor, model, device
    
    if processor is None or model is None:
        logger.info("Loading BLIP model and processor...")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model loaded successfully on {device}!")
    
    return processor, model, device

def preprocess_image(image_input) -> Optional[Image.Image]:
    """Preprocess image for the BLIP model."""
    try:
        if isinstance(image_input, str):
            # Handle base64 encoded images
            if image_input.startswith('data:image'):
                header, data = image_input.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(BytesIO(image_data))
            elif image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_input)
        else:
            image = image_input

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def generate_caption(
    image_input,
    max_length: int = 50,
    num_beams: int = 5
) -> str:
    """Generate caption for the given image using BLIP model."""
    try:
        # Load model if not already loaded
        proc, mdl, dev = load_blip_model()
        
        image = preprocess_image(image_input)
        if image is None:
            return "Error: Could not process the image"

        inputs = proc(image, return_tensors="pt").to(dev)

        with torch.no_grad():
            generated_ids = mdl.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                pad_token_id=proc.tokenizer.eos_token_id
            )

        caption = proc.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        return f"Error generating caption: {str(e)}"

@app.route('/')
def index():
    """Serve the chatbot interface."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "image-captioning-chatbot"})

@app.route('/api/caption', methods=['POST'])
def caption_image():
    """Generate caption for uploaded image."""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        max_length = data.get('max_length', 50)
        num_beams = data.get('num_beams', 5)
        
        # Generate caption
        caption = generate_caption(image_data, max_length, num_beams)
        
        return jsonify({
            "caption": caption,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in caption endpoint: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot conversation."""
    try:
        data = request.get_json()
        message = data.get('message', '').lower()
        
        # Simple chatbot responses
        responses = {
            'hello': "Hello! I'm an AI image captioning assistant. Upload an image and I'll describe what I see!",
            'hi': "Hi there! I can help you caption images. Just upload one and I'll tell you what's in it.",
            'help': "I can generate captions for your images using AI. Simply upload an image and I'll describe what I see. You can also ask me about image captioning!",
            'what can you do': "I can analyze images and generate descriptive captions using advanced AI models. Upload any image and I'll describe what I see!",
            'how do you work': "I use the BLIP (Bootstrapping Language-Image Pre-training) model to understand images and generate natural language descriptions.",
            'thanks': "You're welcome! Feel free to upload more images for captioning.",
            'thank you': "You're welcome! Happy to help with image captioning.",
            'bye': "Goodbye! Come back anytime you need image captioning help!",
            'goodbye': "See you later! Don't forget to try uploading some images next time!"
        }
        
        # Find best match
        response = "I'm here to help with image captioning! Upload an image and I'll describe what I see. You can also ask me 'help' for more information."
        for key in responses:
            if key in message:
                response = responses[key]
                break
        
        return jsonify({
            "response": response,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    # Load model on startup
    try:
        load_blip_model()
        logger.info("Model preloaded successfully!")
    except Exception as e:
        logger.error(f"Failed to preload model: {str(e)}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)