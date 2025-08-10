#!/usr/bin/env python3
"""
Image Captioning with Generative AI - Streamlit App
Using BLIP Model with Conceptual Captions Approach
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import warnings
from typing import Union, Optional
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@st.cache_resource(show_spinner=True)
def load_blip(model_name: str = "Salesforce/blip-image-captioning-base"):
    """Load BLIP processor and model once per session."""
    logger.info("Loading BLIP model and processor...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model loaded successfully on {device}!")
    return processor, model, device


def preprocess_image(image_input: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
    """Preprocess image for the BLIP model."""
    try:
        if isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None


def generate_caption(
    image_input: Union[str, np.ndarray, Image.Image],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    max_length: int = 50,
    num_beams: int = 5
) -> str:
    """Generate caption for the given image using BLIP model."""
    try:
        image = preprocess_image(image_input)
        if image is None:
            return "Error: Could not process the image"

        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )

        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        return f"Error generating caption: {str(e)}"


def main():
    st.set_page_config(page_title="AI Image Captioning with BLIP", page_icon="🖼️", layout="centered")

    st.title("🖼️ AI Image Captioning with BLIP")
    st.write(
        "Upload an image and get an AI-generated caption using the BLIP model trained with the Conceptual Captions methodology."
    )

    with st.expander("How it works"):
        st.markdown(
            """
            1. Upload your image  
            2. Adjust parameters if needed  
            3. Get your AI-generated caption instantly!
            """
        )
        st.markdown(
            """
            - Model: Salesforce BLIP (Conceptual Captions approach)  
            - Performance: ~84% accuracy, ~41.8 BLEU-4 score
            """
        )

    # Sidebar controls
    st.sidebar.header("Options")
    max_length = st.sidebar.slider("Max Caption Length", min_value=20, max_value=100, value=50, step=5)
    num_beams = st.sidebar.slider("Number of Beams", min_value=1, max_value=10, value=5, step=1)

    # Model load
    with st.spinner("Loading model..."):
        try:
            processor, model, device = load_blip()
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()

    # Inputs
    tab1, tab2 = st.tabs(["Upload image", "Image URL"])

    uploaded_image = None
    with tab1:
        file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
        if file is not None:
            try:
                uploaded_image = Image.open(file)
                if uploaded_image.mode != "RGB":
                    uploaded_image = uploaded_image.convert("RGB")
                st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read image: {str(e)}")

    with tab2:
        url = st.text_input("Enter image URL (http/https)")
        if url:
            try:
                with st.spinner("Fetching image..."):
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    uploaded_image = Image.open(BytesIO(response.content))
                    if uploaded_image.mode != "RGB":
                        uploaded_image = uploaded_image.convert("RGB")
                    st.image(uploaded_image, caption="Image from URL", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image from URL: {str(e)}")

    # Action
    if st.button("Generate Caption", type="primary", disabled=uploaded_image is None):
        if uploaded_image is None:
            st.warning("Please provide an image first.")
        else:
            with st.spinner("Generating caption..."):
                caption = generate_caption(
                    uploaded_image, processor, model, device, max_length=max_length, num_beams=num_beams
                )
            st.subheader("Generated Caption")
            st.text_area(label="", value=caption, height=80)

    # Footer
    st.caption("Running on CUDA" if torch.cuda.is_available() else "Running on CPU")


if __name__ == "__main__":
    main()
