# image_captioning_app.py
# !pip install streamlit transformers torch pillow datasets requests

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit as st
import numpy as np
import os
import requests
from io import BytesIO
import pandas as pd
from datasets import load_dataset
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Image Captioning with AI - Conceptual Captions",
    page_icon="🖼️",
    layout="wide"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load and cache the BLIP model and processor"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Cache dataset loading
@st.cache_data
def load_conceptual_captions(subset="validation", num_samples=100):
    """Load Google's Conceptual Captions dataset"""
    try:
        # Load the dataset - using validation split as it's smaller for demo
        dataset = load_dataset("conceptual_captions", split=f"{subset}[:{num_samples}]")
        return dataset
    except Exception as e:
        st.error(f"Error loading Conceptual Captions dataset: {e}")
        return None

def download_image_from_url(url, timeout=10):
    """Download image from URL with error handling"""
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        return None

def caption_image(image, processor, model):
    """
    Takes a PIL image, generates and returns a caption.
    """
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50, num_beams=5)
        caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {e}"

def caption_local_dataset(folder_path, processor, model):
    """
    Takes a folder path, generates captions for all images inside, and returns them.
    """
    if not os.path.exists(folder_path):
        return "Error: Folder not found."
    
    captions = []
    valid_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_formats)]
    
    if not image_files:
        return "No valid images found in the folder."
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        status_text.text(f"Processing {filename}...")
        
        try:
            image = Image.open(image_path).convert("RGB")
            generated_caption = caption_image(image, processor, model)
            captions.append({
                'filename': filename,
                'generated_caption': generated_caption,
                'image_path': image_path
            })
        except Exception as e:
            captions.append({
                'filename': filename,
                'generated_caption': f"Error - {str(e)}",
                'image_path': image_path
            })
        
        # Update progress
        progress_bar.progress((i + 1) / len(image_files))
    
    status_text.text("Processing complete!")
    return captions

def process_conceptual_captions(dataset, processor, model, max_images=10):
    """Process Conceptual Captions dataset samples"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_count = 0
    for i, sample in enumerate(dataset):
        if processed_count >= max_images:
            break
            
        status_text.text(f"Processing image {processed_count + 1}/{max_images}...")
        
        # Download image from URL
        image = download_image_from_url(sample['image_url'])
        
        if image is not None:
            # Generate caption using BLIP
            generated_caption = caption_image(image, processor, model)
            
            results.append({
                'original_caption': sample['caption'],
                'generated_caption': generated_caption,
                'image_url': sample['image_url'],
                'image': image
            })
            processed_count += 1
        
        # Update progress
        progress_bar.progress(processed_count / max_images)
        time.sleep(0.1)  # Small delay to prevent rate limiting
    
    status_text.text(f"Processing complete! Successfully processed {processed_count} images.")
    return results

# Main app
def main():
    st.title("🖼️ Image Captioning with BLIP - Google Conceptual Captions")
    st.markdown("Compare BLIP-generated captions with Google's Conceptual Captions dataset, or process your own images.")
    
    # Load model with spinner
    with st.spinner("Loading BLIP model... This may take a moment."):
        processor, model = load_model()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📷 Single Image Captioning", 
        "🌐 Conceptual Captions Dataset", 
        "📁 Local Dataset Processing"
    ])
    
    with tab1:
        st.header("Single Image Captioning")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to generate a caption"
        )
        
        if uploaded_file is not None:
            # Display the image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Your uploaded image", use_column_width=True)
            
            with col2:
                st.subheader("Generated Caption")
                if st.button("Generate Caption", type="primary", key="single_image"):
                    with st.spinner("Generating caption..."):
                        caption = caption_image(image, processor, model)
                        st.success(f"**Generated Caption:** {caption}")
                        st.code(caption, language=None)
    
    with tab2:
        st.header("Google's Conceptual Captions Dataset")
        st.markdown("Process samples from Google's Conceptual Captions dataset and compare with BLIP-generated captions.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            dataset_split = st.selectbox(
                "Choose dataset split",
                ["validation", "train"],
                help="Validation split is smaller and faster to load"
            )
            
            num_samples = st.slider(
                "Number of samples to load",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="More samples take longer to load"
            )
        
        with col2:
            max_process = st.slider(
                "Max images to process",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Number of images to actually download and caption"
            )
        
        if st.button("Load & Process Conceptual Captions", type="primary", key="load_cc"):
            with st.spinner(f"Loading {num_samples} samples from Conceptual Captions dataset..."):
                dataset = load_conceptual_captions(dataset_split, num_samples)
            
            if dataset is not None:
                st.success(f"Loaded {len(dataset)} samples from Conceptual Captions!")
                
                with st.spinner(f"Processing {max_process} images..."):
                    results = process_conceptual_captions(dataset, processor, model, max_process)
                
                if results:
                    st.subheader("Results Comparison")
                    
                    for i, result in enumerate(results):
                        st.markdown(f"### Image {i+1}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(result['image'], caption=f"Image {i+1}", use_column_width=True)
                        
                        with col2:
                            st.markdown("**Original Conceptual Caption:**")
                            st.info(result['original_caption'])
                            
                            st.markdown("**BLIP Generated Caption:**")
                            st.success(result['generated_caption'])
                            
                            # Show similarity or differences
                            if result['original_caption'].lower() in result['generated_caption'].lower() or \
                               result['generated_caption'].lower() in result['original_caption'].lower():
                                st.markdown("🟢 *Captions show similarity*")
                            else:
                                st.markdown("🟡 *Captions show different perspectives*")
                        
                        st.markdown("---")
                    
                    # Create downloadable results
                    results_df = pd.DataFrame([{
                        'image_index': i+1,
                        'original_caption': r['original_caption'],
                        'generated_caption': r['generated_caption'],
                        'image_url': r['image_url']
                    } for i, r in enumerate(results)])
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="conceptual_captions_comparison.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.header("Local Dataset Processing")
        st.markdown("Process all images in a local folder and generate captions for each.")
        
        folder_path = st.text_input(
            "Enter path to dataset folder",
            placeholder="e.g., ./dataset or /path/to/your/images",
            help="Enter the full path to the folder containing your images"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            process_folder = st.button("Process Folder", type="primary", key="process_local")
        
        with col2:
            if folder_path and os.path.exists(folder_path):
                valid_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                image_count = len([f for f in os.listdir(folder_path) 
                                if f.lower().endswith(valid_formats)])
                st.info(f"Found {image_count} images in the folder")
            elif folder_path:
                st.error("Folder not found. Please check the path.")
        
        if process_folder and folder_path:
            if os.path.exists(folder_path):
                st.subheader("Processing Results")
                captions = caption_local_dataset(folder_path, processor, model)
                
                if isinstance(captions, list):
                    # Display results with images
                    for result in captions:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if os.path.exists(result['image_path']):
                                try:
                                    img = Image.open(result['image_path'])
                                    st.image(img, caption=result['filename'], use_column_width=True)
                                except:
                                    st.error(f"Could not load {result['filename']}")
                        
                        with col2:
                            st.markdown(f"**{result['filename']}**")
                            st.success(result['generated_caption'])
                        
                        st.markdown("---")
                    
                    # Create downloadable results
                    results_text = "\n".join([f"{r['filename']}: {r['generated_caption']}" for r in captions])
                    st.download_button(
                        label="Download Results as Text File",
                        data=results_text,
                        file_name="local_image_captions.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(captions)
            else:
                st.error("Please enter a valid folder path.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Salesforce BLIP model, Google Conceptual Captions dataset, and Streamlit*")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Google Conceptual Captions** is a dataset containing ~3.3M images paired with natural language captions.
        
        **BLIP** (Bootstrapping Language-Image Pre-training) is a vision-language model that can generate captions for images.
        
        Use this app to:
        - Compare BLIP captions with original Conceptual Captions
        - Generate captions for your own images
        - Process entire folders of images
        """)
        
        st.header("Tips")
        st.markdown("""
        - Start with small sample sizes to test
        - Some Conceptual Captions images may not load due to broken URLs
        - BLIP generates different but often semantically similar captions
        - Download results for further analysis
        """)

if __name__ == "__main__":
    main()
