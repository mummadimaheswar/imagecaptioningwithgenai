# imagecaptioningwithgenai
# 🖼️ Image Captioning with Generative AI

This project implements an interactive Image Captioning Web App using the BLIP model (`Salesforce/blip-image-captioning-base`) from Hugging Face Transformers. The application allows users to generate human-like captions for:

- A single uploaded image
- A batch of images from a folder

The interface is built using Gradio, making it easy to use through a web browser with no coding required.
## 🚀 Features
✅ Caption a single uploaded image  
✅ Automatically generate captions for all images in a folder  
✅ Built-in error handling for corrupted or unsupported images  
✅ Clean, user-friendly interface with Gradio Tabs  
✅ Easily extendable for additional vision-language models  
 How to Use the App
➤ Single Image Captioning
Upload a single image (JPG, PNG, etc.).
The app will generate and display a caption.
➤ Dataset Folder Captioning
Enter a folder path (e.g., ./dataset) containing multiple images.
The app will return captions for each image in that folder.
