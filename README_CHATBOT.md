# 🤖 AI Image Captioning Chatbot

An interactive web-based chatbot that generates AI-powered captions for uploaded images using the BLIP (Bootstrapping Language-Image Pre-training) model.

## 🚀 Features

✅ **Interactive Chatbot Interface** - Natural conversation with AI assistant  
✅ **Image Upload & Captioning** - Drag & drop or select images for instant AI analysis  
✅ **Real-time Processing** - Fast image captioning using state-of-the-art BLIP model  
✅ **Customizable Settings** - Adjust caption length and generation quality  
✅ **Responsive Design** - Works perfectly on desktop and mobile devices  
✅ **Auto-Deployment** - Automatic deployment to GitHub Pages and Render  

## 🌐 Live Demo

- **Frontend**: [GitHub Pages Deployment](https://mummadimaheswar.github.io/imagecaptioningwithgenai/)
- **Backend API**: [Render Deployment](https://imagecaptioningwithgenai.onrender.com/)

## 🛠️ Quick Start

### Option 1: Use the Live Demo
Simply visit the live demo link above and start chatting with the AI!

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/mummadimaheswar/imagecaptioningwithgenai.git
   cd imagecaptioningwithgenai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 🔧 Deployment

The project includes automatic deployment workflows:

### GitHub Pages (Frontend)
- Frontend automatically deploys to GitHub Pages on push to `main`
- No additional configuration needed

### Render (Backend)
To deploy the backend to Render:

1. **Create a Render account** at [render.com](https://render.com)

2. **Set up GitHub secrets**:
   - Go to your repo → Settings → Secrets and variables → Actions
   - Add these secrets:
     - `RENDER_SERVICE_ID`: Your Render service ID
     - `RENDER_API_KEY`: Your Render API key

3. **Push to main branch** - Deployment happens automatically!

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │   BLIP Model    │
│   (GitHub Pages)│◄──►│   (Render)      │◄──►│   (Hugging Face)│
│   HTML/CSS/JS   │    │   Python/Flask  │    │   AI Captioning │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
.
├── app.py                 # Flask API backend
├── image_captioning.py    # Original Streamlit app (legacy)
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── static/               # Frontend files
│   ├── index.html        # Main chatbot interface
│   ├── style.css         # Styling
│   └── script.js         # JavaScript functionality
├── .github/workflows/    # GitHub Actions
│   └── deploy.yml        # Auto-deployment workflow
└── README.md            # This file
```

## 🎯 How to Use

1. **Start a conversation** - Type a greeting or question
2. **Upload an image** - Click the image icon and select a photo
3. **Get AI captions** - The AI will analyze and describe your image
4. **Adjust settings** - Fine-tune caption length and quality
5. **Continue chatting** - Ask questions about image captioning

## 🔧 Configuration

### Caption Settings
- **Max Length**: 20-100 characters (default: 50)
- **Generation Quality**: 1-10 beams (default: 5)

### Environment Variables (for deployment)
- `PORT`: Server port (default: 5000)
- `RENDER_SERVICE_ID`: Render service identifier
- `RENDER_API_KEY`: Render API key for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Salesforce BLIP Model** - For the amazing image captioning capabilities
- **Hugging Face Transformers** - For the easy-to-use model interface
- **GitHub Actions & Render** - For seamless deployment automation

---

**Made with ❤️ for AI enthusiasts and developers**