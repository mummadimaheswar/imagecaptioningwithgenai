#!/bin/bash

# Local development server script
echo "🚀 Starting Image Captioning Chatbot..."
echo "📦 Installing dependencies..."

# Install dependencies
pip install -r requirements.txt

echo "🌐 Starting Flask server..."
echo "💡 Open http://localhost:5000 in your browser"
echo "⏹️  Press Ctrl+C to stop the server"

# Start the Flask app
python app.py