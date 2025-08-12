class ChatBot {
    constructor() {
        this.apiBase = window.location.origin; // For deployment flexibility
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.imageInput = document.getElementById('imageInput');
        this.sendButton = document.getElementById('sendButton');
        this.settingsPanel = document.getElementById('settingsPanel');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        
        this.initializeEventListeners();
        this.initializeSettings();
    }

    initializeEventListeners() {
        // Send message
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Image upload
        this.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));

        // Settings
        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.settingsPanel.classList.toggle('open');
        });
        
        document.getElementById('closeSettings').addEventListener('click', () => {
            this.settingsPanel.classList.remove('open');
        });

        // Clear chat
        document.getElementById('clearBtn').addEventListener('click', () => this.clearChat());
    }

    initializeSettings() {
        const maxLengthSlider = document.getElementById('maxLength');
        const maxLengthValue = document.getElementById('maxLengthValue');
        const numBeamsSlider = document.getElementById('numBeams');
        const numBeamsValue = document.getElementById('numBeamsValue');

        maxLengthSlider.addEventListener('input', () => {
            maxLengthValue.textContent = maxLengthSlider.value;
        });

        numBeamsSlider.addEventListener('input', () => {
            numBeamsValue.textContent = numBeamsSlider.value;
        });
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.sendButton.disabled = true;

        try {
            const response = await fetch(`${this.apiBase}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            if (data.success) {
                this.addMessage(data.response, 'bot');
            } else {
                this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, I\'m having trouble connecting. Please try again later.', 'bot');
        } finally {
            this.sendButton.disabled = false;
        }
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.addMessage('Please upload a valid image file.', 'bot');
            return;
        }

        // Validate file size (5MB limit)
        if (file.size > 5 * 1024 * 1024) {
            this.addMessage('Image file is too large. Please upload an image smaller than 5MB.', 'bot');
            return;
        }

        this.showLoading(true);

        try {
            const base64Image = await this.fileToBase64(file);
            
            // Add user message with image
            this.addImageMessage(base64Image, 'user');

            // Get caption settings
            const maxLength = parseInt(document.getElementById('maxLength').value);
            const numBeams = parseInt(document.getElementById('numBeams').value);

            const response = await fetch(`${this.apiBase}/api/caption`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image,
                    max_length: maxLength,
                    num_beams: numBeams
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.addMessage(`I can see: ${data.caption}`, 'bot');
            } else {
                this.addMessage('Sorry, I couldn\'t process that image. Please try another one.', 'bot');
            }
        } catch (error) {
            console.error('Error processing image:', error);
            this.addMessage('Sorry, I encountered an error processing your image. Please try again.', 'bot');
        } finally {
            this.showLoading(false);
            // Clear the input
            this.imageInput.value = '';
        }
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = `<p>${text}</p>`;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addImageMessage(base64Image, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = `
            <p>Here's my image:</p>
            <img src="${base64Image}" alt="Uploaded image" style="max-width: 200px; border-radius: 10px; margin-top: 10px;">
        `;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    clearChat() {
        // Keep only the initial bot message
        const messages = this.chatMessages.querySelectorAll('.message');
        for (let i = 1; i < messages.length; i++) {
            messages[i].remove();
        }
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});

// Service worker registration for PWA (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}