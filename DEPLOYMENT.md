# Deployment Guide

## Required GitHub Secrets

To deploy the application, you need to set up the following secrets in your GitHub repository:

### Setting up GitHub Secrets

1. Go to your repository on GitHub
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret" and add the following:

#### RENDER_SERVICE_ID
- **Description**: Your Render service ID for the backend deployment
- **How to get**: 
  1. Create a new Web Service on [Render](https://render.com)
  2. Connect your GitHub repository
  3. Copy the Service ID from the service URL or settings

#### RENDER_API_KEY
- **Description**: Your Render API key for automated deployments
- **How to get**:
  1. Go to [Render Dashboard](https://dashboard.render.com)
  2. Navigate to Settings → API Keys
  3. Create a new API key
  4. Copy the key value

## Deployment Process

### Automatic Deployment (Recommended)

1. **Push to main branch**: Any push to the main branch triggers automatic deployment
2. **Frontend**: Automatically deploys to GitHub Pages
3. **Backend**: Automatically deploys to Render

### Manual Deployment

#### Backend (Render)
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following configuration:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
   - **Environment**: Python 3.9

#### Frontend (GitHub Pages)
1. Go to repository Settings → Pages
2. Select "Deploy from a branch"
3. Choose "gh-pages" branch
4. The frontend will be available at: `https://[username].github.io/[repository-name]`

## Configuration Notes

- The frontend automatically detects the backend URL in production
- CORS is configured to allow requests from GitHub Pages
- Static files are properly served by Flask
- The application uses gunicorn for production deployment

## Testing Deployment

After deployment, test the following:
1. Frontend loads at GitHub Pages URL
2. Backend health endpoint responds at Render URL
3. Chat functionality works
4. Image upload and captioning works (when ML dependencies are available)

## Troubleshooting

### Common Issues
1. **502/503 errors**: Render service may be starting up (cold start)
2. **CORS errors**: Check that the backend URL is correctly configured
3. **Image upload fails**: Ensure the backend has sufficient memory for ML models

### Logs
- **Render logs**: Available in Render dashboard → Service → Logs
- **GitHub Actions logs**: Available in repository → Actions tab