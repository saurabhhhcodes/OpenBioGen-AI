#!/bin/bash

# OpenBioGen AI - Advanced Platform Deployment Script

echo "🧬 OpenBioGen AI - Advanced Platform Deployment"
echo "================================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not initialized. Please run:"
    echo "   git init"
    echo "   git add ."
    echo "   git commit -m 'Initial commit'"
    exit 1
fi

# Check if remote is set
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ Git remote not set. Please add your GitHub repository:"
    echo "   git remote add origin https://github.com/yourusername/OpenBioGen-AI-1.git"
    exit 1
fi

echo "✅ Git repository ready"

# Push to GitHub
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Deploy advanced OpenBioGen AI platform"
git push origin main

echo "✅ Code pushed to GitHub successfully!"
echo ""
echo "🚀 Deployment Options:"
echo ""
echo "1. Streamlit Cloud (Recommended):"
echo "   - Go to https://share.streamlit.io"
echo "   - Connect your GitHub repository"
echo "   - Set path: advanced_main.py"
echo "   - Add environment variables:"
echo "     TAVILY_API_KEY=your_key"
echo "     HUGGINGFACE_API_TOKEN=your_token"
echo ""
echo "2. Heroku:"
echo "   - heroku create your-app-name"
echo "   - heroku config:set TAVILY_API_KEY=your_key"
echo "   - heroku config:set HUGGINGFACE_API_TOKEN=your_token"
echo "   - git push heroku main"
echo ""
echo "3. Railway:"
echo "   - Connect GitHub repository to Railway"
echo "   - Set environment variables"
echo "   - Set start command: streamlit run advanced_main.py --server.port=\$PORT --server.address=0.0.0.0"
echo ""
echo "🎉 Your advanced OpenBioGen AI platform is ready for deployment!" 