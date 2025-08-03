# OpenBioGen AI - Advanced Platform Deployment Guide

## 🚀 Deployment Options for Advanced Main Version

This guide covers deployment of the `advanced_main.py` version with all advanced features.

### 📋 Prerequisites

1. **Environment Variables** (set in your deployment platform):
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   ```

2. **Required Files**:
   - `advanced_main.py` (main application)
   - `advanced_core.py` (core features)
   - `performance_optimizer.py` (performance features)
   - `memory_system.py` (memory system)
   - `enhanced_analysis_engine.py` (analysis engine)
   - `global_database_integrator.py` (database integration)
   - `security_validator.py` (security features)
   - `enhanced_logging.py` (logging system)
   - `advanced_ui_components.py` (UI components)
   - `requirements-deploy.txt` (dependencies)
   - `.streamlit/config.toml` (Streamlit config)
   - `runtime.txt` (Python version)
   - `packages.txt` (system dependencies)
   - `Procfile` (deployment config)

## 🌐 Deployment Platforms

### 1. Streamlit Cloud (Recommended)

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set deployment path to: `advanced_main.py`
5. Add environment variables in the Streamlit Cloud dashboard
6. Deploy!

**Environment Variables in Streamlit Cloud:**
```
TAVILY_API_KEY=your_tavily_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

### 2. Heroku

**Steps:**
1. Install Heroku CLI
2. Create new Heroku app:
   ```bash
   heroku create your-app-name
   ```
3. Set environment variables:
   ```bash
   heroku config:set TAVILY_API_KEY=your_tavily_api_key_here
   heroku config:set HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   ```
4. Deploy:
   ```bash
   git add .
   git commit -m "Deploy advanced OpenBioGen AI"
   git push heroku main
   ```

### 3. Railway

**Steps:**
1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Set start command: `streamlit run advanced_main.py --server.port=$PORT --server.address=0.0.0.0`
4. Deploy!

### 4. Render

**Steps:**
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set build command: `pip install -r requirements-deploy.txt`
4. Set start command: `streamlit run advanced_main.py --server.port=$PORT --server.address=0.0.0.0`
5. Add environment variables in Render dashboard
6. Deploy!

### 5. Google Cloud Run

**Steps:**
1. Create Dockerfile:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements-deploy.txt .
   RUN pip install -r requirements-deploy.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "advanced_main.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Build and deploy:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/openbiogen-ai
   gcloud run deploy openbiogen-ai --image gcr.io/PROJECT_ID/openbiogen-ai --platform managed
   ```

## 🔧 Configuration

### Environment Variables
Set these in your deployment platform:

```bash
TAVILY_API_KEY=your_tavily_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

### Advanced Features Status
The deployment includes:
- ✅ Memory System (semantic, episodic, procedural)
- ✅ Performance Optimizer (caching, parallel processing)
- ✅ Enhanced Analysis Engine (multi-database integration)
- ✅ Global Database Integrator (UniProt, KEGG, Reactome, NCBI)
- ✅ Security Validator (input validation, auditing)
- ✅ Advanced UI Components (professional visualizations)

## 🚀 Quick Deploy Commands

### Streamlit Cloud (Fastest)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set path: `advanced_main.py`
5. Add environment variables
6. Deploy!

### Heroku
```bash
heroku create your-app-name
heroku config:set TAVILY_API_KEY=your_key
heroku config:set HUGGINGFACE_API_TOKEN=your_token
git push heroku main
```

## 📊 Monitoring

After deployment, monitor:
- Application logs for errors
- Performance metrics
- Memory usage
- API rate limits

## 🔒 Security Notes

- Environment variables are encrypted in deployment platforms
- Input validation is active
- Rate limiting is implemented
- Security auditing is enabled

## 🎉 Success!

Your advanced OpenBioGen AI platform will be available at your deployment URL with all features:
- Comprehensive gene-disease analysis
- Clinical decision support
- Network visualizations
- Batch processing
- Memory system
- Performance optimization
- Global database integration 