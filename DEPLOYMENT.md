# OpenBioGen-AI Deployment Guide

## Streamlit Community Cloud Deployment

### Prerequisites
1. GitHub account
2. Push this repository to GitHub
3. Sign up for Streamlit Community Cloud (free)

### Deployment Steps
1. Go to https://share.streamlit.io/
2. Connect your GitHub account
3. Select this repository
4. Set main file: `advanced_main.py`
5. Deploy!

### Environment Variables (Optional)
Add these to Streamlit Community Cloud secrets:
- `TAVILY_API_KEY`: For enhanced literature search
- `HUGGINGFACE_TOKEN`: For enhanced AI models

### Features
- ✅ Global database integration (PubChem, UniProt, KEGG, etc.)
- ✅ Gene-disease association analysis
- ✅ Compound analysis with drug-likeness predictions
- ✅ Network analysis with protein interactions
- ✅ Advanced memory system
- ✅ Clinical decision support
- ✅ Export capabilities
