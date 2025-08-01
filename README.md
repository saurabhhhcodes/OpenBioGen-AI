# 🧬 OpenBioGen AI

**Gene-Disease Association Prediction using LangChain & Open-Source LLMs**

OpenBioGen AI is an advanced bioinformatics system that predicts gene-disease associations by combining the power of LangChain orchestration, open-source language models, and scientific literature search through Tavily.

## 🌟 Features

- **LangChain Integration**: Sophisticated AI workflow orchestration
- **Open-Source LLMs**: Uses HuggingFace transformers (Microsoft DialoGPT and others)
- **Scientific Literature Search**: Powered by Tavily API with access to PubMed, OMIM, and GeneCards
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Batch Processing**: Analyze multiple gene-disease pairs simultaneously
- **Confidence Scoring**: AI-driven confidence assessment for predictions
- **Source Attribution**: Full traceability to scientific literature

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Tavily API key (get free at [tavily.com](https://tavily.com/))
- Optional: HuggingFace API token for enhanced model access

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/saurabh/CascadeProjects/OpenBioGen-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Tavily API key
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

5. **Open your browser** and navigate to the provided local URL (typically `http://localhost:8501`)

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
TAVILY_API_KEY=your_tavily_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here  # Optional
DEFAULT_LLM_MODEL=microsoft/DialoGPT-medium
MAX_SEARCH_RESULTS=5
CONFIDENCE_THRESHOLD=0.7
```

### Supported Models

The system supports various open-source models from HuggingFace:

- `microsoft/DialoGPT-medium` (default)
- `microsoft/DialoGPT-large`
- `gpt2`
- `distilgpt2`
- Custom models compatible with HuggingFace transformers

## 📊 Usage

### Single Prediction

1. Enter a gene symbol (e.g., `BRCA1`, `TP53`, `APOE`)
2. Enter a disease name (e.g., `breast cancer`, `Alzheimer's disease`)
3. Click "Predict Association"
4. View the AI-generated analysis with confidence score and sources

### Batch Prediction

1. Prepare a CSV file with `gene` and `disease` columns
2. Upload the file in the "Batch Prediction" tab
3. Click "Run Batch Prediction"
4. Download results as CSV

### Example CSV Format

```csv
gene,disease
BRCA1,breast cancer
BRCA2,ovarian cancer
TP53,Li-Fraumeni syndrome
APOE,Alzheimer's disease
```

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
python test_system.py
```

This will test:
- Gene and disease validation
- Sample data creation
- Basic system initialization
- Core functionality

## 🏗️ Architecture

### Core Components

1. **TavilyRetriever**: Custom LangChain retriever for scientific literature
2. **OpenBioGenAI**: Main prediction engine
3. **Streamlit Interface**: Web-based user interface
4. **Utility Functions**: Validation, processing, and formatting tools

### Workflow

1. **Input Validation**: Validate gene symbols and disease names
2. **Literature Search**: Query scientific databases via Tavily
3. **Context Preparation**: Process and summarize retrieved documents
4. **LLM Analysis**: Generate predictions using open-source language models
5. **Result Formatting**: Present results with confidence scores and sources

### Data Sources

- **PubMed**: Scientific literature and research papers
- **OMIM**: Online Mendelian Inheritance in Man database
- **GeneCards**: Comprehensive gene database
- **Additional curated biomedical resources**

## 📈 Performance

- **Accuracy**: Depends on available literature and model quality
- **Speed**: ~5-15 seconds per prediction (varies with model size)
- **Scalability**: Supports batch processing of multiple predictions
- **Reliability**: Fallback mechanisms for robust operation

## 🔒 Privacy & Security

- **API Keys**: Stored securely in environment variables
- **Data Processing**: No user data stored permanently
- **Literature Access**: Read-only access to public scientific databases
- **Local Processing**: LLM inference can run locally

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is open-source. Please ensure compliance with:
- HuggingFace model licenses
- Tavily API terms of service
- Scientific database usage policies

## 🆘 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure Tavily API key is correctly set in `.env`
2. **Model Loading Issues**: Check internet connection and HuggingFace access
3. **Memory Errors**: Use smaller models or reduce batch sizes
4. **Search Failures**: Verify Tavily API quota and network connectivity

### Getting Help

- Check the test script output for diagnostic information
- Review logs for detailed error messages
- Ensure all dependencies are correctly installed
- Verify API keys and network connectivity

## 🔮 Future Enhancements

- Support for additional open-source models
- Integration with more biological databases
- Advanced confidence scoring algorithms
- Real-time model fine-tuning capabilities
- API endpoint for programmatic access
- Enhanced visualization and reporting

## 📚 References

- [LangChain Documentation](https://docs.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Tavily Search API](https://tavily.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**OpenBioGen AI** - Advancing gene-disease association research through AI 🧬✨
