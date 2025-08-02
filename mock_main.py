"""
OpenBioGen AI - Simplified Demo Version
Gene-Disease Association Prediction System (Mock Version)
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockTavilyRetriever:
    """Mock retriever that simulates Tavily search for gene-disease information"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        # Mock database of gene-disease associations
        self.mock_data = {
            "BRCA1": {
                "breast cancer": {
                    "confidence": "High",
                    "sources": [
                        {"title": "BRCA1 mutations in breast cancer", "url": "https://pubmed.ncbi.nlm.nih.gov/mock1"},
                        {"title": "Hereditary breast cancer and BRCA1", "url": "https://omim.org/mock1"}
                    ],
                    "prediction": "Strong association - BRCA1 mutations significantly increase breast cancer risk"
                },
                "ovarian cancer": {
                    "confidence": "High", 
                    "sources": [
                        {"title": "BRCA1 and ovarian cancer risk", "url": "https://pubmed.ncbi.nlm.nih.gov/mock2"}
                    ],
                    "prediction": "Strong association - BRCA1 mutations increase ovarian cancer risk"
                }
            },
            "TP53": {
                "lung cancer": {
                    "confidence": "Medium",
                    "sources": [
                        {"title": "TP53 mutations in lung cancer", "url": "https://pubmed.ncbi.nlm.nih.gov/mock3"}
                    ],
                    "prediction": "Moderate association - TP53 mutations found in some lung cancers"
                }
            },
            "APOE": {
                "alzheimer's disease": {
                    "confidence": "High",
                    "sources": [
                        {"title": "APOE and Alzheimer's disease", "url": "https://pubmed.ncbi.nlm.nih.gov/mock4"}
                    ],
                    "prediction": "Strong association - APOE4 variant increases Alzheimer's risk"
                }
            }
        }
    
    def search(self, gene: str, disease: str) -> List[Dict]:
        """Mock search that returns predefined results"""
        gene_upper = gene.upper()
        disease_lower = disease.lower()
        
        if gene_upper in self.mock_data:
            for stored_disease, data in self.mock_data[gene_upper].items():
                if disease_lower in stored_disease.lower() or stored_disease.lower() in disease_lower:
                    return data["sources"]
        
        # Return generic mock results if no specific match
        return [
            {"title": f"Research on {gene} and {disease}", "url": "https://pubmed.ncbi.nlm.nih.gov/generic"},
            {"title": f"Clinical studies: {gene}-{disease} association", "url": "https://omim.org/generic"}
        ]

class MockOpenBioGenAI:
    """Simplified version of OpenBioGen AI for demonstration"""
    
    def __init__(self, tavily_api_key: str = None):
        self.tavily_api_key = tavily_api_key
        self.retriever = MockTavilyRetriever(tavily_api_key)
        
    def predict_association(self, gene: str, disease: str) -> Dict[str, Any]:
        """Predict gene-disease association using mock data"""
        try:
            # Get mock search results
            sources = self.retriever.search(gene, disease)
            
            # Check if we have specific mock data
            gene_upper = gene.upper()
            disease_lower = disease.lower()
            
            prediction = "Unknown association"
            confidence = "Low"
            
            if gene_upper in self.retriever.mock_data:
                for stored_disease, data in self.retriever.mock_data[gene_upper].items():
                    if disease_lower in stored_disease.lower() or stored_disease.lower() in disease_lower:
                        prediction = data["prediction"]
                        confidence = data["confidence"]
                        break
            
            if prediction == "Unknown association":
                # Generate a generic prediction
                prediction = f"Based on available literature, {gene} may have a potential association with {disease}. Further research is needed to establish definitive links."
                confidence = "Low"
            
            return {
                "gene": gene.upper(),
                "disease": disease.title(),
                "prediction": prediction,
                "confidence": confidence,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                "gene": gene.upper(),
                "disease": disease.title(),
                "prediction": f"Error occurred during analysis: {str(e)}",
                "confidence": "Error",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_predict(self, gene_disease_pairs: List[tuple]) -> List[Dict[str, Any]]:
        """Predict associations for multiple gene-disease pairs"""
        results = []
        for gene, disease in gene_disease_pairs:
            result = self.predict_association(gene, disease)
            results.append(result)
        return results

def create_sample_data() -> pd.DataFrame:
    """Create sample gene-disease pairs for testing"""
    data = [
        ("BRCA1", "breast cancer"),
        ("BRCA1", "ovarian cancer"),
        ("TP53", "lung cancer"),
        ("APOE", "Alzheimer's disease"),
        ("HTT", "Huntington's disease"),
        ("SOD1", "ALS")
    ]
    return pd.DataFrame(data, columns=['gene', 'disease'])

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="🧬 OpenBioGen AI - Demo",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 OpenBioGen AI - Gene-Disease Association Predictor")
    st.markdown("**Demo Version** - Simplified system for demonstration purposes")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Check for Tavily API key
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        st.sidebar.warning("⚠️ No Tavily API key found. Using mock data for demonstration.")
        tavily_key = "demo_key"
    else:
        st.sidebar.success("✅ Tavily API key configured")
    
    # Initialize the system
    try:
        bio_ai = MockOpenBioGenAI(tavily_key)
        st.sidebar.success("✅ System initialized successfully")
    except Exception as e:
        st.sidebar.error(f"❌ System initialization failed: {str(e)}")
        st.stop()
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📋 Sample Data"])
    
    with tab1:
        st.header("Single Gene-Disease Association Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gene = st.text_input("Gene Symbol", placeholder="e.g., BRCA1, TP53, APOE")
        
        with col2:
            disease = st.text_input("Disease Name", placeholder="e.g., breast cancer, Alzheimer's disease")
        
        if st.button("🔬 Predict Association", type="primary"):
            if gene and disease:
                with st.spinner("Analyzing gene-disease association..."):
                    result = bio_ai.predict_association(gene, disease)
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gene", result["gene"])
                with col2:
                    st.metric("Disease", result["disease"])
                with col3:
                    confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴", "Error": "⚫"}
                    st.metric("Confidence", f"{confidence_color.get(result['confidence'], '⚪')} {result['confidence']}")
                
                st.markdown("### 🧬 Prediction")
                st.info(result["prediction"])
                
                if result["sources"]:
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"{i}. [{source['title']}]({source['url']})")
                
            else:
                st.warning("⚠️ Please enter both gene symbol and disease name.")
    
    with tab2:
        st.header("Batch Gene-Disease Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file with gene-disease pairs", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'gene' in df.columns and 'disease' in df.columns:
                    st.success(f"✅ Loaded {len(df)} gene-disease pairs")
                    st.dataframe(df.head())
                    
                    if st.button("🔬 Analyze All Pairs", type="primary"):
                        with st.spinner("Analyzing all gene-disease pairs..."):
                            pairs = [(row['gene'], row['disease']) for _, row in df.iterrows()]
                            results = bio_ai.batch_predict(pairs)
                        
                        # Display results
                        results_df = pd.DataFrame(results)
                        st.subheader("📊 Batch Analysis Results")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"biogen_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ CSV must contain 'gene' and 'disease' columns")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        
        # Manual batch input
        st.markdown("### Manual Batch Input")
        if st.button("📋 Use Sample Data"):
            sample_df = create_sample_data()
            st.session_state['batch_data'] = sample_df
        
        if 'batch_data' in st.session_state:
            st.dataframe(st.session_state['batch_data'])
            
            if st.button("🔬 Analyze Sample Data", type="primary"):
                with st.spinner("Analyzing sample data..."):
                    pairs = [(row['gene'], row['disease']) for _, row in st.session_state['batch_data'].iterrows()]
                    results = bio_ai.batch_predict(pairs)
                
                results_df = pd.DataFrame(results)
                st.subheader("📊 Sample Data Results")
                st.dataframe(results_df)
    
    with tab3:
        st.header("📋 Sample Gene-Disease Pairs")
        st.markdown("Here are some example gene-disease associations you can test:")
        
        sample_df = create_sample_data()
        st.dataframe(sample_df)
        
        st.markdown("### 🧬 Gene Information")
        st.markdown("""
        - **BRCA1**: Breast cancer gene 1, associated with hereditary breast and ovarian cancer
        - **TP53**: Tumor protein p53, known as the "guardian of the genome"
        - **APOE**: Apolipoprotein E, associated with Alzheimer's disease risk
        - **HTT**: Huntingtin gene, mutations cause Huntington's disease
        - **SOD1**: Superoxide dismutase 1, associated with ALS
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🧬 **OpenBioGen AI Demo** - Powered by mock data for demonstration purposes")

if __name__ == "__main__":
    main()
