"""
OpenBioGen AI - Gene-Disease Association Prediction System
A LangChain-based system using open-source LLMs and Tavily search
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from tavily import TavilyClient
import pandas as pd
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TavilyRetriever(BaseRetriever):
    """Custom retriever using Tavily search for gene-disease information"""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.client = TavilyClient(api_key=api_key)
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents using Tavily search"""
        try:
            # Search for gene-disease information
            search_results = self.client.search(
                query=f"gene disease association {query}",
                search_depth="advanced",
                max_results=5,
                include_domains=["pubmed.ncbi.nlm.nih.gov", "omim.org", "genecards.org"]
            )
            
            documents = []
            for result in search_results.get('results', []):
                doc = Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'score': result.get('score', 0)
                    }
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error in Tavily search: {e}")
            return []

class OpenBioGenAI:
    """Main class for gene-disease association prediction"""
    
    def __init__(self, tavily_api_key: str, model_name: str = "microsoft/DialoGPT-medium"):
        self.tavily_api_key = tavily_api_key
        self.model_name = model_name
        self.retriever = TavilyRetriever(tavily_api_key)
        self.llm = self._initialize_llm()
        self.prediction_chain = self._create_prediction_chain()
    
    def _initialize_llm(self):
        """Initialize the open-source LLM using HuggingFace"""
        try:
            from transformers import pipeline
            
            # Use a more suitable model for text generation
            model_name = "microsoft/DialoGPT-medium"  # You can change this to other models
            
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                device_map="auto" if os.getenv("CUDA_VISIBLE_DEVICES") else None
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Successfully initialized LLM: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Fallback to a simpler approach
            return None
    
    def _create_prediction_chain(self):
        """Create the LangChain for gene-disease association prediction"""
        
        prompt_template = """
        You are an expert bioinformatics AI assistant specializing in gene-disease associations.
        
        Based on the following scientific literature and database information:
        {context}
        
        Question: What is the association between gene "{gene}" and disease "{disease}"?
        
        Please provide a comprehensive analysis including:
        1. Direct associations (if any)
        2. Indirect associations through pathways
        3. Confidence level (High/Medium/Low)
        4. Supporting evidence from literature
        5. Potential therapeutic implications
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "gene", "disease"]
        )
        
        if self.llm:
            return LLMChain(llm=self.llm, prompt=prompt)
        return None
    
    def predict_association(self, gene: str, disease: str) -> Dict[str, Any]:
        """Predict gene-disease association"""
        try:
            # Retrieve relevant documents
            query = f"{gene} {disease} association"
            documents = self.retriever._get_relevant_documents(query, run_manager=None)
            
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content[:500]}..."
                for doc in documents[:3]  # Use top 3 documents
            ])
            
            # Generate prediction using LLM
            if self.prediction_chain:
                result = self.prediction_chain.run(
                    context=context,
                    gene=gene,
                    disease=disease
                )
            else:
                # Fallback analysis without LLM
                result = self._fallback_analysis(gene, disease, documents)
            
            return {
                "gene": gene,
                "disease": disease,
                "prediction": result,
                "sources": [doc.metadata.get('url', '') for doc in documents],
                "confidence": self._calculate_confidence(documents)
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                "gene": gene,
                "disease": disease,
                "prediction": f"Error occurred during prediction: {str(e)}",
                "sources": [],
                "confidence": "Low"
            }
    
    def _fallback_analysis(self, gene: str, disease: str, documents: List[Document]) -> str:
        """Fallback analysis when LLM is not available"""
        if not documents:
            return f"No scientific literature found for {gene} and {disease} association."
        
        analysis = f"Analysis of {gene} and {disease} association:\n\n"
        
        for i, doc in enumerate(documents[:3], 1):
            analysis += f"{i}. {doc.metadata.get('title', 'Research Finding')}\n"
            analysis += f"   Content: {doc.page_content[:200]}...\n"
            analysis += f"   Source: {doc.metadata.get('url', 'N/A')}\n\n"
        
        analysis += f"Based on {len(documents)} scientific sources, there appears to be documented research on the {gene}-{disease} association."
        
        return analysis
    
    def _calculate_confidence(self, documents: List[Document]) -> str:
        """Calculate confidence level based on available evidence"""
        if len(documents) >= 3:
            return "High"
        elif len(documents) >= 1:
            return "Medium"
        else:
            return "Low"
    
    def batch_predict(self, gene_disease_pairs: List[tuple]) -> List[Dict[str, Any]]:
        """Predict associations for multiple gene-disease pairs"""
        results = []
        for gene, disease in gene_disease_pairs:
            result = self.predict_association(gene, disease)
            results.append(result)
        return results

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="OpenBioGen AI",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 OpenBioGen AI")
    st.subtitle("Gene-Disease Association Prediction using LangChain & Open-Source LLMs")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key input
    tavily_api_key = st.sidebar.text_input(
        "Tavily API Key",
        type="password",
        help="Enter your Tavily API key for literature search"
    )
    
    if not tavily_api_key:
        st.warning("Please enter your Tavily API key in the sidebar to continue.")
        st.info("Get your free API key at: https://tavily.com/")
        return
    
    # Initialize the AI system
    try:
        ai_system = OpenBioGenAI(tavily_api_key)
        st.sidebar.success("✅ System initialized successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error initializing system: {e}")
        return
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "About"])
    
    with tab1:
        st.header("Single Gene-Disease Association Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gene = st.text_input("Gene Symbol", placeholder="e.g., BRCA1, TP53, APOE")
        
        with col2:
            disease = st.text_input("Disease Name", placeholder="e.g., breast cancer, Alzheimer's disease")
        
        if st.button("Predict Association", type="primary"):
            if gene and disease:
                with st.spinner("Analyzing gene-disease association..."):
                    result = ai_system.predict_association(gene, disease)
                
                st.subheader("Prediction Results")
                
                # Display confidence
                confidence_color = {
                    "High": "🟢",
                    "Medium": "🟡", 
                    "Low": "🔴"
                }
                st.write(f"**Confidence Level:** {confidence_color.get(result['confidence'], '⚪')} {result['confidence']}")
                
                # Display prediction
                st.write("**Analysis:**")
                st.write(result['prediction'])
                
                # Display sources
                if result['sources']:
                    st.write("**Sources:**")
                    for i, source in enumerate(result['sources'], 1):
                        if source:
                            st.write(f"{i}. {source}")
            else:
                st.error("Please enter both gene and disease names.")
    
    with tab2:
        st.header("Batch Gene-Disease Association Prediction")
        
        st.write("Upload a CSV file with 'gene' and 'disease' columns:")
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'gene' in df.columns and 'disease' in df.columns:
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    if st.button("Run Batch Prediction"):
                        gene_disease_pairs = list(zip(df['gene'], df['disease']))
                        
                        with st.spinner(f"Processing {len(gene_disease_pairs)} predictions..."):
                            results = ai_system.batch_predict(gene_disease_pairs)
                        
                        # Convert results to DataFrame
                        results_df = pd.DataFrame(results)
                        
                        st.subheader("Batch Prediction Results")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="gene_disease_predictions.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("CSV file must contain 'gene' and 'disease' columns.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.header("About OpenBioGen AI")
        
        st.markdown("""
        **OpenBioGen AI** is an advanced gene-disease association prediction system that combines:
        
        - 🔗 **LangChain**: For orchestrating complex AI workflows
        - 🤖 **Open-Source LLMs**: Using HuggingFace transformers
        - 🔍 **Tavily Search**: For retrieving relevant scientific literature
        - 📊 **Streamlit**: For an intuitive web interface
        
        ### Features:
        - Single and batch prediction capabilities
        - Literature-backed analysis
        - Confidence scoring
        - Source attribution
        - Downloadable results
        
        ### How it works:
        1. **Literature Search**: Tavily searches scientific databases for relevant papers
        2. **Context Preparation**: Retrieved documents are processed and summarized
        3. **LLM Analysis**: Open-source language model analyzes the evidence
        4. **Prediction**: System provides association prediction with confidence level
        
        ### Data Sources:
        - PubMed scientific literature
        - OMIM database
        - GeneCards database
        - Other curated biomedical resources
        """)

if __name__ == "__main__":
    main()
