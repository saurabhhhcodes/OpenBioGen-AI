"""
Minimal Streamlit App for Testing Deployment
This is a simplified version to help identify deployment issues.
"""

import streamlit as st
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('minimal_app.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Minimal Streamlit app for testing deployment."""
    try:
        logger.info("Starting minimal app...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Basic Streamlit UI
        st.set_page_config(
            page_title="Minimal Test App",
            page_icon="🧪",
            layout="centered"
        )
        
        st.title("🎯 Minimal Test App")
        st.write("This is a minimal test app to verify Streamlit deployment.")
        
        # Test basic functionality
        if st.button("Click me!"):
            st.success("Button clicked! Basic functionality works!")
            
        # Display environment information
        with st.expander("Show Environment Info"):
            st.subheader("System Information")
            st.write(f"Python: {sys.version}")
            st.write(f"Current directory: {os.getcwd()}")
            st.write(f"Files in directory: {os.listdir('.')}")
            
        logger.info("Minimal app initialized successfully")
        
    except Exception as e:
        error_msg = f"Error in minimal app: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"An error occurred: {error_msg}")
        raise

if __name__ == "__main__":
    main()
