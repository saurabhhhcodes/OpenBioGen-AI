"""
Minimal Streamlit App for Deployment Testing
This is a simplified version to help identify deployment issues.
"""

import streamlit as st
import os
import sys
import logging

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main application function."""
    try:
        # Basic app configuration
        st.set_page_config(
            page_title="OpenBioGen AI - Test",
            page_icon="🧪",
            layout="centered"
        )
        
        st.title("OpenBioGen AI - Test Deployment")
        st.write("This is a test deployment to verify Streamlit Cloud functionality.")
        
        # Display system information
        with st.expander("System Information"):
            st.write(f"Python version: {sys.version}")
            st.write(f"Current directory: {os.getcwd()}")
            st.write(f"Files in directory: {os.listdir('.')}")
            
        # Test basic functionality
        if st.button("Test Button"):
            st.success("Success! Basic functionality is working.")
            
        logger.info("Test app initialized successfully")
        
    except Exception as e:
        error_msg = f"Error in test app: {str(e)}"
        logger.error(error_msg)
        st.error(f"An error occurred: {error_msg}")
        raise

if __name__ == "__main__":
    main()
