"""
Minimal Streamlit App for Testing
This is the simplest possible Streamlit app to test deployment.
"""

import streamlit as st
import sys
import os

# Simple app
try:
    st.title("🚀 Streamlit Test App")
    st.write("This is a minimal test app to verify Streamlit deployment.")
    
    # Display system info
    st.write("### System Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Current directory: {os.getcwd()}")
    
    # List files in current directory
    try:
        files = os.listdir('.')
        st.write("### Files in current directory:")
        st.code('\n'.join(files))
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")
    
    # Test button
    if st.button("Click me!"):
        st.balloons()
        st.success("🎉 It works!")
        
    st.write("### Environment Variables")
    st.json(dict(os.environ))
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    raise
