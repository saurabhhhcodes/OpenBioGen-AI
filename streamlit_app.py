"""
Streamlit App Entry Point for OpenBioGen AI
This file serves as the main entry point for Streamlit Cloud deployment.
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Configure logging to both console and file
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def log_environment():
    """Log important environment information."""
    logger.info("=" * 80)
    logger.info(f"Starting OpenBioGen AI application")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Environment variables: {dict(os.environ)}")
    logger.info("=" * 80)

def check_dependencies():
    """Check and log important dependencies."""
    try:
        import pandas as pd
        import streamlit as st
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"Streamlit version: {st.__version__}")
    except ImportError as e:
        logger.error(f"Dependency error: {str(e)}")
        raise

def main_wrapper():
    """Wrapper function to handle main execution with error handling."""
    try:
        # Add the project root to the Python path
        project_root = str(Path(__file__).parent.resolve())
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        log_environment()
        check_dependencies()
        
        # Import the main function from advanced_main
        from advanced_main import main
        
        # Run the main application
        main()
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Fatal error in application: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main_wrapper()
