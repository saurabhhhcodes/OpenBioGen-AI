"""
Streamlit App Entry Point for OpenBioGen AI
This file serves as the main entry point for Streamlit Cloud deployment.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    # Add the project root to the Python path
    project_root = str(Path(__file__).parent.resolve())
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    logger.info("Starting OpenBioGen AI application...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Import the main function from advanced_main
    from advanced_main import main
    
    # Run the main application
    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            logger.error(f"Error in main application: {str(e)}", exc_info=True)
            raise
            
except Exception as e:
    logger.critical(f"Failed to start application: {str(e)}", exc_info=True)
    raise
