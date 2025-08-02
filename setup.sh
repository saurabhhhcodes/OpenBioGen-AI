#!/bin/bash
set -e  # Exit on error
set -x  # Print commands as they are executed

echo "=== Starting OpenBioGen-AI Setup ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1 || echo 'Python not found')"

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Install Python dependencies with retries
log "Installing Python dependencies..."
for i in {1..3}; do
    log "Attempt $i/3: Installing requirements..."
    if pip install --no-cache-dir -r requirements.txt; then
        log "Successfully installed requirements"
        break
    else
        if [ $i -eq 3 ]; then
            log "ERROR: Failed to install requirements after 3 attempts"
            log "Trying with --no-deps flag..."
            pip install --no-deps -r requirements.txt || {
                log "ERROR: Failed to install requirements with --no-deps flag"
                exit 1
            }
        fi
        log "Retrying pip install in 5 seconds..."
        sleep 5
    fi
done

# Verify critical packages
log "Verifying critical packages..."
python -c "import pandas, streamlit, numpy; print(f'Pandas: {pandas.__version__}, Streamlit: {streamlit.__version__}, NumPy: {numpy.__version__}')" || {
    log "ERROR: Failed to verify critical packages"
    exit 1
}

# Create necessary directories
log "Creating necessary directories..."
mkdir -p logs data/cache

# Set permissions
log "Setting permissions..."
chmod -R 777 logs/ data/

log "=== Setup completed successfully! ==="
