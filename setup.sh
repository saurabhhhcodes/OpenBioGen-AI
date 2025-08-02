#!/bin/bash
set -e  # Exit on error

# Install Python dependencies with retries
for i in {1..3}; do
    pip install --no-cache-dir -r requirements.txt && break || {
        if [ $i -eq 3 ]; then
            echo "Failed to install requirements after 3 attempts"
            exit 1
        }
        echo "Retrying pip install..."
        sleep 5
    }
done

# Install additional system dependencies if needed
# apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

echo "Setup completed successfully!"
