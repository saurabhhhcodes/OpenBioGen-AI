FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-deploy.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p cache memory_storage logs security_logs

# Expose port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "advanced_main.py", "--server.port=8501", "--server.address=0.0.0.0"] 