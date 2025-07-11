FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with retry logic
RUN pip install --upgrade pip setuptools wheel

# Install large packages first with individual retry logic
RUN pip install --no-cache-dir --timeout=300 --retries=10 pandas numpy
RUN pip install --no-cache-dir --timeout=300 --retries=10 streamlit
RUN pip install --no-cache-dir --timeout=300 --retries=10 chromadb

# Install remaining requirements
RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy all project files
COPY . .

# Expose port 
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]