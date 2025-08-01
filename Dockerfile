FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY all-MiniLM-L6-v2-finetuned/ ./all-MiniLM-L6-v2-finetuned/
COPY Challenge_1b/ ./Challenge_1b/
COPY pdf_analyzer.py ./

# Default command (can override with --collection argument)
ENTRYPOINT ["python", "pdf_analyzer.py"] 