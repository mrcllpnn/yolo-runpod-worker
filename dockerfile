FROM ultralytics/ultralytics:latest

WORKDIR /app

# Install serverless runtime + huggingface downloader
RUN pip install --no-cache-dir runpod huggingface-hub

# Copy handler
COPY handler.py .

# Run the serverless worker
CMD ["python", "-u", "handler.py"]
