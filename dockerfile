FROM runpod/pytorch:2.3.0-py3.10-cuda12.1.1-devel-ubuntu22.04  # or similar RunPod base

WORKDIR /workspace

RUN pip install --no-cache-dir runpod ultralytics huggingface_hub torch torchvision

# Copy your code
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
