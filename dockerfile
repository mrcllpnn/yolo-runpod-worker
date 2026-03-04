# Use a valid recent RunPod PyTorch base image
# Recommended: one of the existing tags like 2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (or check Docker Hub for latest)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Install additional packages needed for your YOLO setup
RUN pip install --no-cache-dir \
    runpod \
    ultralytics \
    huggingface_hub \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # match your CUDA if needed

# Copy your handler code
COPY handler.py /workspace/handler.py

# Optional: if you want to pre-download the model during build (faster cold starts, but increases image size)
# RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='macpaw-research/yolov11l-ui-elements-detection', filename='ui-elements-detection.pt')"

CMD ["python", "-u", "/workspace/handler.py"]
