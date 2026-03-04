# Recommended base: Latest RunPod PyTorch with high versions (PyTorch 2.9.1 + CUDA 12.9)
FROM runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204

WORKDIR /workspace

# Install required Python packages (runpod serverless + ultralytics + huggingface_hub)
# Torch is already in the base image, but ensure compatibility
RUN pip install --no-cache-dir \
    runpod \
    ultralytics \
    huggingface_hub

# Copy your handler code (make sure handler.py is in the same directory as this Dockerfile)
COPY handler.py /workspace/handler.py

# Optional: Pre-download the model during build for faster cold starts
# (increases image size by ~ a few hundred MB, but worth it for serverless)
# RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='macpaw-research/yolov11l-ui-elements-detection', filename='ui-elements-detection.pt')"

# Run the RunPod serverless handler
CMD ["python", "-u", "/workspace/handler.py"]
