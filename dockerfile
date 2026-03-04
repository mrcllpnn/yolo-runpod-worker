# 1. Use the stable RunPod PyTorch base
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. Set the working directory
WORKDIR /

# 3. Copy your files from GitHub into the container
COPY handler.py /handler.py
COPY requirements.txt /requirements.txt

# 4. Install dependencies
# We upgrade pip first to avoid installation glitches
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

# 5. Create a writable directory for YOLO settings
# This prevents the "Read-only file system" error on RunPod
RUN mkdir -p /tmp/ultralytics && chmod -R 777 /tmp/ultralytics

# 6. Set the command to start your handler
# The -u flag ensures logs are sent to the RunPod dashboard in real-time
CMD [ "python", "-u", "/handler.py" ]
