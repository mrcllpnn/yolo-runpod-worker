FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /workspace

RUN pip install --no-cache-dir runpod ultralytics huggingface-hub pillow requests

COPY handler.py .

CMD ["python", "-u", "handler.py"]
