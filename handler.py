import os
import base64
import cv2
import numpy as np
import runpod
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# 1. Force YOLO to use a writable directory for internal settings
os.environ['ULTRALYTICS_CONFIG_DIR'] = '/tmp/ultralytics'

# 2. PROPER HUGGING FACE DOWNLOAD
# This downloads the actual file and returns the local path string
print("Downloading model from Hugging Face...")
try:
    model_path = hf_hub_download(
        repo_id="macpaw-research/yolov11l-ui-elements-detection", 
        filename="model.pt"
    )
    # 3. Load the model from the local path
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

def handler(job):
    try:
        job_input = job.get("input", {})
        base64_image = job_input.get("image")
        
        if not base64_image:
            return {"error": "No image provided"}

        # Decode Image
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}

        # Run Inference
        results = model.predict(img, conf=0.25)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy.tolist()[0]]
                })
        
        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
