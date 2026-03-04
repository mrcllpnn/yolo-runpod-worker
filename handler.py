import os
import base64
import cv2
import numpy as np
import runpod
from ultralytics import YOLO

# 1. Force YOLO to use a writable directory for its internal settings
os.environ['ULTRALYTICS_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'

# 2. Load the model. 
# Ultralytics will automatically download the .pt file from Hugging Face 
# if you provide the repo string. This is much safer than manual requests.
print("Initializing MacPaw YOLO11l...")
try:
    model = YOLO("macpaw-research/yolov11l-ui-elements-detection")
except Exception as e:
    # Fallback: Direct download link if the repo string fails
    print(f"Primary load failed, trying direct URL... Error: {e}")
    model = YOLO("https://huggingface.co")

def handler(job):
    try:
        # Get input image
        job_input = job.get("input", {})
        base64_image = job_input.get("image")
        
        if not base64_image:
            return {"error": "No image provided in input"}

        # Decode Base64 Image
        image_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Failed to decode image. Ensure it is a valid Base64 string."}

        # Run Inference
        results = model.predict(img, conf=0.25)
        
        # Format Results
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()]
                })
        
        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

# Start the RunPod Serverless worker
runpod.serverless.start({"handler": handler})
