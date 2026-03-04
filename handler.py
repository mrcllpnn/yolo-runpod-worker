import os
import requests
import base64
import cv2
import numpy as np
import runpod
from ultralytics import YOLO

# Force YOLO to use a writable directory to avoid "Read-only" crashes
os.environ['ULTRALYTICS_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'

# Direct download for the MacPaw model
MODEL_URL = "https://huggingface.co"
MODEL_PATH = "/tmp/model.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading MacPaw YOLO11l...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load the model
model = YOLO(MODEL_PATH)

def handler(job):
    try:
        # Decode Base64 Image
        image_data = base64.b64decode(job["input"]["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Inference (0.1s speed)
        results = model.predict(img, conf=0.25)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2),
                    "bbox": [round(float(x), 1) for x in box.xyxy]
                })
        return {"detections": detections}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
