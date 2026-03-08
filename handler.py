import os
# FORCE EVERYTHING TO /tmp (The only place guaranteed to be writable on RunPod)
os.environ['ULTRALYTICS_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'

import runpod
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image

model = None

def load_model():
    global model
    if model is None:
        print("Downloading macpaw-research/yolov11l-ui-elements-detection ...")
        model_path = hf_hub_download(
            repo_id="macpaw-research/yolov11l-ui-elements-detection",
            filename="ui-elements-detection.pt"
        )
        print(f"Model downloaded to: {model_path}")
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
    return model

def handler(job):
    try:
        # Load model (cached after first load)
        model = load_model()
        
        # Decode Image using PIL (simpler than cv2)
        image_data = base64.b64decode(job["input"]["image"])
        img = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Run Inference
        results = model.predict(img, conf=0.25)
        
        detections = []
        for r in results:
            for box in r.boxes:
                bbox = [round(float(x), 1) for x in box.xyxy]
                # Calculate center from bbox [x1, y1, x2, y2]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    center = [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)]
                else:
                    center = [0, 0]
                
                detections.append({
                    "label": model.names[int(box.cls)],  # VisionClient expects 'label' not 'class'
                    "confidence": float(box.conf),  # Keep as float for consistency
                    "bbox": bbox,
                    "center": center  # VisionClient expects 'center' field
                })
        return {"detections": detections}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
