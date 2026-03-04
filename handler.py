import runpod
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import torch

# Global model (loaded once at cold start for speed)
model = None

def load_model():
    global model
    if model is None:
        print("Downloading model from HF...")
        model_path = hf_hub_download(
            repo_id="macpaw-research/yolov11l-ui-elements-detection",
            filename="ui-elements-detection.pt",  # ← THIS is the fix!
            cache_dir="/workspace/cache"  # optional, helps with persistence if using network volume
        )
        model = YOLO(model_path)
        print("Model loaded successfully.")
    return model

def handler(job):
    job_input = job['input']
    
    # Expecting base64 image or URL in input — adapt as needed
    # Example: image_path = job_input.get('image')  # could be path, URL, or base64
    
    # For demo, assume you pass {"image": "https://example.com/screenshot.png"}
    source = job_input.get('image')
    
    if not source:
        return {"error": "No 'image' provided in input."}
    
    model = load_model()
    
    # Run inference
    results = model.predict(source, conf=0.25, iou=0.45, imgsz=640)  # tune thresholds as needed
    
    # Format output (adapt to what you need: boxes, classes, etc.)
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]  # [x1,y1,x2,y2]
            })
    
    return {
        "detections": detections,
        "num_detections": len(detections)
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
