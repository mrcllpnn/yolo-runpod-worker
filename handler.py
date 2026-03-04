import runpod
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import requests

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
        job_input = job["input"]
        image_input = job_input.get("image")

        if not image_input:
            return {"error": "Missing 'image' key in input. Provide a URL or base64 string."}

        # Load image
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input, timeout=15)
            response.raise_for_status()
            img_bytes = response.content
        elif image_input.startswith("data:image"):
            # base64 data URI
            header, encoded = image_input.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        else:
            return {"error": "Unsupported image format. Use URL or data:image/... base64."}

        img = Image.open(BytesIO(img_bytes))

        # Load model (only once)
        model_instance = load_model()

        # Run prediction
        results = model_instance.predict(
            img,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                detections.append({
                    "class": result.names[cls_id],
                    "confidence": round(float(box.conf), 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()]  # x1,y1,x2,y2
                })

        return {
            "detections": detections,
            "count": len(detections),
            "status": "ok"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
