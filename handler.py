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
        print("Loading model...")
        model_path = hf_hub_download(
            repo_id="macpaw-research/yolov11l-ui-elements-detection",
            filename="ui-elements-detection.pt"  # This is the only change needed
        )
        model = YOLO(model_path)
        print("Model ready.")
    return model

def handler(job):
    input_data = job['input']
    image_src = input_data.get('image')

    if not image_src:
        return {"error": "Missing 'image' in input (URL, base64, or path)"}

    # Load image
    if image_src.startswith(('http://', 'https://')):
        r = requests.get(image_src)
        r.raise_for_status()
        img_bytes = r.content
    elif image_src.startswith('data:'):
        _, b64 = image_src.split(',', 1)
        img_bytes = base64.b64decode(b64)
    else:
        with open(image_src, 'rb') as f:
            img_bytes = f.read()

    img = Image.open(BytesIO(img_bytes))

    model = load_model()
    results = model.predict(img, conf=0.25, iou=0.45, imgsz=640)

    detections = []
    for res in results:
        for box in res.boxes:
            detections.append({
                "class": res.names[int(box.cls)],
                "conf": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]]
            })

    return {"detections": detections, "count": len(detections)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
