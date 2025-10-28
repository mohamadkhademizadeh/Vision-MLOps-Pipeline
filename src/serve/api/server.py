from fastapi import FastAPI, UploadFile, File
import numpy as np, cv2, io, os
from ultralytics import YOLO

app = FastAPI(title="Vision MLOps Serve")

model_path = os.environ.get('MODEL_PATH','models/best.pt')
model = None

def get_model():
    global model
    if model is None:
        model = YOLO(model_path)
    return model

@app.get('/health')
def health():
    return {'status': 'ok', 'model_path': model_path}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        return {'error': 'decode failed'}
    m = get_model()
    res = m.predict(img, verbose=False)[0]
    dets = []
    if hasattr(res, 'boxes') and res.boxes is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for b,s,c in zip(xyxy, conf, cls):
            dets.append({'xyxy': [float(x) for x in b], 'conf': float(s), 'cls': int(c)})
    return {'detections': dets}
