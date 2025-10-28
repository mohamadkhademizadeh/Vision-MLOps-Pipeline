import argparse, os, shutil, glob
from ultralytics import YOLO
ap = argparse.ArgumentParser()
ap.add_argument('--weights', default='models/best.pt')
ap.add_argument('--onnx', default='models/yolov8.onnx')
ap.add_argument('--imgsz', type=int, default=640)
args = ap.parse_args()
model = YOLO(args.weights)
model.export(format='onnx', imgsz=args.imgsz, opset=12, dynamic=False, optimize=True)
files = sorted(glob.glob('runs/onnx/*/*.onnx'), key=os.path.getmtime)
if not files: raise SystemExit('Export failed')
os.makedirs(os.path.dirname(args.onnx), exist_ok=True)
shutil.copy2(files[-1], args.onnx)
print('Saved ONNX ->', args.onnx)
