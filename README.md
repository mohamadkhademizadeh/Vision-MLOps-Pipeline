# Vision MLOps Pipeline (DVC + YOLOv8 + Triton + CI)

A compact, portfolio-ready **end-to-end MLOps** template for computer vision:
- **Dataset versioning** with **DVC**
- **Training** (Ultralytics YOLOv8) with reproducible configs
- **Evaluation** + metrics export (mAP, precision/recall, confusion)
- **Model export** to **ONNX** and packaging as a **Triton** model repository (Python backend wrapper)
- **Serving** via Triton *or* a lightweight **FastAPI** wrapper
- **CI** workflow checks style, runs a tiny smoke test, and validates the Triton repo layout

> This is a realistic scaffold that you can extend with your data and infra; it runs locally without cloud creds.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Initialize DVC (local remote)
dvc init
dvc remote add -d localstore ./dvcstore

# 2) Put images and labels (YOLO format)
# data/raw/*.jpg, data/labels/*.txt
# Optionally: `python scripts/synthetic_data.py` to generate toy data

# 3) Reproduce pipeline (train -> evaluate -> export -> package Triton repo)
dvc repro
```

Artifacts:
- Trained weights under `models/`
- Metrics `reports/metrics.json` + `reports/confusion.png`
- ONNX model in `models/yolov8.onnx`
- Triton model repo under `src/serve/triton_model_repo/yolov8/`

---

## Directory layout
```
Vision-MLOps-Pipeline/
├── data/                     # versioned with DVC
├── src/
│   ├── training/train.py     # YOLOv8 training
│   ├── eval/evaluate.py      # metrics & plots
│   └── serve/
│       ├── triton_model_repo/yolov8/{1,config.pbtxt,model.py}
│       └── api/server.py     # optional FastAPI service
├── scripts/
│   ├── prepare_yolo_yaml.py  # generates Ultralytics data yaml
│   ├── export_onnx.py        # weights -> ONNX
│   └── synthetic_data.py     # tiny toy dataset generator
├── dvc.yaml                  # pipeline: prepare -> train -> eval -> export -> package
├── params.yaml               # hyperparams for training
├── requirements.txt
└── .github/workflows/ci.yml
```
