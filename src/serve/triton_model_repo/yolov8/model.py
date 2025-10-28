# Triton Python backend: package helper + runtime stub
import argparse, os, shutil, json

def pack(onnx_path, repo):
    dst_dir = os.path.join(repo, '1')
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(onnx_path, os.path.join(dst_dir, 'model.onnx'))
    cfg = os.path.join(repo, 'config.pbtxt')
    if not os.path.exists(cfg):
        open(cfg,'w').write('name: "yolov8"\nbackend: "onnxruntime"\n')
    print('Packed Triton repo at', repo)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', action='store_true')
    ap.add_argument('--onnx', default='models/yolov8.onnx')
    ap.add_argument('--repo', default='src/serve/triton_model_repo/yolov8')
    args = ap.parse_args()
    if args.pack:
        pack(args.onnx, args.repo)
