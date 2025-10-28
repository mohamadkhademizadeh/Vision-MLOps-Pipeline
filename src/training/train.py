import argparse, yaml, os
from ultralytics import YOLO

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='scripts/yolo_data.yaml')
    ap.add_argument('--params', default='params.yaml')
    ap.add_argument('--out_dir', default='models')
    args = ap.parse_args()

    P = yaml.safe_load(open(args.params))
    E = P['train']
    os.makedirs(args.out_dir, exist_ok=True)

    model = YOLO(E['weights'])
    model.train(data=args.data, epochs=E['epochs'], imgsz=E['imgsz'], batch=E['batch'],
                patience=E['patience'], lr0=E['lr0'], lrf=E['lrf'], project='runs/detect', name='exp', exist_ok=True)
    # copy best.pt into models/
    import glob, shutil
    bests = sorted(glob.glob('runs/detect/exp*/weights/best.pt'))
    if not bests: raise SystemExit('No best.pt found after training')
    shutil.copy2(bests[-1], os.path.join(args.out_dir, 'best.pt'))
    print('Saved best weights to', os.path.join(args.out_dir, 'best.pt'))
