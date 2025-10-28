import argparse, json, os, glob, shutil
from ultralytics.utils.plotting import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='runs/detect')
ap.add_argument('--out', default='reports')
args = ap.parse_args()

exps = sorted(glob.glob(os.path.join(args.runs, 'exp*')), key=os.path.getmtime)
if not exps: raise SystemExit('No runs found')
last = exps[-1]
metrics_file = os.path.join(last, 'results.csv')
os.makedirs(args.out, exist_ok=True)

# Try to gather simple metrics
metrics = {}
try:
    import pandas as pd
    df = pd.read_csv(metrics_file)
    # take last row
    row = df.iloc[-1].to_dict()
    metrics = {k: float(row[k]) for k in row if isinstance(row[k], (int,float))}
except Exception as e:
    metrics['note'] = f'could not parse metrics: {e}'

with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# Placeholder confusion plot (random if not available)
cm_path = os.path.join(args.out, 'confusion.png')
try:
    # Ultralytics often stores confusion in runs; we stub a simple plot
    fig = plt.figure()
    plt.imshow(np.array([[0.8,0.2],[0.3,0.7]]))
    plt.title('Confusion (placeholder)')
    plt.colorbar()
    fig.savefig(cm_path); plt.close(fig)
except Exception:
    pass

print('Wrote reports to', args.out)
