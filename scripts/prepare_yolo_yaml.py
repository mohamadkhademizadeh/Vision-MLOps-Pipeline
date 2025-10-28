import argparse, yaml, os, json
parser = argparse.ArgumentParser()
parser.add_argument('--params', default='params.yaml')
parser.add_argument('--out', default='scripts/yolo_data.yaml')
args = parser.parse_args()
P = yaml.safe_load(open(args.params))
names = P['data']['classes']
data_yaml = {
  'path': '.',
  'train': P['data']['train_dir'],
  'val': P['data']['val_dir'],
  'names': {int(k): v for k,v in names.items()}
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w') as f:
    yaml.safe_dump(data_yaml, f)
print('Wrote', args.out)
