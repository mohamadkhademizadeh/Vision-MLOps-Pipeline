# Optional: tiny toy dataset generator (YOLO format) for smoke tests
import os, cv2, numpy as np, random, argparse, shutil

def make_dirs(root):
    for p in ['data/images/train','data/images/val','data/labels/train','data/labels/val']:
        os.makedirs(p, exist_ok=True)

def gen(split, n=50, w=640, h=480):
    img_dir=f'data/images/{split}'; lab_dir=f'data/labels/{split}'
    for i in range(n):
        img=np.zeros((h,w,3), dtype=np.uint8)
        # draw random rectangle (class 0) or circle (class 1)
        cls = random.randint(0,1)
        if cls==0:
            x1=random.randint(50, w-150); y1=random.randint(50, h-150)
            x2=x1+random.randint(40,120); y2=y1+random.randint(40,120)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),-1)
            cx=(x1+x2)/2/w; cy=(y1+y2)/2/h; bw=(x2-x1)/w; bh=(y2-y1)/h
        else:
            cx=random.uniform(0.2,0.8); cy=random.uniform(0.2,0.8)
            r=random.uniform(0.05,0.15); bw=bh=2*r
            cv2.circle(img,(int(cx*w),int(cy*h)), int(r*min(w,h)), (255,255,255), -1)
        stem=f'{split}_{i:04d}'
        cv2.imwrite(f'{img_dir}/{stem}.jpg', img)
        with open(f'{lab_dir}/{stem}.txt','w') as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=60)
    parser.add_argument('--val', type=int, default=20)
    args=parser.parse_args()
    make_dirs('data')
    gen('train', args.train); gen('val', args.val)
    print('Synthetic dataset created under data/images and data/labels')
