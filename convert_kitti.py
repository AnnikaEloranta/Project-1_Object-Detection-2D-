import os, shutil, random
from pathlib import Path

KITTI_IMAGES = "data/raw/training/image_2"
KITTI_LABELS = "data/raw/training/label_2"
OUT_IMG_TRAIN = "data/images/train"
OUT_IMG_VAL   = "data/images/val"
OUT_LBL_TRAIN = "data/labels/train"
OUT_LBL_VAL   = "data/labels/val"

CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
SKIP = {"DontCare", "Misc", "Tram", "Truck", "Van", "Person_sitting"}
IMG_W, IMG_H = 1242, 375

for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
    os.makedirs(d, exist_ok=True)

all_ids = sorted([f.stem for f in Path(KITTI_LABELS).glob("*.txt")])
random.seed(42)
random.shuffle(all_ids)
split = int(0.8 * len(all_ids))
train_ids, val_ids = all_ids[:split], all_ids[split:]

def convert(ids, img_out, lbl_out):
    for fid in ids:
        src_img = f"{KITTI_IMAGES}/{fid}.png"
        src_lbl = f"{KITTI_LABELS}/{fid}.txt"
        shutil.copy(src_img, f"{img_out}/{fid}.png")
        lines = []
        for row in open(src_lbl).read().strip().split("\n"):
            parts = row.split()
            cls = parts[0]
            if cls in SKIP or cls not in CLASS_MAP:
                continue
            x1,y1,x2,y2 = map(float, parts[4:8])
            cx = ((x1+x2)/2) / IMG_W
            cy = ((y1+y2)/2) / IMG_H
            w  = (x2-x1) / IMG_W
            h  = (y2-y1) / IMG_H
            lines.append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(f"{lbl_out}/{fid}.txt", "w") as f:
            f.write("\n".join(lines))

convert(train_ids, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
convert(val_ids,   OUT_IMG_VAL,   OUT_LBL_VAL)

val_ids_sorted = sorted(val_ids)
with open("val_ids.txt", "w") as f:
    f.write("\n".join(val_ids_sorted))

print(f"Done. Train: {len(train_ids)}, Val: {len(val_ids)}")
print(f"val_ids.txt saved with {len(val_ids_sorted)} entries")
