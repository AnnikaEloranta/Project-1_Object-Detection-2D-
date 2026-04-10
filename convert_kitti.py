import os, shutil, random
from pathlib import Path

# =============================================================================
# KITTI to YOLO Format Converter
# =============================================================================
# INPUT:
#   - data/raw/training/image_2/   → original KITTI images (7481 PNG files)
#   - data/raw/training/label_2/   → original KITTI labels (7481 TXT files)
#
# OUTPUT:
#   - data/images/train/           → 80% of images for training (~5984 images)
#   - data/images/val/             → 20% of images for validation (~1497 images)
#   - data/labels/train/           → converted YOLO labels for training
#   - data/labels/val/             → converted YOLO labels for validation
#   - val_ids.txt                  → list of validation image IDs (needed for KITTI evaluator)



# --- File paths ---
# Source: original KITTI dataset
KITTI_IMAGES = "data/raw/training/image_2"   # original KITTI images
KITTI_LABELS = "data/raw/training/label_2"   # original KITTI label files

# Destination: converted data in YOLO format
OUT_IMG_TRAIN = "data/images/train"   # training images
OUT_IMG_VAL   = "data/images/val"     # validation images
OUT_LBL_TRAIN = "data/labels/train"   # training labels in YOLO format
OUT_LBL_VAL   = "data/labels/val"     # validation labels in YOLO format

# --- Class mapping ---
# These 3 classes are used for 2D object detection
CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# These classes are ignored
SKIP = {"DontCare", "Misc", "Tram", "Truck", "Van", "Person_sitting"}

# --- Image dimensions ---
# KITTI images are size: 1242 x 375 pixels
# Used to normalize bounding box coordinates to the 0-1 range
IMG_W, IMG_H = 1242, 375

# --- Create output folders if they don't exist ---
for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
    os.makedirs(d, exist_ok=True)

# --- Train/validation split ---
all_ids = sorted([f.stem for f in Path(KITTI_LABELS).glob("*.txt")])

# Shuffle randomly
random.seed(42)
random.shuffle(all_ids)

split = int(0.8 * len(all_ids))
train_ids, val_ids = all_ids[:split], all_ids[split:]

# --- Conversion function ---
def convert(ids, img_out, lbl_out):
    """
    For each image ID:
    1. Copies the image to the output folder
    2. Reads the KITTI label file
    3. Converts each bounding box from KITTI format to YOLO format
    4. Saves the converted label file

    KITTI format:  class_name x1 y1 x2 y2  (absolute pixel coordinates, top-left & bottom-right)
    YOLO format:   class_id cx cy w h       (normalized center coordinates, values between 0-1)
    """
    for fid in ids:
        src_img = f"{KITTI_IMAGES}/{fid}.png"
        src_lbl = f"{KITTI_LABELS}/{fid}.txt"

        # Copy image as-is — no modifications needed
        shutil.copy(src_img, f"{img_out}/{fid}.png")

        lines = []
        for row in open(src_lbl).read().strip().split("\n"):
            parts = row.split()
            cls = parts[0]  # class name is always the first column

            # Skip irrelevant or unsupported classes
            if cls in SKIP or cls not in CLASS_MAP:
                continue

            # Extract 2D bounding box coordinates (columns 4-7 in KITTI format)
            x1, y1, x2, y2 = map(float, parts[4:8])

            # Convert to YOLO format:
            # cx, cy = center of the box, normalized by image width/height
            # w, h   = width and height of the box, normalized by image width/height
            cx = ((x1 + x2) / 2) / IMG_W
            cy = ((y1 + y2) / 2) / IMG_H
            w  = (x2 - x1) / IMG_W
            h  = (y2 - y1) / IMG_H

            # Write one line per object: class_id cx cy width height
            lines.append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save converted label file (same filename as image, but .txt extension)
        with open(f"{lbl_out}/{fid}.txt", "w") as f:
            f.write("\n".join(lines))

print("Starting label converting")

# --- Run conversion for train and val splits ---
convert(train_ids, OUT_IMG_TRAIN, OUT_LBL_TRAIN)
convert(val_ids,   OUT_IMG_VAL,   OUT_LBL_VAL)

# --- Save validation image IDs ---
# val_ids.txt is required later by the KITTI evaluation tool
# It tells the evaluator which images belong to the validation set
val_ids_sorted = sorted(val_ids)
with open("val_ids.txt", "w") as f:
    f.write("\n".join(val_ids_sorted))

print(f"Done. Train: {len(train_ids)}, Val: {len(val_ids)}")
print(f"val_ids.txt saved with {len(val_ids_sorted)} entries")
