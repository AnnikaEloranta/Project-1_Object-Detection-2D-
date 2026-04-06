import os
from pathlib import Path
from PIL import Image

OUT_IMG_TRAIN = Path("data/images/train")
OUT_IMG_VAL   = Path("data/images/val")
OUT_LBL_TRAIN = Path("data/labels/train")
OUT_LBL_VAL   = Path("data/labels/val")

MOSAIC_IMG_TRAIN = Path("mosaics/images/training")
MOSAIC_IMG_VAL   = Path("mosaics/images/val")
MOSAIC_LBL_TRAIN = Path("mosaics/labels/training")
MOSAIC_LBL_VAL   = Path("mosaics/labels/val")

for folder in [MOSAIC_IMG_TRAIN, MOSAIC_IMG_VAL, MOSAIC_LBL_TRAIN, MOSAIC_LBL_VAL]:
    folder.mkdir(parents=True, exist_ok=True)

# Horizontal overlap (as images are vertically <400 pixels)
OVERLAP = 70

def create_mosaics(img_folder: Path, lbl_folder: Path, out_img_folder: Path, out_lbl_folder: Path):
    img_files = sorted([f for f in img_folder.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]) # Only png should be enough but for safety
    
    for img_file in img_files:
        fid = img_file.stem
        img_path = img_file
        lbl_path = lbl_folder / f"{fid}.txt"
        
        # ------------------------ READ THE IMAGE AND LABELS -----------------------------------
        try:  # Image reading errored on windows before so just for safety if any image is corrupted
            with Image.open(img_path) as im:
                im.load()
                width, height = im.size
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue
        
        labels = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = (cx - w/2) * width
                    x2 = (cx + w/2) * width
                    y1 = (cy - h/2) * height
                    y2 = (cy + h/2) * height
                    labels.append((cls, x1, y1, x2, y2))
        

        # ------------------------ BUILD THE MOSAICED IMAGES AND LABELS -----------------------------------
        tile_w = (width + 2*OVERLAP) // 3
        x_starts = [0, tile_w - OVERLAP, 2*tile_w - 2*OVERLAP]
        
        for i, x0 in enumerate(x_starts):
            x0 = int(x0)
            x1 = min(x0 + tile_w, width)
            tile = im.crop((x0, 0, x1, height))
            
            # Build the mosaiced labels based on our image-mosaic boundaries
            tile_labels = []
            for cls, bx1, by1, bx2, by2 in labels:
                if bx2 < x0 or bx1 > x1:
                    continue
                new_x1 = max(bx1, x0) - x0
                new_x2 = min(bx2, x1) - x0
                new_y1 = by1
                new_y2 = by2
                if new_x2 - new_x1 <= 0 or new_y2 - new_y1 <= 0:
                    continue
                nw = new_x2 - new_x1
                nh = new_y2 - new_y1
                ncx = (new_x1 + new_x2)/2 / (x1 - x0)
                ncy = (new_y1 + new_y2)/2 / height
                tile_labels.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw/(x1-x0):.6f} {nh/height:.6f}")
            
            out_img_name = out_img_folder / f"{fid}_m{i+1}.png"
            tile.save(out_img_name)
            
            out_lbl_name = out_lbl_folder / f"{fid}_m{i+1}.txt"
            with open(out_lbl_name, "w") as f:
                f.write("\n".join(tile_labels))

print("Starting mosaicing.")

create_mosaics(OUT_IMG_TRAIN, OUT_LBL_TRAIN, MOSAIC_IMG_TRAIN, MOSAIC_LBL_TRAIN)
create_mosaics(OUT_IMG_VAL,   OUT_LBL_VAL,   MOSAIC_IMG_VAL,   MOSAIC_LBL_VAL)

print("Mosaic generation completed!")