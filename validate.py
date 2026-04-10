import os
import cv2
import glob
import torch
import numpy as np
from ultralytics import YOLO
import torchvision

# =============================================================================
# Regular (non KITTI) validation and bounding box visualizations script.
# Also NMS and building whole image labels for further evaluations
# =============================================================================
# INPUT:
#   - final_80_20_run/weights/best.pt  → Trained YOLO model weights
#   - mosaics/images/val/              → MAX 640x640 validation image tiles
#
# OUTPUT:
#   - output/run_{id}/labels/             → Reconstructed whole-image YOLO labels
#   - output/run_{id}/images/             → Stitched visualization images with predicted bboxes
#
# LOGIC:
#   - Runs the trained YOLO model on the given validation set and builds the original image back from the tiles
#   - NMS per class applied to remove overlapping bounding boxes (yolo12n is very prone to these).
#   - Ensures that output label format is YOLO format.
# =============================================================================

MODEL_PATH = "final_80_20_run/weights/best.pt"
IMG_FOLDER = "mosaics/images/val"

OUTPUT_DIR = "output/run_2"
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
IMG_OUT_DIR = os.path.join(OUTPUT_DIR, "images")

OVERLAP = 70
TILES_PER_IMAGE = 3
VIS_COUNT = 30 # How many images do we draw the bbox for. 0 for no visualizations - max is image count
IOU_THRESHOLD = 0.3 # For NMS in the overlap region

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)


def get_image_groups(folder, tiles_per_image):
    """
    Group images according to the 3-split logic
    """
    all_images = sorted(glob.glob(os.path.join(folder, "*.*")))
    groups = {}
    
    for img_path in all_images:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        base_name = "_".join(name.split("_")[:-1]) 
        
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(img_path)
        
    # Filter out incomplete groups for safety
    valid_groups = {k: sorted(v) for k, v in groups.items() if len(v) == tiles_per_image}
    return valid_groups

def xyxy_to_yolo(boxes, img_w, img_h):
    """
    Label-format conversion
    """
    x_center = ((boxes[:, 0] + boxes[:, 2]) / 2) / img_w
    y_center = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_h
    w = (boxes[:, 2] - boxes[:, 0]) / img_w
    h = (boxes[:, 3] - boxes[:, 1]) / img_h
    return torch.stack((x_center, y_center, w, h), dim=1)



def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    groups = get_image_groups(IMG_FOLDER, TILES_PER_IMAGE)
    print(f"Found {len(groups)} valid image sets (each with {TILES_PER_IMAGE} tiles).")
    
    vis_counter = 0

    # Process each group (original image)
    for base_name, tile_paths in groups.items():
        all_boxes = []
        all_scores = []
        all_classes = []
        
        stitched_img = None
        tile_w = 0
        tile_h = 0
        
        tiles = [cv2.imread(p) for p in tile_paths]
        if any(t is None for t in tiles):
            print(f"Error reading tiles for {base_name}. Skipping.")
            continue
            
        tile_h, tile_w = tiles[0].shape[:2]
        
        # Calculate full stitched image dimensions
        full_w = (tile_w * TILES_PER_IMAGE) - (OVERLAP * (TILES_PER_IMAGE - 1))
        full_h = tile_h
        
        # Empty base for the whole image
        stitched_img = np.zeros((full_h, full_w, 3), dtype=np.uint8)

        # Predict and build the whole image
        for i, (tile_path, tile_img) in enumerate(zip(tile_paths, tiles)):
            # Predict on the individual tile
            results = model.predict(tile_img, verbose=False)[0]
            x_offset = i * (tile_w - OVERLAP)
            
            stitched_img[0:tile_h, x_offset : x_offset + tile_w] = tile_img
            
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu()
                scores = results.boxes.conf.cpu()
                classes = results.boxes.cls.cpu()
                
                boxes[:, 0] += x_offset
                boxes[:, 2] += x_offset
                
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)

        # -------------------- NMS ------------------------
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_classes = torch.cat(all_classes, dim=0)
            
            # Apply NMS to remove duplicate bounding boxes in the overlap regions
            # NMS is per class to avoid removing boxes where e.g. human on top of car
            keep_indices = torchvision.ops.batched_nms(all_boxes, all_scores, all_classes, IOU_THRESHOLD)
            
            final_boxes = all_boxes[keep_indices]
            final_scores = all_scores[keep_indices]
            final_classes = all_classes[keep_indices]
        else:
            final_boxes = torch.empty((0, 4))
            final_scores = torch.empty((0,))
            final_classes = torch.empty((0,))

        # ------------------------ Save labels ----------------------
        label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
        with open(label_path, "w") as f:
            if len(final_boxes) > 0:
                yolo_boxes = xyxy_to_yolo(final_boxes, full_w, full_h)
                for cls, box, score in zip(final_classes, yolo_boxes, final_scores):
                    f.write(f"{int(cls)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {float(score):.6f}\n")
        
        # --------------- Optional visualization based on our VIS_COUNT ---------------
        if vis_counter < VIS_COUNT:
            for box, score, cls in zip(final_boxes, final_scores, final_classes):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                conf = float(score)
                
                cv2.rectangle(stitched_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"cls{class_id} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(stitched_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(stitched_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            vis_path = os.path.join(IMG_OUT_DIR, f"{base_name}.jpg")
            cv2.imwrite(vis_path, stitched_img)
            vis_counter += 1

    print(f"Processing complete! Labels saved to '{LABEL_DIR}'.")
    print(f"Saved {min(vis_counter, VIS_COUNT)} visualizations to '{IMG_OUT_DIR}'.")

if __name__ == "__main__":
    main()
