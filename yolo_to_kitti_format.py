import os

# ==============================================================================
# --- PATHS ---
# ==============================================================================
# This path points to the inference predictions floder in YOLO-format (6 columns: cls cx cy w h conf)
YOLO_LABELS_DIR = "output/run_1/labels"  

# Path to corresponding KITTI labels
KITTI_OUTPUT_DIR = "output/run_1/labels_kitti_format"

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# Must match the class indices used in yolo.yaml
INV_CLASS_MAP = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

IMG_W = 1240 
IMG_H = 375 

def convert_yolo_to_kitti():
    """
    Converts YOLO format labels (normalized) into the 16-column KITTI 
    format (absolute pixels) required by the run_evaluation.py script.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(KITTI_OUTPUT_DIR):
        os.makedirs(KITTI_OUTPUT_DIR)
        print(f"Created directory: {KITTI_OUTPUT_DIR}")

    yolo_files = [f for f in os.listdir(YOLO_LABELS_DIR) if f.endswith(".txt")]
    
    if len(yolo_files) == 0:
        print(f"Error: No files found in {YOLO_LABELS_DIR}. Check your inference output.")
        return

    print(f"Converting {len(yolo_files)} files to KITTI format...")
    
    count = 0
    for filename in yolo_files:
        yolo_path = os.path.join(YOLO_LABELS_DIR, filename)
        
        with open(yolo_path, "r") as f:
            lines = f.readlines()
        
        kitti_lines = []
        for line in lines:
            parts = line.strip().split()
            
            # Check for at least 6 columns: cls, cx, cy, w, h, score
            if len(parts) < 6:
                # If inference didn't save scores yet, this will skip the line
                continue 
            
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            # Extract the actual confidence score from the 6th column
            score = float(parts[5]) 
            
            # Convert normalized YOLO (0-1) to absolute KITTI pixels
            # x1, y1 is top-left | x2, y2 is bottom-right
            x1 = (cx - w / 2) * IMG_W
            y1 = (cy - h / 2) * IMG_H
            x2 = (cx + w / 2) * IMG_W
            y2 = (cy + h / 2) * IMG_H
            
            # Get class name (defaults to DontCare if class ID is unknown)
            cls_name = INV_CLASS_MAP.get(cls_id, "DontCare")
            
            # KITTI Format Columns (16 total):
            # 1: Type (string)
            # 2: Truncated (0.00)
            # 3: Occluded (0)
            # 4: Alpha (0.00)
            # 5-8: BBox x1, y1, x2, y2 (float)
            # 9-11: Dimensions h, w, l (0.00 - dummy for 2D)
            # 12-14: Location x, y, z (0.00 - dummy for 2D)
            # 15: Rotation_y (0.00)
            # 16: Confidence Score (float)
            
            kitti_row = [
                cls_name,
                "0.00", "0", "0.00",
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                "0.00", "0.00", "0.00",
                "0.00", "0.00", "0.00",
                "0.00",
                f"{score:.4f}"
            ]
            kitti_lines.append(" ".join(kitti_row))
            
        # Write the new KITTI-formatted file
        with open(os.path.join(KITTI_OUTPUT_DIR, filename), "w") as f:
            f.write("\n".join(kitti_lines))
        count += 1
            
    print(f"Success! {count} files converted and saved to: {KITTI_OUTPUT_DIR}")

if __name__ == "__main__":
    convert_yolo_to_kitti()