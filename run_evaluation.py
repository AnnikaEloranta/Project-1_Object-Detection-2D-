import os
import sys
import glob
import matplotlib.pyplot as plt

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

try:
    from kitti_object_eval_python import kitti_common as kitti
    from kitti_object_eval_python.eval import get_official_eval_result
except ModuleNotFoundError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- PATHS ---
GROUND_TRUTH_DIR = "data/raw/training/label_2"
PREDICTIONS_DIR = "output/run_1/labels_kitti_format" 

def main():
    print("Loading Ground Truth and Predictions...")
    
    #val_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*.txt"))
    #val_image_ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in val_files]

    with open("val_ids.txt", "r") as f:
        val_image_ids = [int(line.strip()) for line in f if line.strip()]
    
    #val_image_ids = val_image_ids[:100] 
    print(f"TEST RUN: Evaluating only {len(val_image_ids)} images...")
    
    if len(val_image_ids) == 0:
        print(f"Error: No prediction files found in {PREDICTIONS_DIR}")
        return

    gt_annos = kitti.get_label_annos(GROUND_TRUTH_DIR, val_image_ids)
    dt_annos = kitti.get_label_annos(PREDICTIONS_DIR, val_image_ids)

    print(f"Evaluating {len(val_image_ids)} images...")
    
    # We now catch BOTH the text result and our newly liberated metrics dictionary!
    result = get_official_eval_result(gt_annos, dt_annos, [0, 1, 2])
    
    if isinstance(result, tuple):
        eval_results = result[0]
        pr_data_dict = result[1] 
    else:
        print("Error: eval.py hack failed. Did you change 'return result' to 'return result, metrics'?")
        return
    
    print("\n================ KITTI EVALUATION RESULTS ================\n")
    print(eval_results)

    # --- PLOTTING SECTION:  ---
    try:
        classes = ["Car", "Pedestrian", "Cyclist"]
        difficulty_names = ["Easy", "Moderate", "Hard"]
        
        # Create the 41 points for the X-axis (0.0 to 1.0)
        recall_points = [j / 40.0 for j in range(41)]

        for d_idx, d_name in enumerate(difficulty_names):
            plt.figure(figsize=(10, 6))
            plotted_in_this_diff = False

            for i, cls_name in enumerate(classes):
                try:
                    # i = class, d_idx = difficulty (0,1,2), 0 = bbox overlap type
                    precision_data = pr_data_dict["bbox"]["precision"][i, d_idx, 0, :]
                    
                    # Only plot if there is actual data (not all zeros or NaNs)
                    if not all(v == 0 for v in precision_data):
                        plt.plot(recall_points, precision_data, label=f'{cls_name}')
                        plotted_in_this_diff = True
                except Exception as e:
                    print(f"Could not plot {cls_name} for {d_name}: {e}")
                    continue

            if plotted_in_this_diff:
                plt.title(f'KITTI 2D Object Detection - PR Curves ({d_name})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.legend(loc='lower left')
                plt.grid(True)
                
                filename = f'pr_curves_{d_name.lower()}.png'
                plt.savefig(filename)
                print(f"SUCCESS: {d_name} curve saved as {filename}")
            
            plt.close()

    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
