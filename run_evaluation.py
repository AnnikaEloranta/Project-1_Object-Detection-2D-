import os
import sys
import glob
import matplotlib.pyplot as plt

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

# --- PATH FIX ---
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
GROUND_TRUTH_DIR = "data_object_label_2/training/label_2"
PREDICTIONS_DIR = "output/run_1/labels_kitti_format" 

def main():
    print("Loading Ground Truth and Predictions...")
    
    val_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*.txt"))
    val_image_ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in val_files]
    
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

    # --- PLOTTING PRECISION-RECALL CURVES ---
    try:
        classes = ["Car", "Pedestrian", "Cyclist"]
        plt.figure(figsize=(10, 6))
        plotted = False

        for i, cls_name in enumerate(classes):
            try:
                # We extract: [Class Index, Moderate Diff (1), Main Overlap (0), All 41 Points]
                precision_data = pr_data_dict["bbox"]["precision"][i, 1, 0, :]
                
                # Create the 41 points for the X-axis (0.0 to 1.0)
                recall_points = [j / 40.0 for j in range(41)]
                
                plt.plot(recall_points, precision_data, label=f'{cls_name} (Moderate)')
                plotted = True

            except Exception as e:
                print(f"Could not plot {cls_name}: {e}")
                continue

        if plotted:
            plt.title('KITTI 2D Object Detection - Precision-Recall Curves (Moderate)')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.savefig('pr_curves.png')
            print("\nSUCCESS: PR curves saved as pr_curves.png")
            
        else:
            print("\nFailed to plot PR curves.")

    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()