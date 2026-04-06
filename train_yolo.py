import torch
from ultralytics import YOLO

def main():
    # ------------------- GPU TRAINING OPTIMIZATION --------------------------
    device = 0 if torch.cuda.is_available() else 'cpu'
    gpu_mem = 0
    if device != 'cpu':
        props = torch.cuda.get_device_properties(device)
        gpu_mem = props.total_memory // (1024 ** 3)

    batch_size = max(1, min(32, gpu_mem * 4))
    print(f"Using device {device}, total GPU memory: {gpu_mem}GB → batch size: {batch_size}")

    # ----------------------- ACTUAL TRAINING --------------------------------

    # Load model
    model = YOLO("yolo12n.pt")

    # Model start training command
    model.train(
        data="yolo.yaml",
        epochs=30,
        imgsz=640,
        batch=batch_size,
        device=device,
        workers=6,
        pretrained=True, # transfer learning
        name="kitti_mosaic_train_final",
        half=True, # The last 3 are to make training a bit faster.
        rect=True,
        plots=False
    )

# Necessity to use this wrap on windows when multiprocessing
if __name__ == "__main__":
    main()