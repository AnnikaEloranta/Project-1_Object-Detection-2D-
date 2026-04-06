# Project-1_Object-Detection-2D-


### Image size distribution (train set):
(1224, 370): 751 images
(1242, 375): 5895 images
(1238, 374): 344 images
(1241, 376): 291 images

## Data folder structure

```
project-root/
├── data/
|   ├── raw/                # TODO: DRAW THIS PATH 
│   ├── images/
│   │   ├── training/       # KITTI training images (split subset for internal training)
│   │   └── val/            # KITTI validation/internal test images
│   └── labels/
│       └── training/       # YOLO labels corresponding to training images
│       └── val/
└── mosaics/
    ├── images/
    │   ├── training/       # Mosaic-generated training tiles (640x640) from data/images/training
    │   └── val/            # Mosaic-generated validation tiles from data/images/test
    └── labels/
        └── training/       # YOLO labels for mosaic-generated training tiles
        └── val/ 
```