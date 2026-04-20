# YOLOv11-seg Pedestrian & Vehicle Instance Segmentation

<video src="results/segmented_output.mp4" controls width="100%"></video>

![Inference demo](results/demo.png)

A complete training and inference pipeline for instance segmentation of pedestrians and 5 vehicle categories, built on YOLOv11-seg and COCO 2017.

## Target Classes

| ID | Class | COCO Original ID | Color |
|----|-------|-----------------|-------|
| 0  | person | 1 | Green |
| 1  | bicycle | 2 | Orange |
| 2  | car | 3 | Blue |
| 3  | motorcycle | 4 | Purple |
| 4  | bus | 6 | Cyan |
| 5  | truck | 8 | Red |

## Project Structure

```
yolo-v11-seg/
├── configs/
│   └── train_config.yaml       # Training hyperparameter config
├── data/
│   ├── coco/                   # ← Download COCO dataset here
│   │   ├── annotations/
│   │   │   ├── instances_train2017.json
│   │   │   └── instances_val2017.json
│   │   └── images/
│   │       ├── train2017/
│   │       └── val2017/
│   └── pedestrian_vehicle/     # Auto-generated after prepare_dataset.py
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       └── dataset.yaml
├── models/                     # Pretrained weights (auto-downloaded on first run)
├── results/                    # Inference outputs
├── runs/                       # Training logs (created by ultralytics)
└── scripts/
    ├── prepare_dataset.py      # COCO → YOLO format conversion
    ├── train.py                # Model training
    ├── evaluate.py             # Validation metrics / image prediction
    ├── video_test.py           # Video inference + save results
    └── realtime.py             # Live webcam / stream detection
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download COCO 2017 Dataset

The full dataset is ~**25 GB**. To verify the pipeline quickly, start with val2017 only (~1 GB).

```bash
cd data/coco

# Annotation files (~241 MB, required)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Validation images (~1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/

# Training images (~18 GB, required for training)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d images/

cd ../..
```

Windows (PowerShell):
```powershell
Invoke-WebRequest http://images.cocodataset.org/zips/val2017.zip -OutFile val2017.zip
```

### 3. Prepare Dataset

Converts COCO polygon annotations to YOLO segmentation format, filtering to the 6 target classes:

```bash
python scripts/prepare_dataset.py
# Optional: --coco_dir data/coco --out_dir data/pedestrian_vehicle
```

This generates images, label files, and `dataset.yaml` under `data/pedestrian_vehicle/`.

---

## Training

```bash
# Default config (configs/train_config.yaml)
python scripts/train.py

# Quick test (nano model, 10 epochs)
python scripts/train.py --model yolo11n-seg.pt --epochs 10 --batch 8

# Higher accuracy
python scripts/train.py --model yolo11m-seg.pt --epochs 150 --batch 8

# Resume from checkpoint
python scripts/train.py --resume runs/train/ped_vehicle_seg/weights/last.pt
```

Training outputs are saved to `runs/train/ped_vehicle_seg/`:

| File | Description |
|------|-------------|
| `weights/best.pt` | Best checkpoint on validation set |
| `weights/last.pt` | Last epoch checkpoint |
| `results.csv` | Training curve data |
| `confusion_matrix.png`, `PR_curve.png` | Evaluation plots |

---

## Evaluation

```bash
# Evaluate on val set, prints both segmentation and detection mAP
python scripts/evaluate.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt

# Run prediction on an image directory and save visualizations
python scripts/evaluate.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --predict --source path/to/images/ --save_dir results/predictions
```

---

## Video Inference

```bash
# Basic (output saved as results/<video_name>_pred.mp4)
python scripts/video_test.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt \
    --source  path/to/video.mp4

# With live preview window
python scripts/video_test.py --weights best.pt --source video.mp4 --preview

# Boxes only (skip mask drawing, faster)
python scripts/video_test.py --weights best.pt --source video.mp4 --no_mask
```

Output video includes semi-transparent segmentation masks (45% opacity), contours, bounding boxes, confidence labels, and a real-time FPS/count statistics panel.

---

## Real-time Detection

```bash
# Default webcam (index 0)
python scripts/realtime.py \
    --weights runs/train/ped_vehicle_seg/weights/best.pt

# Specific camera or RTSP stream
python scripts/realtime.py --weights best.pt --source 1
python scripts/realtime.py --weights best.pt --source rtsp://192.168.1.100/stream

# Record video simultaneously
python scripts/realtime.py --weights best.pt --save
```

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| M | Toggle segmentation mask display |
| S | Screenshot (saved to results/screenshots/) |
| + / - | Increase / decrease confidence threshold |
| R | Reset statistics |

---

## Configuration

Edit `configs/train_config.yaml` to adjust training parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Pretrained weights | yolo11n-seg.pt |
| `epochs` | Training epochs | 100 |
| `batch` | Batch size | 16 |
| `imgsz` | Input image size | 640 |
| `device` | GPU index or `cpu` | 0 |
| `patience` | Early stopping patience | 20 |
| `optimizer` | Optimizer | AdamW |
| `lr0` | Initial learning rate | 0.01 |
| `copy_paste` | Copy-Paste augmentation | 0.1 |

All `train.py` CLI arguments override their corresponding YAML values.

---

## Troubleshooting

**Out of memory (OOM):** Reduce `batch` (to 8 or 4), use a smaller model (`yolo11n-seg`), or reduce `imgsz` (e.g., 416).

**Windows multiprocessing error:** Set `workers` to 0 or 2 in `train_config.yaml`.

**Already have a COCO dataset:** Point `--coco_dir` to your existing dataset path. Use `--no_copy` to skip image copying and only generate label files.
