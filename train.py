"""
Train YOLOv8s on the ship detection dataset.
Run prepare_dataset.py first.
"""

from pathlib import Path
import torch
from ultralytics import YOLO


DATASET_YAML = Path("dataset/dataset.yaml")
MODEL = "yolov8s.pt"
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16
PROJECT = "runs"
NAME = "ship_detection"


def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError("dataset/dataset.yaml not found. Run prepare_dataset.py first.")

    model = YOLO(MODEL)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        patience=20,
        device="cpu",
        augment=True,
        degrees=10.0,
        flipud=0.2,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        save_period=10,
        verbose=True,
    )

    print(f"\nTraining complete. Best model: {Path(PROJECT) / NAME / 'weights/best.pt'}")
    return results


if __name__ == "__main__":
    main()
