"""
Evaluate the trained model on the test set.
"""

from pathlib import Path
from ultralytics import YOLO


BEST_MODEL = Path("runs/ship_detection/weights/best.pt")
DATASET_YAML = Path("dataset/dataset.yaml")


def main():
    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"Model not found at {BEST_MODEL}. Run train.py first.")

    model = YOLO(str(BEST_MODEL))

    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        verbose=True,
    )

    print("\n--- Test Set Results ---")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")

    return metrics


if __name__ == "__main__":
    main()
