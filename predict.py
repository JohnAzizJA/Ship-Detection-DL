"""
Run inference on a single image or a directory of images.

Usage:
    python predict.py --source path/to/image.png
    python predict.py --source path/to/folder/
    python predict.py --source data2/images/boat0.png
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


BEST_MODEL = Path("runs/ship_detection/weights/best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image or directory path")
    parser.add_argument("--model", default=str(BEST_MODEL), help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--save-dir", default="runs/predict", help="Output directory")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    model = YOLO(str(model_path))

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.save_dir,
        name="",
        exist_ok=True,
        verbose=True,
    )

    print(f"\nResults saved to {args.save_dir}")

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes):
            print(f"{r.path}: {len(boxes)} ship(s) detected")
        else:
            print(f"{r.path}: no ships detected")

    return results


if __name__ == "__main__":
    main()
