"""
Run inference with a trained YOLOv8 model on images or a directory.

Usage:
    python predict.py --source data2/images/boat0.png
    python predict.py --source data2/images/
    python predict.py --source dataset/test/images/ --save
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 boat detection inference")
    parser.add_argument(
        "--weights",
        default=str(RUNS_DIR / "train" / "boat_yolov8" / "weights" / "best.pt"),
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--source",
        default=str(BASE_DIR / "dataset" / "test" / "images"),
        help="Image path, directory, or glob pattern",
    )
    parser.add_argument("--imgsz",  type=int,   default=640,  help="Input image size")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",    type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", default="",               help="Device: '' (auto), 'cpu', '0'")
    parser.add_argument("--save",   action="store_true",      help="Save annotated images")
    parser.add_argument("--show",   action="store_true",      help="Display results (requires display)")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source  = args.source,
        imgsz   = args.imgsz,
        conf    = args.conf,
        iou     = args.iou,
        device  = args.device,
        save    = args.save,
        show    = args.show,
        project = str(RUNS_DIR / "predict"),
        name    = "boat_detections",
        verbose = True,
    )

    # Summary
    total_boxes = sum(len(r.boxes) for r in results)
    print(f"\nProcessed {len(results)} image(s) — {total_boxes} boat(s) detected.")

    if args.save:
        print(f"Annotated images saved to: {RUNS_DIR / 'predict' / 'boat_detections'}")


if __name__ == "__main__":
    main()
