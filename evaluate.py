"""
Evaluate a trained YOLOv8 model on the test set.

Usage:
    python evaluate.py --weights runs/train/boat_yolov8/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


BASE_DIR  = Path(__file__).parent
YAML_PATH = BASE_DIR / "dataset.yaml"
RUNS_DIR  = BASE_DIR / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 boat detection model")
    parser.add_argument(
        "--weights",
        default=str(RUNS_DIR / "train" / "boat_yolov8" / "weights" / "best.pt"),
        help="Path to trained model weights",
    )
    parser.add_argument("--imgsz",  type=int,   default=640,   help="Input image size")
    parser.add_argument("--batch",  type=int,   default=16,    help="Batch size")
    parser.add_argument("--conf",   type=float, default=0.25,  help="Confidence threshold")
    parser.add_argument("--iou",    type=float, default=0.5,   help="IoU threshold for NMS")
    parser.add_argument("--device", default="",                help="Device: '' (auto), 'cpu', '0'")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    # Validate on test split
    metrics = model.val(
        data    = str(YAML_PATH),
        split   = "test",
        imgsz   = args.imgsz,
        batch   = args.batch,
        conf    = args.conf,
        iou     = args.iou,
        device  = args.device,
        project = str(RUNS_DIR / "eval"),
        name    = "test_results",
        plots   = True,
        verbose = True,
    )

    print("\n── Test Set Metrics ──────────────────────────────")
    print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"  Precision:     {metrics.box.mp:.4f}")
    print(f"  Recall:        {metrics.box.mr:.4f}")
    print("──────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
