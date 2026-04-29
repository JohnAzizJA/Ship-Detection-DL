"""
Train YOLOv8 on the boat detection dataset.

Usage:
    python train.py
    python train.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


BASE_DIR   = Path(__file__).parent
YAML_PATH  = BASE_DIR / "dataset.yaml"
RUNS_DIR   = BASE_DIR / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for boat detection")
    parser.add_argument("--model",   default="yolov8n.pt",  help="Pretrained model weights")
    parser.add_argument("--epochs",  type=int, default=100,  help="Number of training epochs")
    parser.add_argument("--imgsz",   type=int, default=640,  help="Input image size")
    parser.add_argument("--batch",   type=int, default=16,   help="Batch size (-1 = auto)")
    parser.add_argument("--patience",type=int, default=20,   help="Early stopping patience")
    parser.add_argument("--device",  default="",             help="Device: '' (auto), 'cpu', '0', '0,1'")
    parser.add_argument("--name",    default="boat_yolov8",  help="Run name")
    return parser.parse_args()


def main():
    args = parse_args()

    if not YAML_PATH.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {YAML_PATH}.\n"
            "Run prepare_dataset.py first."
        )

    model = YOLO(args.model)

    results = model.train(
        data      = str(YAML_PATH),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        patience  = args.patience,
        device    = args.device,
        project   = str(RUNS_DIR / "train"),
        name      = args.name,
        # Augmentation
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 10.0,
        translate = 0.1,
        scale     = 0.5,
        flipud    = 0.5,
        fliplr    = 0.5,
        mosaic    = 1.0,
        # Logging
        plots     = True,
        save      = True,
        verbose   = True,
    )

    print("\nTraining complete.")
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
