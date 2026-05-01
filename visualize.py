"""
Visualize dataset samples and training results.

Usage:
    python visualize.py --mode dataset   # show random training samples with GT boxes
    python visualize.py --mode results   # plot training curves
    python visualize.py --mode predict   # run model on random test images and show predictions
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


DATASET_DIR = Path("dataset")
RUNS_DIR = Path("runs/detect/runs/ship_detection")


def show_dataset_samples(n: int = 9):
    img_dir = DATASET_DIR / "images" / "train"
    lbl_dir = DATASET_DIR / "labels" / "train"

    images = sorted(img_dir.glob("*.png"))
    if not images:
        print("No training images found. Run prepare_dataset.py first.")
        return

    samples = random.sample(images, min(n, len(images)))
    cols = 3
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, img_path in zip(axes, samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        ax.imshow(img)

        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = list(map(float, line.split()))
                _, cx, cy, bw, bh = parts
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                rect = patches.Rectangle(
                    (x, y), bw * w, bh * h,
                    linewidth=2, edgecolor="lime", facecolor="none"
                )
                ax.add_patch(rect)

        ax.set_title(img_path.stem, fontsize=9)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.suptitle("Dataset Samples (green = ground truth)", fontsize=13)
    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150)
    plt.show()
    print("Saved to dataset_samples.png")


def show_training_results():
    results_csv = RUNS_DIR / "results.csv"
    if not results_csv.exists():
        print(f"results.csv not found at {results_csv}. Run train.py first.")
        return

    import csv
    data: dict[str, list[float]] = {}
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                k = k.strip()
                try:
                    data.setdefault(k, []).append(float(v))
                except ValueError:
                    pass

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def plot(ax, key, title, color):
        if key in data:
            ax.plot(data[key], color=color)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)

    plot(axes[0, 0], "train/box_loss", "Train Box Loss", "tab:blue")
    plot(axes[0, 1], "val/box_loss", "Val Box Loss", "tab:orange")
    plot(axes[1, 0], "metrics/mAP50(B)", "mAP@50", "tab:green")
    plot(axes[1, 1], "metrics/mAP50-95(B)", "mAP@50-95", "tab:red")

    plt.suptitle("Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved to training_curves.png")


def show_predictions(n: int = 6):
    from ultralytics import YOLO

    model_path = RUNS_DIR / "weights/best.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run train.py first.")
        return

    model = YOLO(str(model_path))
    img_dir = DATASET_DIR / "images" / "test"
    images = sorted(img_dir.glob("*.png"))
    if not images:
        print("No test images found.")
        return

    samples = random.sample(images, min(n, len(images)))
    results = model.predict(source=[str(p) for p in samples], conf=0.25, verbose=False)

    cols = 3
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, r in zip(axes, results):
        img = r.plot()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        n_det = len(r.boxes) if r.boxes is not None else 0
        ax.set_title(f"{Path(r.path).stem} — {n_det} ship(s)", fontsize=9)
        ax.axis("off")

    for ax in axes[len(results):]:
        ax.axis("off")

    plt.suptitle("Model Predictions on Test Set", fontsize=13)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.show()
    print("Saved to predictions.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dataset", "results", "predict"],
        default="dataset",
        help="What to visualize",
    )
    args = parser.parse_args()

    if args.mode == "dataset":
        show_dataset_samples()
    elif args.mode == "results":
        show_training_results()
    elif args.mode == "predict":
        show_predictions()


if __name__ == "__main__":
    main()
