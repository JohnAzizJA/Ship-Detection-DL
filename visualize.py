"""
Visualize random samples from the dataset with their bounding boxes.

Usage:
    python visualize.py              # shows 9 random training samples
    python visualize.py --split val --n 6
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n",     type=int, default=9, help="Number of samples to show")
    parser.add_argument("--seed",  type=int, default=0)
    return parser.parse_args()


def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Convert YOLO normalized labels back to absolute pixel boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def main():
    args = parse_args()
    random.seed(args.seed)

    img_dir   = DATASET_DIR / args.split / "images"
    label_dir = DATASET_DIR / args.split / "labels"

    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        print(f"No images found in {img_dir}. Run prepare_dataset.py first.")
        return

    samples = random.sample(img_paths, min(args.n, len(img_paths)))

    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, img_path in zip(axes, samples):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        label_path = label_dir / (img_path.stem + ".txt")
        boxes = load_yolo_labels(label_path, w, h)

        ax.imshow(img_rgb)
        for (cls, x1, y1, x2, y2) in boxes:
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, "boat", color="red", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, pad=1))

        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.suptitle(f"{args.split.capitalize()} split — {len(samples)} samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"visualize_{args.split}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to visualize_{args.split}.png")


if __name__ == "__main__":
    main()
