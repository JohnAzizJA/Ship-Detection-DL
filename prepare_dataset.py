"""
Convert Pascal VOC XML annotations to YOLO format and split into train/val/test sets.
Run this once before training.
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path


DATA_DIR = Path("data2")
DATASET_DIR = Path("dataset")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test = remaining 0.1

SEED = 42


def voc_to_yolo(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in ("boat", "ship"):
            continue
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    xml_files = sorted(DATA_DIR.glob("annotations/*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {DATA_DIR / 'annotations'}")

    print(f"Found {len(xml_files)} annotations.")

    random.seed(SEED)
    random.shuffle(xml_files)

    n = len(xml_files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train : n_train + n_val],
        "test": xml_files[n_train + n_val :],
    }

    for split, files in splits.items():
        print(f"  {split}: {len(files)} images")

    for split in ("train", "val", "test"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped = 0
    for split, files in splits.items():
        for xml_path in files:
            stem = xml_path.stem
            img_path = DATA_DIR / "images" / f"{stem}.png"

            if not img_path.exists():
                skipped += 1
                continue

            label_lines = voc_to_yolo(xml_path)
            if not label_lines:
                skipped += 1
                continue

            shutil.copy(img_path, DATASET_DIR / "images" / split / img_path.name)

            label_file = DATASET_DIR / "labels" / split / f"{stem}.txt"
            label_file.write_text("\n".join(label_lines))

    if skipped:
        print(f"Skipped {skipped} files (missing image or no valid boxes).")

    yaml_content = f"""\
path: {DATASET_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['boat']
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nDataset ready. Config saved to {yaml_path}")


if __name__ == "__main__":
    main()
