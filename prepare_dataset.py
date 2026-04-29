"""
Convert Pascal VOC XML annotations to YOLO format and split into train/val/test.

YOLO label format per line: <class_id> <x_center> <y_center> <width> <height>
All values normalized to [0, 1] relative to image dimensions.
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA2_DIR   = BASE_DIR / "data2"
IMAGES_DIR  = DATA2_DIR / "images"
ANNOT_DIR   = DATA2_DIR / "annotations"

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "val"
TEST_DIR    = DATASET_DIR / "test"

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
SEED        = 42

CLASS_NAMES = ["boat"]   # single class


def parse_voc_xml(xml_path: Path):
    """Return image size and list of (class_id, xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width  = int(size.find("width").text)
    height = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_NAMES:
            continue
        class_id = CLASS_NAMES.index(name)
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append((class_id, xmin, ymin, xmax, ymax))

    return width, height, boxes


def voc_to_yolo(class_id, xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert absolute VOC box to normalized YOLO format."""
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h
    return class_id, x_center, y_center, width, height


def create_dirs():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)


def copy_sample(stem: str, split_dir: Path):
    """Copy image and write YOLO label for one sample."""
    xml_path = ANNOT_DIR / f"{stem}.xml"
    img_path = IMAGES_DIR / f"{stem}.png"

    if not xml_path.exists() or not img_path.exists():
        return False

    img_w, img_h, boxes = parse_voc_xml(xml_path)
    if not boxes:
        return False

    # Copy image
    shutil.copy(img_path, split_dir / "images" / f"{stem}.png")

    # Write YOLO label
    label_path = split_dir / "labels" / f"{stem}.txt"
    with open(label_path, "w") as f:
        for (cls, xmin, ymin, xmax, ymax) in boxes:
            cls, xc, yc, w, h = voc_to_yolo(cls, xmin, ymin, xmax, ymax, img_w, img_h)
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    return True


def main():
    create_dirs()

    # Collect all stems that have both image and annotation
    stems = []
    for xml_file in sorted(ANNOT_DIR.glob("*.xml")):
        stem = xml_file.stem
        if (IMAGES_DIR / f"{stem}.png").exists():
            stems.append(stem)

    print(f"Total valid samples: {len(stems)}")

    random.seed(SEED)
    random.shuffle(stems)

    n_total = len(stems)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_stems = stems[:n_train]
    val_stems   = stems[n_train:n_train + n_val]
    test_stems  = stems[n_train + n_val:]

    counts = {"train": 0, "val": 0, "test": 0}
    for stem in train_stems:
        if copy_sample(stem, TRAIN_DIR):
            counts["train"] += 1
    for stem in val_stems:
        if copy_sample(stem, VAL_DIR):
            counts["val"] += 1
    for stem in test_stems:
        if copy_sample(stem, TEST_DIR):
            counts["test"] += 1

    print(f"Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")

    # Write dataset YAML for YOLO
    yaml_content = f"""path: {DATASET_DIR.as_posix()}
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = BASE_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Dataset YAML written to: {yaml_path}")


if __name__ == "__main__":
    main()
