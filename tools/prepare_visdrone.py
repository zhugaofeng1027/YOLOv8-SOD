from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def _convert_annotation_file(label_src: Path, label_dst: Path, w: int, h: int) -> None:
    lines = label_src.read_text(encoding="utf-8").strip().splitlines() if label_src.exists() else []
    yolo_lines: list[str] = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        x, y, bw, bh, _, cls_id, _, _ = parts[:8]
        x = float(x)
        y = float(y)
        bw = float(bw)
        bh = float(bh)
        cls_id = int(cls_id)

        # VisDrone 1~10 are valid detection categories, others are ignored.
        if cls_id < 1 or cls_id > 10 or bw <= 0 or bh <= 0:
            continue

        xc = (x + bw / 2.0) / w
        yc = (y + bh / 2.0) / h
        nw = bw / w
        nh = bh / h
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        yolo_lines.append(f"{cls_id - 1} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    label_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.write_text("\n".join(yolo_lines), encoding="utf-8")


def _convert_split(src_split_dir: Path, out_root: Path, split_name: str, has_labels: bool = True) -> None:
    img_src_dir = src_split_dir / "images"
    ann_src_dir = src_split_dir / "annotations"
    img_dst_dir = out_root / "images" / split_name
    lbl_dst_dir = out_root / "labels" / split_name
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    if has_labels:
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(img_src_dir.glob("*.jpg")) + list(img_src_dir.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_src_dir}")

    for img_path in image_paths:
        dst_img_path = img_dst_dir / img_path.name
        shutil.copy2(img_path, dst_img_path)

        if has_labels:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            ann_path = ann_src_dir / f"{img_path.stem}.txt"
            out_label_path = lbl_dst_dir / f"{img_path.stem}.txt"
            _convert_annotation_file(ann_path, out_label_path, w=w, h=h)

    print(f"[Done] split={split_name}, images={len(image_paths)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone2019-DET annotations to YOLO format.")
    parser.add_argument("--src-root", type=str, required=True, help="Root containing train/val/test-dev folders.")
    parser.add_argument("--out-root", type=str, default="datasets/VisDrone2019-DET-YOLO")
    parser.add_argument("--with-test", action="store_true", help="Also copy test-dev images (without labels).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    train_dir = src_root / "VisDrone2019-DET-train"
    val_dir = src_root / "VisDrone2019-DET-val"
    test_dir = src_root / "VisDrone2019-DET-test-dev"

    _convert_split(train_dir, out_root, "train", has_labels=True)
    _convert_split(val_dir, out_root, "val", has_labels=True)
    if args.with_test and test_dir.exists():
        _convert_split(test_dir, out_root, "test", has_labels=False)

    print(f"[Done] YOLO dataset written to: {out_root}")


if __name__ == "__main__":
    main()
