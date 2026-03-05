from __future__ import annotations

import argparse

from ultralytics import YOLO

from ablation import register_custom_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with a trained YOLOv8 checkpoint.")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--source", type=str, required=True, help="Image/video/folder path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true", help="Save prediction visualizations.")
    parser.add_argument("--project", type=str, default="runs/predict")
    parser.add_argument("--name", type=str, default="exp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_custom_modules()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        project=args.project,
        name=args.name,
    )

    print(f"[Done] Predicted {len(results)} samples.")
    for idx, r in enumerate(results):
        print(f"  - sample={idx} boxes={len(r.boxes) if r.boxes is not None else 0}")


if __name__ == "__main__":
    main()
