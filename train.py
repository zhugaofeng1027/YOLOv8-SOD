from __future__ import annotations

import argparse
import platform
from pathlib import Path

import yaml


def _read_nc(data_yaml: str | Path) -> int:
    data_yaml = str(data_yaml)
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        from ultralytics.utils.checks import check_yaml

        resolved = check_yaml(data_yaml, hard=False)
        if resolved:
            yaml_path = Path(resolved)
        else:
            raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get("names")
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names.keys())
    nc = data_cfg.get("nc")
    if nc is None:
        raise ValueError(f"Cannot infer nc from data yaml: {data_yaml}")
    return int(nc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8s with A/B/C ablation modules.")
    parser.add_argument("--data", type=str, default="VisDrone.yaml", help="Dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    default_workers = 0 if platform.system().lower() == "windows" else 8
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--pretrained", type=str, default="yolov8s.pt", help="Pretrained weight for transfer.")
    parser.add_argument("--model-cache-dir", type=str, default="generated_models")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--module-a", action="store_true", help="Enable A: ECA in backbone.")
    parser.add_argument("--module-b", action="store_true", help="Enable B: P2 Detection Head.")
    parser.add_argument("--module-c", action="store_true", help="Enable C: SIoU loss.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO
    from ablation import apply_siou_patch, create_model_yaml, register_custom_modules

    register_custom_modules()
    if args.module_c:
        apply_siou_patch()

    nc = _read_nc(args.data)
    model_yaml = create_model_yaml(
        output_dir=args.model_cache_dir,
        nc=nc,
        module_a=args.module_a,
        module_b=args.module_b,
    )

    exp_suffix = "baseline"
    if args.module_a or args.module_b or args.module_c:
        tags = []
        if args.module_a:
            tags.append("A")
        if args.module_b:
            tags.append("B")
        if args.module_c:
            tags.append("C")
        exp_suffix = "".join(tags)

    run_name = args.name or f"yolov8s_visdrone_{exp_suffix}"
    model = YOLO(str(model_yaml))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=args.pretrained,
        project=args.project,
        name=run_name,
    )

    print(f"[Done] Training finished. Run name: {run_name}")
    print(f"[Info] Model yaml used: {model_yaml}")


if __name__ == "__main__":
    main()
