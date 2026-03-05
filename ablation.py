from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class ECA(nn.Module):
    """Efficient Channel Attention for 2D feature maps."""

    def __init__(self, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


def register_custom_modules() -> None:
    """Make custom modules visible to ultralytics parse_model()."""
    import ultralytics.nn.tasks as tasks

    if getattr(tasks, "ECA", None) is None:
        tasks.ECA = ECA


def _load_ultralytics_yaml(use_p2_head: bool) -> dict[str, Any]:
    import ultralytics

    cfg_dir = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "v8"
    model_name = "yolov8-p2.yaml" if use_p2_head else "yolov8.yaml"
    yaml_path = cfg_dir / model_name
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _insert_eca_to_backbone(
    backbone: list[list[Any]], k_size: int = 3
) -> tuple[list[list[Any]], dict[int, int], int]:
    new_backbone: list[list[Any]] = []
    old_to_new: dict[int, int] = {}
    for layer in backbone:
        old_idx = len(old_to_new)
        new_backbone.append(layer)
        old_to_new[old_idx] = len(new_backbone) - 1
        if len(layer) >= 3 and layer[2] == "C2f":
            new_backbone.append([-1, 1, "ECA", [k_size]])
    num_inserted = len(new_backbone) - len(backbone)
    return new_backbone, old_to_new, num_inserted


def _remap_head_indices(
    head: list[list[Any]], backbone_len: int, old_to_new: dict[int, int], num_inserted: int
) -> list[list[Any]]:
    def _map_index(idx: int) -> int:
        if idx < 0:
            return idx
        if idx < backbone_len:
            return old_to_new[idx]
        return idx + num_inserted

    new_head: list[list[Any]] = []
    for layer in head:
        f, n, m, args = layer
        if isinstance(f, list):
            f = [_map_index(int(x)) for x in f]
        else:
            f = _map_index(int(f))
        new_head.append([f, n, m, args])
    return new_head


def create_model_yaml(
    output_dir: str | Path,
    nc: int,
    module_a: bool,
    module_b: bool,
    eca_kernel_size: int = 3,
) -> Path:
    """
    Build a YOLOv8s model yaml with optional A/B modules.

    A: ECA in backbone.
    B: P2 Detection Head.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_ultralytics_yaml(use_p2_head=module_b)
    cfg["nc"] = int(nc)
    if module_a:
        old_backbone = cfg["backbone"]
        old_head = cfg["head"]
        new_backbone, old_to_new, num_inserted = _insert_eca_to_backbone(old_backbone, k_size=eca_kernel_size)
        new_head = _remap_head_indices(old_head, len(old_backbone), old_to_new, num_inserted)
        cfg["backbone"] = new_backbone
        cfg["head"] = new_head

    tag = "baseline"
    if module_a or module_b:
        tags = []
        if module_a:
            tags.append("A")
        if module_b:
            tags.append("B")
        tag = "".join(tags)

    out_path = output_dir / f"yolov8s_visdrone_{tag}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return out_path


def _bbox_iou_siou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """SIoU for xyxy boxes (N, 4)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = (b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)).clamp(min=eps)
    ch = (b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)).clamp(min=eps)

    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
    sigma = torch.sqrt(s_cw.pow(2) + s_ch.pow(2) + eps)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = math.sqrt(2) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)

    angle_cost = torch.cos(torch.asin(sin_alpha).mul(2) - math.pi / 2)
    rho_x = (s_cw / cw).pow(2)
    rho_y = (s_ch / ch).pow(2)
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    omiga_w = torch.abs(w1 - w2) / torch.maximum(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.maximum(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-omiga_w), 4) + torch.pow(1 - torch.exp(-omiga_h), 4)

    siou = iou - 0.5 * (distance_cost + shape_cost)
    return siou


def apply_siou_patch() -> None:
    """Monkey-patch ultralytics BboxLoss.forward to use SIoU instead of CIoU."""
    import ultralytics.utils.loss as yloss

    if getattr(yloss.BboxLoss, "_siou_patched", False):
        return

    def _forward_siou(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = _bbox_iou_siou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = yloss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = yloss.bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    yloss.BboxLoss.forward = _forward_siou
    yloss.BboxLoss._siou_patched = True
