#!/usr/bin/env python3
"""Train CCS v4 severity and scalar opening-area heads.

The locked test split is rejected. Outputs are aggregate-only and checkpoints
contain no sample identifiers or paths.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF

from ccs_model.architecture_v4 import CcsMultiTaskV4, remap_source_grade
from ccs_model.metrics_v4 import area_metrics, severity_metrics


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row.get("split") in {"test", "external_test"} for row in rows):
        rows = [row for row in rows if row.get("split") in {"train", "val"}]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"manifest has no {split} records")
    required = ("local_image_path", "ccs_grade", "opening_area_mm2", "wtw_mm")
    if any(any(row.get(key) is None for key in required) for row in selected):
        raise ValueError(f"{split} contains missing CCS targets")
    return selected


class CcsDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        root: Path,
        *,
        size: int,
        train: bool,
        grade_mapping: str = "light_moderate_severe",
    ) -> None:
        self.rows = rows
        self.root = root
        self.size = size
        self.train = train
        self.grade_mapping = grade_mapping
        self.jitter = ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, path: Path) -> Tensor:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size))
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        image = canvas
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
            image = TF.rotate(
                image,
                random.uniform(-8.0, 8.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            image = self.jitter(image)
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor[:, : round(self.size * 0.06), : round(self.size * 0.35)] = 0.0
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.rows[index]
        source_grade = torch.tensor(int(row["ccs_grade"]), dtype=torch.long)
        if self.grade_mapping == "light_moderate_severe":
            severity = remap_source_grade(source_grade)
        elif self.grade_mapping == "grade_0_grade_1_grade_2_or_3":
            severity = source_grade.clamp_max(2)
        else:
            raise ValueError(f"unsupported grade mapping: {self.grade_mapping}")
        return {
            "image": self._image(self.root / str(row["local_image_path"])),
            "severity": severity,
            "area_mm2": torch.tensor(float(row["opening_area_mm2"]), dtype=torch.float32),
            "wtw_mm": torch.tensor(float(row["wtw_mm"]), dtype=torch.float32),
        }


def _loader(
    dataset: CcsDataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    balanced: bool,
    grade_mapping: str = "light_moderate_severe",
) -> DataLoader[dict[str, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = not balanced
    if balanced:
        if grade_mapping == "light_moderate_severe":
            labels = [
                0 if int(row["ccs_grade"]) <= 1 else int(row["ccs_grade"]) - 1
                for row in dataset.rows
            ]
        elif grade_mapping == "grade_0_grade_1_grade_2_or_3":
            labels = [min(int(row["ccs_grade"]), 2) for row in dataset.rows]
        else:
            raise ValueError(f"unsupported grade mapping: {grade_mapping}")
        counts = Counter(labels)
        weights = torch.tensor([counts[label] ** -0.35 for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _loss(outputs: Any, severity: Tensor, area_mm2: Tensor, wtw_mm: Tensor) -> tuple[Tensor, dict[str, float]]:
    ce = F.cross_entropy(outputs.severity_logits, severity, label_smoothing=0.03)
    indices = torch.arange(3, device=severity.device, dtype=torch.float32)
    expected = (outputs.severity_probabilities * indices).sum(dim=1)
    ordinal = F.smooth_l1_loss(expected, severity.float(), beta=0.5)
    collapsed = F.binary_cross_entropy(outputs.collapsed_probability, (severity >= 1).float())
    target_fraction = area_mm2 / wtw_mm.square()
    predicted_fraction = outputs.opening_area_mm2 / wtw_mm.square()
    area = F.smooth_l1_loss(predicted_fraction, target_fraction, beta=0.02)
    total = ce + 0.25 * ordinal + 0.20 * collapsed + 2.0 * area
    return total, {
        "total": float(total.detach()),
        "classification": float(ce.detach()),
        "ordinal": float(ordinal.detach()),
        "collapsed": float(collapsed.detach()),
        "area": float(area.detach()),
    }


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _evaluate(model: CcsMultiTaskV4, loader: DataLoader[dict[str, Tensor]], device: torch.device) -> dict[str, Any]:
    model.eval()
    reference_grade: list[int] = []
    predicted_grade: list[int] = []
    collapsed_probability: list[float] = []
    reference_area: list[float] = []
    predicted_area: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["image"], batch["wtw_mm"], compute_geometry=False)
                loss, _ = _loss(output, batch["severity"], batch["area_mm2"], batch["wtw_mm"])
            losses.append(float(loss))
            reference_grade.extend(int(x) for x in batch["severity"].cpu())
            predicted_grade.extend(int(x) for x in output.severity_logits.argmax(1).cpu())
            collapsed_probability.extend(float(x) for x in output.collapsed_probability.cpu())
            reference_area.extend(float(x) for x in batch["area_mm2"].cpu())
            predicted_area.extend(float(x) for x in output.opening_area_mm2.cpu())
    severity = severity_metrics(reference_grade, predicted_grade, collapsed_probability)
    area = area_metrics(reference_area, predicted_area)
    score = (
        0.45 * float(severity["macro_f1"])
        + 0.25 * float(severity["balanced_accuracy"])
        + 0.30 * max(0.0, 1.0 - area["mae_mm2"] / 10.0)
    )
    return {"loss": sum(losses) / len(losses), "selection_score": score, "severity": severity, "opening_area": area}


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested training run")

    train_rows = _load_rows(args.manifest, "train")
    val_rows = _load_rows(args.manifest, "val")
    train_set = CcsDataset(train_rows, args.data_root, size=args.image_size, train=True)
    val_set = CcsDataset(val_rows, args.data_root, size=args.image_size, train=False)
    train_loader = _loader(train_set, batch_size=args.batch_size, workers=args.workers, seed=args.seed, balanced=True)
    val_loader = _loader(val_set, batch_size=args.batch_size, workers=args.workers, seed=args.seed, balanced=False)

    model = CcsMultiTaskV4(pretrained=not args.no_pretrained).to(device)
    for module in (model.decoder, model.opening_head, model.wtw_endpoint_head):
        module.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.backbone_lr},
            {"params": model.severity_head.parameters(), "lr": args.head_lr},
            {"params": model.area_fraction_head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.backbone_lr * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    best_score = -math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in (model.decoder, model.opening_head, model.wtw_endpoint_head):
            module.eval()
        running: list[float] = []
        for raw in train_loader:
            batch = _move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(batch["image"], batch["wtw_mm"], compute_geometry=False)
                loss, _ = _loss(output, batch["severity"], batch["area_mm2"], batch["wtw_mm"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach()))
        scheduler.step()
        validation = _evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": sum(running) / len(running),
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "architecture": "ccs.multitask.v4",
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "validation": validation,
            "severity_mapping": {"0": "light", "1": "moderate", "2": "severe"},
        }
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if validation["selection_score"] > best_score:
            best_score = float(validation["selection_score"])
            stale = 0
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
        else:
            stale += 1
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= args.patience:
            break
    summary = {
        "status": "complete",
        "best_selection_score": best_score,
        "epochs_completed": len(history),
        "train_records": len(train_rows),
        "validation_records": len(val_rows),
        "test_accessed": False,
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--no-pretrained", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
