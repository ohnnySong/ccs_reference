#!/usr/bin/env python3
"""Train document-aligned opacity, contraction, and WTW area tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from ccs_model.architecture_document_v1 import CcsDocumentMultiTaskV1
from ccs_model.metrics_v4 import area_metrics, multiclass_metrics
from ccs_model.train_v4 import CcsDataset, _load_rows, _loader, _move


OPACITY_NAMES = ("opacity_0", "opacity_1", "opacity_2", "opacity_3")
CONTRACTION_NAMES = ("contraction_0", "contraction_1", "contraction_2_or_3")


def contraction_class(area_mm2: float) -> int:
    return 0 if area_mm2 > 20.0 else 1 if area_mm2 > 16.0 else 2


class DocumentDataset(CcsDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["opacity"] = torch.tensor(int(self.rows[index]["ccs_grade"]), dtype=torch.long)
        item["contraction"] = torch.tensor(
            contraction_class(float(self.rows[index]["opening_area_mm2"])), dtype=torch.long
        )
        item.pop("severity")
        return item


def _weights(labels: list[int], classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=classes).float()
    weights = torch.sqrt(counts.sum() / counts)
    return (weights / weights.mean()).to(device)


def _loss(output, batch, opacity_weights, contraction_weights):
    opacity_ce = F.cross_entropy(
        output.opacity_logits, batch["opacity"], weight=opacity_weights, label_smoothing=0.03
    )
    opacity_index = torch.arange(4, device=batch["opacity"].device, dtype=torch.float32)
    opacity_expected = (output.opacity_probabilities * opacity_index).sum(1)
    opacity_ordinal = F.smooth_l1_loss(opacity_expected, batch["opacity"].float(), beta=0.5)

    contraction_ce = F.cross_entropy(
        output.contraction_logits,
        batch["contraction"],
        weight=contraction_weights,
        label_smoothing=0.02,
    )
    contraction_index = torch.arange(3, device=batch["contraction"].device, dtype=torch.float32)
    contraction_expected = (output.contraction_probabilities * contraction_index).sum(1)
    contraction_ordinal = F.smooth_l1_loss(
        contraction_expected, batch["contraction"].float(), beta=0.5
    )

    target_fraction = batch["area_mm2"] / batch["wtw_mm"].square()
    predicted_fraction = output.opening_area_mm2 / batch["wtw_mm"].square()
    area = F.smooth_l1_loss(predicted_fraction, target_fraction, beta=0.02)
    boundary_losses = []
    for threshold in (16.0, 20.0):
        target = (batch["area_mm2"] > threshold).float()
        logit = (output.opening_area_mm2 - threshold) / 2.0
        boundary_losses.append(F.binary_cross_entropy_with_logits(logit, target))
    boundary = torch.stack(boundary_losses).mean()
    total = (
        0.85 * opacity_ce + 0.15 * opacity_ordinal
        + 0.75 * contraction_ce + 0.10 * contraction_ordinal
        + 3.0 * area + 0.15 * boundary
    )
    return total


def _evaluate(model, loader, device, opacity_weights, contraction_weights):
    model.eval()
    opacity_ref, opacity_pred = [], []
    contraction_ref, contraction_pred = [], []
    area_ref, area_pred = [], []
    losses = []
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(batch["image"], batch["wtw_mm"])
                loss = _loss(output, batch, opacity_weights, contraction_weights)
            losses.append(float(loss))
            opacity_ref.extend(int(x) for x in batch["opacity"].cpu())
            opacity_pred.extend(int(x) for x in output.opacity_logits.argmax(1).cpu())
            contraction_ref.extend(int(x) for x in batch["contraction"].cpu())
            contraction_pred.extend(int(x) for x in output.contraction_logits.argmax(1).cpu())
            area_ref.extend(float(x) for x in batch["area_mm2"].cpu())
            area_pred.extend(float(x) for x in output.opening_area_mm2.cpu())
    opacity = multiclass_metrics(opacity_ref, opacity_pred, OPACITY_NAMES)
    contraction = multiclass_metrics(contraction_ref, contraction_pred, CONTRACTION_NAMES)
    area = area_metrics(area_ref, area_pred)
    score = (
        0.20 * opacity["macro_f1"] + 0.20 * opacity["balanced_accuracy"]
        + 0.20 * contraction["macro_f1"] + 0.15 * contraction["balanced_accuracy"]
        + 0.15 * max(0.0, 1.0 - area["mae_mm2"] / 10.0)
        + 0.10 * max(0.0, 1.0 - area["rmse_mm2"] / 12.0)
    )
    return {
        "loss": sum(losses) / len(losses), "selection_score": score,
        "opacity_0_1_2_3": opacity, "contraction_0_1_23": contraction,
        "opening_area": area,
    }


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    torch.autograd.set_multithreading_enabled(False)
    device = torch.device("cuda")
    train_rows = _load_rows(args.manifest, "train")
    val_rows = _load_rows(args.manifest, "val")
    train_set = DocumentDataset(train_rows, args.data_root, size=args.image_size, train=True)
    val_set = DocumentDataset(val_rows, args.data_root, size=args.image_size, train=False)
    train_loader = _loader(train_set, batch_size=args.batch_size, workers=args.workers, seed=args.seed, balanced=False)
    val_loader = _loader(val_set, batch_size=args.batch_size, workers=args.workers, seed=args.seed, balanced=False)
    opacity_weights = _weights([int(r["ccs_grade"]) for r in train_rows], 4, device)
    contraction_weights = _weights([contraction_class(float(r["opening_area_mm2"])) for r in train_rows], 3, device)
    model = CcsDocumentMultiTaskV1(pretrained=not args.no_pretrained).to(device)
    if args.encoder_checkpoint is not None:
        source = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in source["state_dict"].items()
            if key.startswith("encoder.")
        }
        model.encoder.load_state_dict(encoder_state, strict=True)
    model.encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {"params": model.opacity_head.parameters(), "lr": args.head_lr},
        {"params": model.contraction_head.parameters(), "lr": args.head_lr},
        {"params": model.area_fraction_head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.backbone_lr * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    history=[]; best=-math.inf; stale=0
    for epoch in range(1,args.epochs+1):
        model.train(); model.encoder.eval(); running=[]
        for raw in train_loader:
            batch=_move(raw,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                output=model(batch["image"],batch["wtw_mm"])
                loss=_loss(output,batch,opacity_weights,contraction_weights)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); running.append(float(loss.detach()))
        scheduler.step()
        validation=_evaluate(model,val_loader,device,opacity_weights,contraction_weights)
        record={"epoch":epoch,"train_loss":sum(running)/len(running),"validation":validation}
        history.append(record); print(json.dumps(record),flush=True)
        checkpoint={"architecture":"ccs.document-aligned.v1","state_dict":model.state_dict(),"epoch":epoch,"validation":validation,
                    "definitions":{"opacity":"0/1/2/3 from filename","contraction":"0:A>20; 1:16<A<=20; 2_or_3:A<=16","area":"area_fraction*WTW_mm^2"}}
        torch.save(checkpoint,args.output/"checkpoint_last.pt")
        if validation["selection_score"]>best:
            best=float(validation["selection_score"]); stale=0; torch.save(checkpoint,args.output/"checkpoint_best.pt")
        else: stale+=1
        (args.output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        if stale>=args.patience: break
    (args.output/"run_summary.json").write_text(json.dumps({"status":"complete","best_selection_score":best,"epochs_completed":len(history),"train_records":len(train_rows),"validation_records":len(val_rows),"test_accessed":False,"evidence_type":"newly_run"},indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--image-size",type=int,default=320); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--workers",type=int,default=2)
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--patience",type=int,default=7); p.add_argument("--backbone-lr",type=float,default=3e-5); p.add_argument("--head-lr",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=0.03); p.add_argument("--seed",type=int,default=20260923); p.add_argument("--no-pretrained",action="store_true"); p.add_argument("--encoder-checkpoint",type=Path); train(p.parse_args())


if __name__ == "__main__": main()
