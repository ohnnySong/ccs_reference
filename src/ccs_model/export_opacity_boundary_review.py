#!/usr/bin/env python3
"""Export a blinded, de-identified review package for opacity 1/2 errors."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageOps
import torch

from ccs_model.train_opacity_features_v2 import Head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    args = parser.parse_args()

    chunks = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(args.feature_dir.glob("chunk_*.pt"))
    ]
    features = torch.cat([chunk["features"] for chunk in chunks])
    labels = torch.cat([chunk["labels"] for chunk in chunks])
    splits = sum([chunk["splits"] for chunk in chunks], [])
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if len(rows) != len(features):
        raise ValueError("manifest and feature counts differ")

    probabilities = []
    for checkpoint_path in sorted(args.model_root.glob("feature_seed*/checkpoint_best.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = Head()
        model.load_state_dict(checkpoint["head_state_dict"])
        model.eval()
        normalized = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        with torch.no_grad():
            probabilities.append(model(normalized).softmax(1))
    predictions = torch.stack(probabilities).mean(0).argmax(1)
    selected_indices = [
        index
        for index, (split, expected, predicted) in enumerate(
            zip(splits, labels.tolist(), predictions.tolist(), strict=True)
        )
        if split == "val" and (expected, predicted) in {(1, 2), (2, 1)}
    ]
    if len(selected_indices) != 23:
        raise ValueError(f"expected 23 opacity 1/2 boundary errors, found {len(selected_indices)}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, mode=0o700)
    mapping = []
    review_rows = []
    for review_index, row_index in enumerate(selected_indices, 1):
        row = rows[row_index]
        review_id = f"CCS12-{review_index:03d}"
        source = Path(str(row["local_image_path"]))
        if not source.is_absolute():
            source = args.data_root / source
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = image.size
        ImageDraw.Draw(image).rectangle(
            (0, 0, round(width * 0.35), round(height * 0.06)), fill=(0, 0, 0)
        )
        filename = f"{review_id}.jpg"
        image.save(args.output_dir / filename, "JPEG", quality=95, optimize=True, exif=b"")
        review_rows.append(
            {
                "复核编号": review_id,
                "医生复核等级": "",
                "是否修改原标签": "",
                "备注": "",
            }
        )
        mapping.append(
            {
                "review_id": review_id,
                "sample_id": row["sample_id"],
                "local_image_path": row["local_image_path"],
                "reference_grade": int(labels[row_index]),
                "model_prediction": int(predictions[row_index]),
            }
        )

    with (args.output_dir / "单医生复核表.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    (args.output_dir / "README.txt").write_text(
        "浑浊1-2级边界复核23张\n\n"
        "请医生在不知道原标签和模型预测的情况下填写单医生复核表。\n"
        "复核等级填写0、1、2或3；如需修改原标签，请填写“是”。\n"
        "图片已顺序重命名、移除EXIF并遮盖左上角信息区。\n",
        encoding="utf-8",
    )
    args.private_map.parent.mkdir(parents=True, exist_ok=True)
    args.private_map.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")
    with ZipFile(args.output_zip, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(args.output_dir.iterdir()):
            archive.write(path, arcname=f"浑浊1-2级边界复核23张/{path.name}")
    print(json.dumps({"images": len(selected_indices), "zip": str(args.output_zip)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
