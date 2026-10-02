#!/usr/bin/env python3
"""Subset the frozen feature cache to the cleaned unique-image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--source-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_manifest)
    clean_rows = read_jsonl(args.clean_manifest)
    chunks = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(args.source_feature_dir.glob("chunk_*.pt"))
    ]
    source_features = torch.cat([chunk["features"] for chunk in chunks])
    if len(source_rows) != len(source_features):
        raise ValueError("source manifest and feature counts differ")
    index_by_sample = {
        str(row["sample_id"]): index for index, row in enumerate(source_rows)
    }
    indices = [index_by_sample[str(row["sample_id"])] for row in clean_rows]
    features = source_features[torch.tensor(indices)]
    labels = torch.tensor([int(row["ccs_grade"]) for row in clean_rows])
    splits = [str(row["split"]) for row in clean_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "start": 0,
            "end": len(clean_rows),
            "features": features,
            "labels": labels,
            "splits": splits,
        },
        args.output_dir / "chunk_0000.pt",
    )
    audit = {
        "source_records": len(source_rows),
        "clean_records": len(clean_rows),
        "train_records": splits.count("train"),
        "validation_records": splits.count("val"),
        "feature_width": features.shape[1],
        "missing_samples": 0,
    }
    (args.output_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
