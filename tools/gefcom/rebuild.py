"""Convert provider GEFCom2014 Solar CSVs into audited forecasting tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from audit import run_audit, save_audit, sha256
from mapping import build_mapping
from windowing import build_windows, save_windows


def validate_spec(spec: dict) -> None:
    if spec.get("format_version") != 1:
        raise ValueError("Expected preprocessing format_version=1")
    data = spec["data"]
    if data["history_length"] != 168 or data["prediction_length"] != 24:
        raise ValueError("This feature mapping requires 168 history and 24 forecast hours")
    boundaries = data["split"]["frozen_boundaries"]
    if set(boundaries) != {"train", "validation", "test"}:
        raise ValueError("Specify train, validation, and test boundaries")
    previous_end = None
    for split in ("train", "validation", "test"):
        start = pd.Timestamp(boundaries[split]["start"])
        end = pd.Timestamp(boundaries[split]["end"])
        if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None:
            raise ValueError("Boundaries must be finite, timezone-naive dates")
        if start != start.normalize() or end != end.normalize() or start > end:
            raise ValueError(f"Invalid inclusive dates for {split}")
        if previous_end is not None and start <= previous_end:
            raise ValueError("Split ranges must be chronological and non-overlapping")
        previous_end = end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Input root containing data/raw/ and extracted provider CSVs")
    parser.add_argument("--output", type=Path, required=True,
                        help="A new output directory; existing directories are not overwritten")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs/gefcom_preprocessing.yaml")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_spec(spec)
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError(f"Choose a new output directory: {dest}")

    report, canonical = run_audit(args.raw_root)
    dest.mkdir(parents=True, exist_ok=False)
    save_audit(dest, report, canonical)
    bundle = build_windows(canonical, spec)
    if set(bundle.metadata["split"]) != {"train", "validation", "test"}:
        raise ValueError("Each split must contain complete samples after the history window")
    save_windows(dest, bundle)
    build_mapping(dest, dest / "data/processed/gefcom")
    (dest / "preprocessing_config.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    inputs = [
        "data/raw/GEFCom2014.zip",
        "data/raw/source/GEFCom2014-S_V2.zip",
        "data/raw/source/Solar/Task 15/predictors15.csv",
        "data/raw/source/Solar/Task 15/train15.csv",
        "data/raw/source/Solar/Solution to Task 15/Solution to Task 15.csv",
    ]
    manifest = {
        "status": "passed",
        "input_sha256": {name: sha256(args.raw_root / name) for name in inputs},
        "config_sha256": sha256(args.config),
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "output_sha256": {p.relative_to(dest).as_posix(): sha256(p)
                          for p in sorted(dest.rglob("*")) if p.is_file()},
    }
    (dest / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"GEFCom preprocessing complete: {dest}")


if __name__ == "__main__":
    main()
