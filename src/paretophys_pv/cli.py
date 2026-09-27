"""Command-line entry points for preparing data, training, search, and prediction."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from .data import demo_hourly, load_bundle, prepare_hourly, save_bundle
from .metrics import evaluate_predictions
from .search import run_search
from .search_space import Genotype, repair_genotype
from .selection import assign_roles, screen_candidates
from .training import predict_checkpoint, train_candidate


def main():
    parser = argparse.ArgumentParser(prog="paretophys", description="Day-ahead PV forecasting and multi-objective model selection")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo-data", help="Generate a small local toy sequence bundle")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=2026)
    prep = commands.add_parser("prepare", help="Convert a canonical hourly CSV to a sequence bundle")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("train", "search", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        sub.add_argument("--threads", type=int, default=1)
        sub.add_argument("--batch-size", type=int, default=256)
        if name != "predict":
            sub.add_argument("--epochs", type=int, default=12 if name == "search" else 40)
            sub.add_argument("--patience", type=int, default=3 if name == "search" else 6)
            sub.add_argument("--seed", type=int, default=2026)
        if name == "train":
            sub.add_argument("--config", type=Path, required=True)
        if name == "search":
            sub.add_argument("--population", type=int, default=16)
            sub.add_argument("--generations", type=int, default=8)
            sub.add_argument("--search-seed", type=int, default=3101)
            sub.add_argument("--method", choices=["nsga2", "random"], default="nsga2")
        if name == "predict":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--split", choices=["train", "validation", "test"], default="test")
    select = commands.add_parser("select", help="Screen Pareto coverage or assign final roles")
    select.add_argument("--input", type=Path, required=True, help="JSON array of configuration evaluation records")
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--mode", choices=["screen", "roles"], default="screen")
    select.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.command in {"demo-data", "prepare"}:
        hourly = demo_hourly(args.seed) if args.command == "demo-data" else pd.read_csv(args.input)
        data = prepare_hourly(hourly)
        if args.output.exists():
            raise FileExistsError(args.output)
        save_bundle(args.output, data)
        print(json.dumps({"samples": len(data["target"]), "output": str(args.output)}, indent=2))
        return
    if args.command == "select":
        records = json.loads(args.input.read_text())
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Selection output directory must be empty")
        args.output.mkdir(parents=True, exist_ok=True)
        selected = screen_candidates(records, args.count) if args.mode == "screen" else assign_roles(records)
        (args.output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
        configs = {f"candidate_{i + 1:02d}": row for i, row in enumerate(selected)} if isinstance(selected, list) else selected
        for name, row in configs.items():
            (args.output / f"{name}.yaml").write_text(yaml.safe_dump(row["genotype"], sort_keys=False))
        print(f"Saved {len(configs)} configurations to {args.output}")
        return
    if min(args.threads, args.batch_size) < 1:
        parser.error("Threads and batch size must be positive")
    torch.set_num_threads(args.threads)
    data = load_bundle(args.data)
    if args.command == "train":
        genotype = Genotype(**yaml.safe_load(args.config.read_text()))
        repair = repair_genotype(genotype)
        if repair.constraint_violation > 1e-12:
            raise ValueError("Model configuration is infeasible after repair")
        result = train_candidate(data, repair.genotype, args.output, epochs=args.epochs, patience=args.patience,
                                 batch_size=args.batch_size, seed=args.seed, device=args.device)
    elif args.command == "search":
        result = run_search(data, args.output, population=args.population, generations=args.generations,
                            method=args.method, search_seed=args.search_seed, seed=args.seed, epochs=args.epochs,
                            patience=args.patience, batch_size=args.batch_size, device=args.device)
    else:
        if args.output.exists() or args.output.with_suffix(".metrics.json").exists():
            raise FileExistsError(args.output)
        frame = predict_checkpoint(data, args.checkpoint, split=args.split, device=args.device, batch_size=args.batch_size)
        result, *_ = evaluate_predictions(frame, "prediction_pu", expected_split=args.split)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        args.output.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
