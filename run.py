"""Train, evaluate, or forecast with LoadMixer on a CSV dataset."""
import argparse
import inspect
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Windows, fingerprint, read_csv
from loadmixer import LoadMixer


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "test", "predict"], default="train")
    parser.add_argument("--data_path", required=True, help="CSV file path")
    parser.add_argument("--target", help="Target column; required for training")
    parser.add_argument("--date_column", default=None, help="Timestamp column (training default: date)")
    parser.add_argument("--output_dir", default="outputs/loadmixer")
    parser.add_argument("--checkpoint", help="Saved checkpoint for test or predict mode")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=positive_int, default=32)
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--loss", choices=["MSE", "MAE"], default="MSE")
    parser.add_argument("--pred_len", type=positive_int, default=12)
    parser.add_argument("--d_model", type=positive_int, default=16)
    parser.add_argument("--d_ff", type=positive_int, default=32)
    parser.add_argument("--e_layers", type=positive_int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=1, help="CPU threads")
    args = parser.parse_args()
    if args.mode == "train" and not args.target:
        parser.error("--target is required for training.")
    if args.mode != "train" and not args.checkpoint:
        parser.error("--checkpoint is required for test or predict mode.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning_rate must be finite and positive.")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1).")
    if args.d_model % 2:
        parser.error("--d_model must be even.")
    return args


def load_checkpoint(path):
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = True
    return torch.load(path, **options)


def validation_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, marks, y in loader:
            loss = criterion(model(x.to(device), marks.to(device)), y.to(device))
            total += loss.item() * len(x)
            count += len(x)
    result = total / count
    if not math.isfinite(result):
        raise ValueError("Non-finite validation loss; check the data and learning rate.")
    return result


def evaluate(model, dataset, batch_size, device, scale, output_dir):
    model.eval()
    absolute, squared, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, marks, y in DataLoader(dataset, batch_size=batch_size):
            prediction = model(x.to(device), marks.to(device)).cpu().double()
            error = prediction - y.double()
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
    if not math.isfinite(absolute + squared):
        raise ValueError("Non-finite test predictions.")
    mae, mse = absolute / count, squared / count
    metrics = {
        "mae": mae * scale, "mse": mse * scale ** 2, "rmse": math.sqrt(mse) * scale,
        "standardized_mae": mae, "standardized_mse": mse,
        "standardized_rmse": math.sqrt(mse), "windows": len(dataset),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def train(args, device, output_dir):
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        raise ValueError("This output directory already has a checkpoint. Choose a new --output_dir.")
    date_column = args.date_column or "date"
    dates, values, marks = read_csv(args.data_path, args.target, date_column)
    size = len(values)
    train_end, test_start = int(size * 0.7), size - int(size * 0.2)
    if train_end < 96 + args.pred_len:
        raise ValueError("The training split is too short. Supply a longer CSV.")
    mean, scale = float(values[:train_end].mean()), float(values[:train_end].std())
    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValueError("Target magnitudes are too large for normalization.")
    scale = scale if scale > 1e-12 else 1.0
    datasets = [Windows(values, marks, start, end, 96, args.pred_len, mean, scale)
                for start, end in [(0, train_end), (train_end, test_start), (test_start, size)]]
    config = {key: getattr(args, key) for key in ["pred_len", "d_model", "d_ff", "e_layers", "dropout"]}
    model = LoadMixer(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss() if args.loss == "MSE" else torch.nn.L1Loss()
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=args.batch_size)
    metadata = dict(model_config=config, target=args.target, date_column=date_column,
                    mean=mean, scale=scale, test_start=test_start, rows=size,
                    data_sha256=fingerprint(args.data_path),
                    interval_ns=int((dates.iloc[1] - dates.iloc[0]).value))
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, time_marks, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device), time_marks.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; check the data and learning rate.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        score = validation_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train {total / count:.6f}, validation {score:.6f}", flush=True)
        if score < best:
            best, stale = score, 0
            torch.save(dict(metadata, model_state=model.state_dict()), checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(load_checkpoint(checkpoint_path)["model_state"])
    evaluate(model, datasets[2], args.batch_size, device, scale, output_dir)
    print(f"Checkpoint: {checkpoint_path}")


def run_saved(args, device, output_dir):
    saved = load_checkpoint(args.checkpoint)
    target = args.target or saved["target"]
    date_column = args.date_column or saved["date_column"]
    if target != saved["target"] or date_column != saved["date_column"]:
        raise ValueError("Use the target and timestamp columns stored in the checkpoint.")
    dates, values, marks = read_csv(args.data_path, target, date_column)
    model = LoadMixer(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state"])
    if args.mode == "test":
        if fingerprint(args.data_path) != saved["data_sha256"]:
            raise ValueError("Test mode requires the training CSV to reproduce its held-out split.")
        dataset = Windows(values, marks, saved["test_start"], len(values), 96,
                          model.pred_len, saved["mean"], saved["scale"])
        evaluate(model, dataset, args.batch_size, device, saved["scale"], output_dir)
    else:
        if len(values) < 96:
            raise ValueError("Prediction requires at least 96 rows of history.")
        if int((dates.iloc[1] - dates.iloc[0]).value) != saved["interval_ns"]:
            raise ValueError("The input frequency must match the training data.")
        x = ((values[-96:] - saved["mean"]) / saved["scale"]).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Input values exceed the supported floating-point range.")
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(x[None, :, None]).to(device),
                               torch.from_numpy(marks[-96:][None]).to(device))
        prediction = prediction.cpu().numpy().reshape(-1).astype(np.float64) * saved["scale"] + saved["mean"]
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite predictions.")
        future = [dates.iloc[-1] + pd.Timedelta(saved["interval_ns"] * step, unit="ns")
                  for step in range(1, model.pred_len + 1)]
        path = output_dir / "forecast.csv"
        pd.DataFrame({date_column: future, target: prediction}).to_csv(path, index=False)
        print(f"Forecast: {path}")


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train(args, device, output_dir)
    else:
        run_saved(args, device, output_dir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
