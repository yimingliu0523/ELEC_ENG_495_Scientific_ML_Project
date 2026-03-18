"""Reusable training loops for discrete-time models and Neural ODEs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.losses import mse_loss
from src.utils.io import ensure_dir, save_json


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate_supervised_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * inputs.shape[0]
            total_examples += inputs.shape[0]
    return total_loss / max(total_examples, 1)


def evaluate_sequence_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for segments, t_grid in data_loader:
            segments = segments.to(device)
            t_grid = t_grid.to(device)
            predictions = model(segments[:, 0], t_grid[0])
            loss = criterion(predictions, segments)
            total_loss += loss.item() * segments.shape[0]
            total_examples += segments.shape[0]
    return total_loss / max(total_examples, 1)


def _save_history(history: list[dict], output_dir: str | Path, stem: str) -> Path:
    output_dir = ensure_dir(output_dir)
    history_path = output_dir / f"{stem}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    return history_path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    path: str | Path,
    extra: dict | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def train_supervised_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    run_name: str,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
    extra_checkpoint_data: dict | None = None,
) -> dict:
    output_dir = ensure_dir(output_dir)
    best_val_loss = float("inf")
    best_checkpoint = output_dir / f"{run_name}_best.pt"
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.shape[0]
            seen += inputs.shape[0]

        train_loss = running_loss / max(seen, 1)
        val_loss = evaluate_supervised_model(model, val_loader, device=device, criterion=criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                best_checkpoint,
                extra=extra_checkpoint_data,
            )

    train_time = time.time() - start_time
    history_path = _save_history(history, output_dir, run_name)
    summary = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "train_time_sec": train_time,
        "params_count": count_parameters(model),
        "history_path": str(history_path),
        "checkpoint_path": str(best_checkpoint),
    }
    save_json(summary, output_dir / f"{run_name}_train_summary.json")
    return summary


def train_sequence_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    run_name: str,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
    extra_checkpoint_data: dict | None = None,
) -> dict:
    output_dir = ensure_dir(output_dir)
    best_val_loss = float("inf")
    best_checkpoint = output_dir / f"{run_name}_best.pt"
    history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for segments, t_grid in train_loader:
            segments = segments.to(device)
            t_grid = t_grid.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(segments[:, 0], t_grid[0])
            loss = criterion(predictions, segments)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * segments.shape[0]
            seen += segments.shape[0]

        train_loss = running_loss / max(seen, 1)
        val_loss = evaluate_sequence_model(model, val_loader, device=device, criterion=criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                best_checkpoint,
                extra=extra_checkpoint_data,
            )

    train_time = time.time() - start_time
    history_path = _save_history(history, output_dir, run_name)
    summary = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "train_time_sec": train_time,
        "params_count": count_parameters(model),
        "history_path": str(history_path),
        "checkpoint_path": str(best_checkpoint),
    }
    save_json(summary, output_dir / f"{run_name}_train_summary.json")
    return summary
