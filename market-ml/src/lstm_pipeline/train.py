from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8
    min_delta: float = 1e-5  # reduced to avoid premature stopping on tiny val_loss improvements
    grad_clip: float = 1.0
    loss: str = "mse"  # "mse" or "huber"
    device: str = "cpu"
    verbose: bool = True


def _loss_fn(name: str) -> nn.Module:
    if name.lower() == "huber":
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: TrainConfig,
) -> Tuple[nn.Module, Dict[str, list]]:
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = _loss_fn(cfg.loss)

    best_val = float("inf")
    best_state = None
    patience = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if cfg.verbose:
            print(f"[lstm][epoch {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # Use a smaller min_delta to accept subtle val_loss improvements.
        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
