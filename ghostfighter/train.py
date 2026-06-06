from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .config import ACTION_NAMES, STYLE_NAMES, TrainConfig
from .dataset import FightTraceDataset
from .models import PolicyNet


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small MLPs train faster and more predictably with a single CPU thread in
    # constrained review environments. Users can remove this for large-GPU runs.
    torch.set_num_threads(1)


def train_behavior_cloning(
    dataset_path: str | Path,
    out_dir: str | Path,
    config: TrainConfig | None = None,
    hidden: int = 192,
    verbose: bool = False,
) -> Dict[str, object]:
    """Train the conditional ghost policy from Generation Zero traces."""
    config = config or TrainConfig()
    set_seeds(config.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = FightTraceDataset(dataset_path)
    n = len(dataset)
    if n < 32:
        raise ValueError("Dataset is too small. Generate more traces.")
    idx = np.arange(n)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(idx)
    val_n = max(1, int(n * config.val_split))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=config.batch_size, shuffle=False)

    obs_dim = int(dataset.obs.shape[1])
    model = PolicyNet(obs_dim=obs_dim, hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0
        for obs, styles, actions in train_loader:
            logits = model(obs, styles)
            loss = criterion(logits, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss += float(loss.item()) * obs.shape[0]
            train_correct += int((logits.argmax(dim=-1) == actions).sum().item())
            train_seen += int(obs.shape[0])
        scheduler.step()
        val_loss, val_acc, per_style = evaluate_loader(model, val_loader, criterion)
        train_acc = train_correct / max(1, train_seen)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_seen),
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        for style_name, acc in per_style.items():
            row[f"val_acc_{style_name}"] = acc
        history.append(row)
        if verbose:
            print(f"epoch {epoch}/{config.epochs} train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_loss={val_loss:.3f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = history[-1].copy()
    metrics["best_val_acc"] = float(best_val_acc)
    metrics["dataset_samples"] = int(n)
    metrics["obs_dim"] = obs_dim
    model_path = out_dir / "ghost_policy.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": obs_dim,
            "hidden": hidden,
            "action_names": ACTION_NAMES,
            "style_names": STYLE_NAMES,
            "metrics": metrics,
            "train_config": config.__dict__,
        },
        model_path,
    )
    pd.DataFrame(history).to_csv(out_dir / "training_curve.csv", index=False)
    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"model_path": str(model_path), "metrics": metrics, "history": history}


@torch.no_grad()
def evaluate_loader(model: PolicyNet, loader: DataLoader, criterion) -> tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    style_correct = {i: 0 for i in range(len(STYLE_NAMES))}
    style_seen = {i: 0 for i in range(len(STYLE_NAMES))}
    for obs, styles, actions in loader:
        logits = model(obs, styles)
        loss = criterion(logits, actions)
        preds = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * obs.shape[0]
        correct += int((preds == actions).sum().item())
        seen += int(obs.shape[0])
        for style_id in range(len(STYLE_NAMES)):
            mask = styles == style_id
            style_seen[style_id] += int(mask.sum().item())
            if mask.any():
                style_correct[style_id] += int((preds[mask] == actions[mask]).sum().item())
    per_style = {
        STYLE_NAMES[i]: (style_correct[i] / style_seen[i] if style_seen[i] else math.nan)
        for i in range(len(STYLE_NAMES))
    }
    return total_loss / max(1, seen), correct / max(1, seen), per_style
