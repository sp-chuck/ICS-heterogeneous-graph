from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required for utils/evaluation.py. "
        "Install it in your environment before running evaluation."
    ) from exc


def collect_scores(
    model: nn.Module,
    sequences: List[List[int]],
    snapshots: List[HeteroData],
    labels: List[int],
    device: torch.device,
    task_mode: str,
) -> Tuple[List[int], List[float]]:
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for seq in sequences:
            seq_data = [snapshots[i].to(device) for i in seq]
            logit, _ = model(seq_data)
            y_true.append(int(labels[seq[-1]]))
            if task_mode == "anomaly_cls":
                y_score.append(float(torch.sigmoid(logit).detach().cpu().item()))
            else:
                y_score.append(float(logit.detach().cpu().item()))
    return y_true, y_score


def evaluate_regression(
    y_true: List[int],
    y_score: List[float],
) -> Dict[str, float]:
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_score = np.asarray(y_score, dtype=np.float64)
    err = arr_score - arr_true
    mae = float(np.mean(np.abs(err))) if err.size > 0 else float("nan")
    mse = float(np.mean(err ** 2)) if err.size > 0 else float("nan")
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def evaluate_with_threshold(
    y_true: List[int],
    y_score: List[float],
    threshold: float,
) -> Dict[str, float]:
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    else:
        auc = float("nan")
        auprc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    return {
        "threshold": float(threshold),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": float(auc),
        "auprc": float(auprc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def resolve_threshold(
    decision_mode: str,
    fixed_threshold: float,
    train_score_quantile: float,
    train_labels: List[int],
    train_scores: List[float],
) -> float:
    if decision_mode == "fixed":
        return float(fixed_threshold)

    if decision_mode == "train_quantile":
        q = float(train_score_quantile)
        if not (0.0 <= q <= 1.0):
            raise ValueError("train_score_quantile must be in [0,1]")
        return float(np.quantile(np.asarray(train_scores, dtype=np.float64), q))

    uniq = set(int(v) for v in train_labels)
    if len(uniq) <= 1:
        q = float(train_score_quantile)
        if not (0.0 <= q <= 1.0):
            raise ValueError("train_score_quantile must be in [0,1]")
        return float(np.quantile(np.asarray(train_scores, dtype=np.float64), q))
    return float(fixed_threshold)


def summarize_epoch_metrics(
    task_mode: str,
    losses: List[float],
    snapshot_losses: List[float],
    device_losses: List[float],
    y_true: List[float],
    y_score: List[float],
    y_true_device: List[int],
    y_score_device: List[float],
) -> Dict[str, float]:
    if task_mode == "anomaly_cls":
        y_true_cls = [int(v) for v in y_true]
        y_pred = [1 if s >= 0.5 else 0 for s in y_score]

        if len(set(y_true_cls)) > 1:
            auc = roc_auc_score(y_true_cls, y_score)
            auprc = average_precision_score(y_true_cls, y_score)
        else:
            auc = float("nan")
            auprc = float("nan")

        if y_true_cls:
            tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred, labels=[0, 1]).ravel().tolist()
        else:
            tn = fp = fn = tp = 0

        acc = accuracy_score(y_true_cls, y_pred) if y_true_cls else float("nan")
        precision = precision_score(y_true_cls, y_pred, zero_division=0) if y_true_cls else float("nan")
        recall = recall_score(y_true_cls, y_pred, zero_division=0) if y_true_cls else float("nan")
        f1 = f1_score(y_true_cls, y_pred, zero_division=0) if y_true_cls else float("nan")
        mae = float("nan")
        mse = float("nan")
        rmse = float("nan")
    else:
        y_true_reg = np.asarray(y_true, dtype=np.float64)
        y_score_reg = np.asarray(y_score, dtype=np.float64)
        err = y_score_reg - y_true_reg
        mae = float(np.mean(np.abs(err))) if err.size > 0 else float("nan")
        mse = float(np.mean(err ** 2)) if err.size > 0 else float("nan")
        rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")

        acc = precision = recall = f1 = auc = auprc = float("nan")
        tn = fp = fn = tp = 0

    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "snapshot_loss": float(np.mean(snapshot_losses)) if snapshot_losses else float("nan"),
        "device_loss": float(np.mean(device_losses)) if device_losses else float("nan"),
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "auprc": auprc,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if y_true_device:
        y_pred_device = [1 if s >= 0.5 else 0 for s in y_score_device]
        if len(set(y_true_device)) > 1:
            auc_device = roc_auc_score(y_true_device, y_score_device)
        else:
            auc_device = float("nan")

        metrics.update(
            {
                "device_acc": accuracy_score(y_true_device, y_pred_device),
                "device_precision": precision_score(y_true_device, y_pred_device, zero_division=0),
                "device_recall": recall_score(y_true_device, y_pred_device, zero_division=0),
                "device_f1": f1_score(y_true_device, y_pred_device, zero_division=0),
                "device_auc": auc_device,
            }
        )
    else:
        metrics.update(
            {
                "device_acc": float("nan"),
                "device_precision": float("nan"),
                "device_recall": float("nan"),
                "device_f1": float("nan"),
                "device_auc": float("nan"),
            }
        )

    return metrics


__all__ = [
    "collect_scores",
    "evaluate_regression",
    "evaluate_with_threshold",
    "resolve_threshold",
    "summarize_epoch_metrics",
]
