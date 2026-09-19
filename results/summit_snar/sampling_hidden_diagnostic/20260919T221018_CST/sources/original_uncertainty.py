"""Train-only preprocessing and calibrated, grouped decision-risk prediction.

Labels describe an observed event or an explicitly named future outcome. A terminal
failure label must never be described as proof that every earlier action was wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
import torch
from torch import nn


@dataclass(frozen=True)
class RiskTrainingConfig:
    label_kind: str
    seed: int = 1729
    hidden_width: int = 64
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 15
    include_error_similarity: bool = True


def _fingerprint(items: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(set(items)), separators=(",", ":")).encode()).hexdigest()


def assert_group_disjoint(train_groups, dev_groups, test_groups=()) -> None:
    splits = {"train": set(train_groups), "dev": set(dev_groups), "test": set(test_groups)}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = splits[a] & splits[b]
        if overlap:
            raise ValueError(f"Group leakage between {a} and {b}: {sorted(overlap)[:5]}")


class TrainOnlyFeatures:
    def fit(self, x: np.ndarray, y: np.ndarray, names: Sequence[str], groups: Sequence[str]):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 2 or len(x) != len(y) or x.shape[1] != len(names):
            raise ValueError("Inconsistent feature, label, or name dimensions")
        if len(groups) != len(x) or not len(x):
            raise ValueError("Every training row requires a group")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Risk labels must be observed binary outcomes")
        self.names = list(names)
        finite = np.isfinite(x)
        self.median = np.array([np.median(x[finite[:, i], i]) if finite[:, i].any() else 0.0 for i in range(x.shape[1])])
        filled = np.where(finite, x, self.median)
        self.mean = filled.mean(0)
        self.std = filled.std(0)
        self.std[self.std < 1e-8] = 1.0
        z = (filled - self.mean) / self.std
        self.error_prototype = z[y == 1].mean(0) if (y == 1).any() else np.zeros(x.shape[1])
        self.has_error_prototype = bool((y == 1).any())
        self.training_group_hash = _fingerprint(groups)
        return self

    def transform(self, x, names: Sequence[str], include_error_similarity=True):
        if list(names) != self.names:
            raise ValueError("Feature schema/order differs from the training schema")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.names):
            raise ValueError("Invalid feature matrix")
        missing = ~np.isfinite(x)
        z = (np.where(missing, self.median, x) - self.mean) / self.std
        values = [z, missing.astype(np.float64)]
        if include_error_similarity:
            distance = np.sqrt(np.mean((z - self.error_prototype) ** 2, axis=1))
            cosine = (z @ self.error_prototype) / np.maximum(np.linalg.norm(z, axis=1) * np.linalg.norm(self.error_prototype), 1e-8)
            values.extend([distance[:, None], cosine[:, None], np.full((len(x), 1), not self.has_error_prototype)])
        result = np.concatenate(values, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced nonfinite features")
        return result

    def state(self):
        return {"names": self.names, "median": self.median.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist(), "error_prototype": self.error_prototype.tolist(), "has_error_prototype": self.has_error_prototype, "training_group_hash": self.training_group_hash}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for key, value in state.items():
            setattr(obj, key, np.asarray(value, dtype=np.float64) if key in {"median", "mean", "std", "error_prototype"} else value)
        return obj


class RiskMLP(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def _episode_weights(episode_ids: Sequence[str]) -> np.ndarray:
    unique, inverse, count = np.unique(episode_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / count[inverse]
    return weights * (len(weights) / weights.sum())


class CalibratedRiskModel:
    def fit(self, train_x, train_y, dev_x, dev_y, *, feature_names, train_groups, dev_groups, train_episode_ids, config: RiskTrainingConfig):
        assert_group_disjoint(train_groups, dev_groups)
        train_y, dev_y = np.asarray(train_y), np.asarray(dev_y)
        if len(np.unique(train_y)) != 2 or len(dev_y) < 2 or not np.isin(dev_y, [0, 1]).all():
            raise ValueError("Training needs both observed classes and development needs valid labels")
        if len(train_episode_ids) != len(train_y):
            raise ValueError("Episode identifiers are required to avoid overweighting long trajectories")
        if len(dev_groups) != len(dev_y):
            raise ValueError("Development groups missing")
        torch.manual_seed(config.seed)
        self.config = config
        self.features = TrainOnlyFeatures().fit(train_x, train_y, feature_names, train_groups)
        x = torch.from_numpy(self.features.transform(train_x, feature_names, config.include_error_similarity))
        dx = torch.from_numpy(self.features.transform(dev_x, feature_names, config.include_error_similarity))
        y = torch.as_tensor(train_y, dtype=torch.float32)
        dy = torch.as_tensor(dev_y, dtype=torch.float32)
        weights = torch.as_tensor(_episode_weights(train_episode_ids), dtype=torch.float32)
        self.model = RiskMLP(x.shape[1], config.hidden_width)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        generator = torch.Generator().manual_seed(config.seed)
        best_loss, best_state, stale = float("inf"), None, 0
        self.history = []
        for epoch in range(config.epochs):
            self.model.train()
            order = torch.randperm(len(x), generator=generator)
            for batch in order.split(config.batch_size):
                logits = self.model(x[batch])
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, y[batch], reduction="none") * weights[batch]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                dev_loss = float(nn.functional.binary_cross_entropy_with_logits(self.model(dx), dy))
            self.history.append({"epoch": epoch + 1, "development_bce": dev_loss})
            if not np.isfinite(dev_loss):
                raise RuntimeError("Nonfinite uncertainty training loss")
            if dev_loss < best_loss - 1e-8:
                best_loss, best_state, stale = dev_loss, {k: v.detach().clone() for k, v in self.model.state_dict().items()}, 0
                self.selected_epoch = epoch + 1
            else:
                stale += 1
            if stale >= config.patience:
                break
        self.model.load_state_dict(best_state)
        with torch.no_grad():
            dev_logits = self.model(dx).numpy().astype(np.float64)
        # Scalar temperature calibration uses development labels only.
        fit = minimize_scalar(lambda log_t: float(np.mean(np.logaddexp(0, dev_logits / np.exp(log_t)) - dev_y * dev_logits / np.exp(log_t))), bounds=(-4, 4), method="bounded")
        self.temperature = float(np.exp(fit.x))
        self.provenance = {"training_group_hash": _fingerprint(train_groups), "development_group_hash": _fingerprint(dev_groups), "train_rows": len(train_y), "development_rows": len(dev_y), "label_kind": config.label_kind, "test_used_for_fit": False}
        return self

    def predict_proba(self, x, feature_names):
        self.model.eval()
        values = self.features.transform(x, feature_names, self.config.include_error_similarity)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(values)).numpy()
        return expit(logits / self.temperature)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": "matdiscovery_risk_v1", "config": asdict(self.config), "features": self.features.state(), "model": self.model.state_dict(), "temperature": self.temperature, "provenance": self.provenance, "history": self.history, "selected_epoch": self.selected_epoch}, path)

    @classmethod
    def load(cls, path: str | Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["schema"] != "matdiscovery_risk_v1":
            raise ValueError("Unknown risk checkpoint schema")
        obj = cls()
        obj.config = RiskTrainingConfig(**state["config"])
        obj.features = TrainOnlyFeatures.from_state(state["features"])
        obj.model = RiskMLP(state["model"]["layers.0.weight"].shape[1], obj.config.hidden_width)
        obj.model.load_state_dict(state["model"])
        for key in ["temperature", "provenance", "history", "selected_epoch"]:
            setattr(obj, key, state[key])
        return obj


def risk_metrics(y, probability, *, bins=10):
    y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if y.shape != p.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Metrics require complete valid binary labels and probabilities")
    ece = 0.0
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        mask = assignments == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    order = np.argsort(p, kind="stable")
    risks = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    return {"n": len(y), "error_rate": float(y.mean()), "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "error_auprc": float(average_precision_score(y, p)) if y.sum() else None, "brier": float(brier_score_loss(y, p)), "nll": float(log_loss(y, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1])), "ece": float(ece), "risk_coverage_auc": float(risks.mean()), "overconfident_error_rate_p_le_0_1": float(np.mean((y == 1) & (p <= .1)))}
