"""Observed failure-type heads that preserve the existing future-risk predictor.

This CPU-only component never infers labels from an LLM judgment. Unknown labels
are NaN, not negatives. The caller supplies the same raw feature columns as the
primary predictor, but may supply additional collection rows with known type
labels and an unknown overall future outcome. Generation-format validity is an
observable interface feature, not evidence of a white-box interpretability gain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from torch import nn

from .accounting import file_sha256, fingerprint
from .uncertainty import CalibratedRiskModel, assert_group_disjoint


HEADS = (
    "generation_invalid", "candidate_generation_failure", "tool_execution_failure", "screening_unavailable",
    "scientific_evaluation_failure", "unstable", "not_new",
)
# Public compatibility name; there is one authoritative ordered tuple.
FAILURE_TYPES = HEADS
SCHEMA = "matdiscovery_failure_risk_v1"


class FailureRiskError(ValueError):
    """A typed-risk fit or checkpoint lacks its required evidence."""


@dataclass(frozen=True)
class FailureRiskTrainingConfig:
    seed: int = 1729
    hidden_width: int = 64
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = .001
    weight_decay: float = .0001
    patience: int = 15

    def __post_init__(self):
        if (type(self.seed) is not int or not 0 <= self.seed < 2**63
                or self.hidden_width != 64 or type(self.hidden_width) is not int
                or any(type(v) is not int or v < 1 for v in (self.epochs, self.batch_size, self.patience))
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise FailureRiskError("Require two 64-wide hidden layers and valid positive training settings")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FailureRiskError(f"{name} must be an explicit SHA256")
    return value


def _names(names):
    values = list(names)
    if not values or any(not isinstance(n, str) or not n for n in values) or len(set(values)) != len(values):
        raise FailureRiskError("Feature names must be nonempty and unique")
    return values


def _x(values, names, *, nonempty=True):
    x = np.asarray(values, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != len(names) or (nonempty and not len(x))
            or np.isinf(x).any()):
        raise FailureRiskError("Raw features must match the primary schema; only NaN denotes missing data")
    return x


def _labels(values, rows):
    y = np.asarray(values, dtype=np.float64)
    if y.shape != (rows, len(FAILURE_TYPES)) or np.isinf(y).any() or not np.isin(y[np.isfinite(y)], [0, 1]).all():
        raise FailureRiskError(f"Failure labels must be N by {len(HEADS)}, with observed 0/1 and NaN for unknown")
    return y


def _ids(values, rows, name):
    result = list(values)
    if len(result) != rows or any(not isinstance(v, str) or not v for v in result):
        raise FailureRiskError(f"Every row needs a nonempty string {name}")
    return result


def _weights(values, rows):
    out = np.ones(rows) if values is None else np.asarray(values, dtype=np.float64)
    if out.shape != (rows,) or not np.isfinite(out).all() or (out <= 0).any():
        raise FailureRiskError("Sample weights must be finite and strictly positive")
    return out


def _episode_weights(groups, episodes, raw_weights, observed=None):
    """Each observed group/episode contributes one unit, then normalize to one.

    With a mask, normalize within the observed rows of each episode. Thus sparse
    labels or long trajectories cannot increase an episode's weight in a head.
    Optional sample weights specify relative row weights *within* an episode.
    """
    mask = np.ones(len(groups), dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    totals = {}
    for g, e, w, use in zip(groups, episodes, raw_weights, mask):
        if use:
            totals[g, e] = totals.get((g, e), 0.) + float(w)
    result = np.zeros(len(groups))
    for i, (g, e, w, use) in enumerate(zip(groups, episodes, raw_weights, mask)):
        if use:
            result[i] = w / totals[g, e] / len(totals)
    return result


class FailureTypeFeatures:
    """Train-only episode-weighted imputation, scaling and per-type error prototypes."""

    def fit(self, x, y, names, groups, episodes, raw_weights):
        self.names = list(names)
        weights = _episode_weights(groups, episodes, raw_weights)
        finite = np.isfinite(x)
        self.median = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            use = finite[:, j]
            if use.any():
                order = np.argsort(x[use, j], kind="stable")
                vals, ws = x[use, j][order], weights[use][order]
                self.median[j] = vals[np.searchsorted(np.cumsum(ws), ws.sum() / 2, side="left")]
        filled = np.where(finite, x, self.median)
        self.mean = np.sum(filled * weights[:, None], axis=0)
        self.std = np.sqrt(np.sum((filled - self.mean)**2 * weights[:, None], axis=0))
        self.std[self.std < 1e-8] = 1.
        z = (filled - self.mean) / self.std
        self.error_prototypes = np.zeros((len(FAILURE_TYPES), x.shape[1]))
        self.has_error_prototype = np.zeros(len(FAILURE_TYPES), dtype=bool)
        for h in range(len(FAILURE_TYPES)):
            positive = y[:, h] == 1
            if positive.any():
                pw = _episode_weights(groups, episodes, raw_weights, positive)
                self.error_prototypes[h] = np.sum(z * pw[:, None], axis=0)
                self.has_error_prototype[h] = True
        self.training_group_hash = _hash(sorted(set(groups)))
        self.training_episode_hash = _hash(sorted(set(zip(groups, episodes))))
        self.transform(x, names)
        return self

    def transform(self, values, names):
        if list(names) != self.names:
            raise FailureRiskError("Feature schema/order differs from the primary schema")
        x = _x(values, names, nonempty=False)
        missing = ~np.isfinite(x)
        z = (np.where(missing, self.median, x) - self.mean) / self.std
        parts = [z, missing.astype(float)]
        for p, available in zip(self.error_prototypes, self.has_error_prototype):
            distance = np.sqrt(np.mean((z - p)**2, axis=1)) if available else np.zeros(len(x))
            cosine = z @ p / np.maximum(np.linalg.norm(z, axis=1) * np.linalg.norm(p), 1e-8) if available else np.zeros(len(x))
            parts.extend((distance[:, None], cosine[:, None], np.full((len(x), 1), not available)))
        result = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise FailureRiskError("Type preprocessing produced nonfinite values")
        return result

    def state(self):
        return {"names": self.names, "median": self.median.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "error_prototypes": self.error_prototypes.tolist(),
                "has_error_prototype": self.has_error_prototype.tolist(),
                "training_group_hash": self.training_group_hash, "training_episode_hash": self.training_episode_hash}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.names = _names(state["names"])
        for key in ("median", "mean", "std", "error_prototypes"):
            values = np.asarray(state[key], dtype=np.float64)
            shape = (len(FAILURE_TYPES), len(obj.names)) if key == "error_prototypes" else (len(obj.names),)
            if values.shape != shape or not np.isfinite(values).all() or (key == "std" and (values <= 0).any()):
                raise FailureRiskError("Invalid saved type preprocessing")
            setattr(obj, key, values)
        flags = state["has_error_prototype"]
        if len(flags) != len(FAILURE_TYPES) or any(type(v) is not bool for v in flags):
            raise FailureRiskError("Invalid prototype availability")
        obj.has_error_prototype = np.asarray(flags, dtype=bool)
        obj.training_group_hash = _sha(state["training_group_hash"], "training_group_hash")
        obj.training_episode_hash = _sha(state["training_episode_hash"], "training_episode_hash")
        return obj


class FailureTypeMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, len(HEADS)))

    def forward(self, values):
        return self.layers(values)


def masked_binary_cross_entropy(logits, labels, weights, *, normalizers=None):
    """Average head BCE with exactly zero loss/gradient for unknown labels.

    Pre-normalized episode weights may be zero for unsupported heads. Training
    minibatches use full-training normalizers, preserving the weighting instead
    of implicitly making every minibatch or observed-row count equally weighted.
    """
    observed = torch.isfinite(labels)
    effective = torch.where(observed, weights, torch.zeros_like(weights))
    normalizers = effective.sum(0) if normalizers is None else normalizers
    active = normalizers > 0
    targets = torch.where(observed, labels, torch.zeros_like(labels))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if not active.any():
        return logits.sum() * 0
    return ((loss * effective).sum(0)[active] / normalizers[active]).mean()


def _support(y, groups, episodes):
    result = {}
    for h, name in enumerate(FAILURE_TYPES):
        observed = np.isfinite(y[:, h])
        result[name] = {"rows": len(y), "observed": int(observed.sum()), "unknown": int((~observed).sum()),
                        "coverage": float(observed.mean()), "positive": int((y[:, h] == 1).sum()),
                        "negative": int((y[:, h] == 0).sum())}
        for prefix, mask in (("observed", observed), ("positive", y[:, h] == 1), ("negative", y[:, h] == 0)):
            result[name][prefix + "_groups"] = len({g for g, use in zip(groups, mask) if use})
            result[name][prefix + "_episodes"] = len({(g, e) for g, e, use in zip(groups, episodes, mask) if use})
    return result


def _head_support(train, dev):
    result = {}
    for name in FAILURE_TYPES:
        reasons = [f"{split}_{kind}_support_missing" for split, data in (("train", train), ("dev", dev))
                   for kind in ("positive", "negative") if data[name][kind] == 0]
        result[name] = {"available": not reasons, "reasons": reasons, "train": train[name], "dev": dev[name]}
    return result


def _source(provenance):
    if not isinstance(provenance, Mapping):
        raise FailureRiskError("Explicit source provenance is required")
    source = provenance.get("collection_provenance", provenance)
    if not isinstance(source, Mapping) or not isinstance(source.get("policy_runtime"), Mapping) or not source["policy_runtime"]:
        raise FailureRiskError("Source provenance needs the full policy runtime")
    if source.get("test_used_for_fit") is not False or source.get("test_used_for_threshold_selection") is not False:
        raise FailureRiskError("Source provenance must explicitly exclude test fitting/selection")
    configuration = source.get("policy_configuration")
    if (not isinstance(configuration, Mapping) or configuration.get("policy_runtime") != source["policy_runtime"]
            or _hash(configuration) != source.get("policy_configuration_fingerprint")):
        raise FailureRiskError("Source policy runtime/configuration fingerprint is inconsistent")
    if "dataset_fingerprint" in source and fingerprint({k: v for k, v in source.items() if k != "dataset_fingerprint"}) != source["dataset_fingerprint"]:
        raise FailureRiskError("Source dataset fingerprint is inconsistent")
    return _copy(source)


def _payload_hash(payload):
    digest = hashlib.sha256(_json({k: v for k, v in payload.items() if k not in {"model", "content_sha256"}}).encode())
    for key, value in sorted(payload["model"].items()):
        value = value.detach().cpu().contiguous()
        digest.update(_json([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _primary_state(primary):
    return {"config": asdict(primary.config), "features": primary.features.state(),
            "temperature": float(primary.temperature), "provenance": primary.provenance,
            "history": primary.history, "selected_epoch": primary.selected_epoch,
            "model": primary.model.state_dict()}


def _validate_primary(primary, checkpoint, expected_sha256, expected_runtime, source):
    _sha(expected_sha256, "expected_primary_sha256")
    if file_sha256(checkpoint) != expected_sha256:
        raise FailureRiskError("Primary checkpoint SHA256 mismatch")
    if not isinstance(primary, CalibratedRiskModel) or primary.provenance.get("test_used_for_fit") is not False:
        raise FailureRiskError("Require the verified CalibratedRiskModel primary with train/dev provenance")
    actual_source = _source(primary.provenance)
    if actual_source != source or _copy(expected_runtime) != source["policy_runtime"]:
        raise FailureRiskError("Primary/source provenance or expected policy runtime differs")
    # The primary loader initializes a temporary MLP before copying weights.
    # This verification must not advance the caller's experiment RNG stream.
    with torch.random.fork_rng(devices=[]):
        on_disk = CalibratedRiskModel.load(checkpoint)
    state_hash = _payload_hash(_primary_state(primary))
    if state_hash != _payload_hash(_primary_state(on_disk)) or file_sha256(checkpoint) != expected_sha256:
        raise FailureRiskError("Primary in-memory state differs from the verified checkpoint")
    return {"checkpoint_sha256": expected_sha256, "feature_names": _names(primary.features.names),
            "primary_state_sha256": state_hash, "primary_provenance_sha256": _hash(primary.provenance),
            "source_provenance_sha256": _hash(source), "policy_runtime": _copy(expected_runtime)}


def _array_hash(x, y, groups, episodes, weights):
    digest = hashlib.sha256(_json({"groups": groups, "episodes": episodes}).encode())
    for value in (x, y, weights):
        array = np.asarray(value, dtype="<f8").copy()
        array[np.isnan(array)] = np.nan
        digest.update(_json(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FailureAwareRiskModel:
    """Wrap a verified primary without changing its features/probabilities.

    ``features``, ``model``, and ``provenance`` remain those of the primary.
    Type-specific preprocessing and provenance are ``typed_features`` and
    ``failure_provenance``. ``head_availability`` maps all HEADS names to bool;
    ``head_support`` records observed classes/groups/episodes and reasons.
    """

    @property
    def features(self):
        return self.primary.features

    @property
    def model(self):
        return self.primary.model

    @property
    def provenance(self):
        return self.primary.provenance

    @property
    def head_availability(self):
        return {name: bool(self.head_support[name]["available"]) for name in FAILURE_TYPES}

    @property
    def control_ready(self):
        return any(self.head_availability[name] for name in FAILURE_TYPES if name != "generation_invalid")

    def require_control_ready(self):
        if not self.control_ready:
            raise FailureRiskError("Typed control requires at least one available non-generation failure head")
        return self

    def predict_proba(self, x, feature_names):
        return self.primary.predict_proba(x, feature_names)

    def predict_failure_types(self, x, names):
        values = self.typed_features.transform(x, names)
        self.typed_model.eval()
        with torch.no_grad():
            logits = self.typed_model(torch.from_numpy(values)).numpy().astype(np.float64)
        return {name: expit(logits[:, h] / self.temperatures[name]) if self.head_availability[name] else None
                for h, name in enumerate(FAILURE_TYPES)}

    @classmethod
    def fit(cls, primary, train_x, train_y_types, dev_x, dev_y_types, *, feature_names,
            train_groups, dev_groups, train_episode_ids, dev_episode_ids,
            primary_checkpoint, expected_primary_sha256, expected_policy_runtime, provenance,
            config: FailureRiskTrainingConfig | None = None, train_sample_weights=None,
            dev_sample_weights=None, train_splits=None, dev_splits=None, test_groups=()):
        """Fit on observed type labels, select epochs/calibrate only on dev.

        Type row counts may exceed primary future-labelled counts. Explicit split
        vectors, when supplied, must be exclusively train/dev respectively;
        provenance and group separation are mandatory even without split vectors.
        """
        obj = cls()
        obj.config = config or FailureRiskTrainingConfig()
        if not isinstance(obj.config, FailureRiskTrainingConfig):
            raise FailureRiskError("Use FailureRiskTrainingConfig")
        source = _source(provenance)
        obj.primary_binding = _validate_primary(primary, primary_checkpoint, expected_primary_sha256, expected_policy_runtime, source)
        names = _names(feature_names)
        if names != obj.primary_binding["feature_names"]:
            raise FailureRiskError("Typed raw feature names must exactly equal primary.features.names")
        train_x, dev_x = _x(train_x, names), _x(dev_x, names)
        train_y, dev_y = _labels(train_y_types, len(train_x)), _labels(dev_y_types, len(dev_x))
        tg, dg = _ids(train_groups, len(train_x), "train group"), _ids(dev_groups, len(dev_x), "dev group")
        te, de = _ids(train_episode_ids, len(train_x), "train episode"), _ids(dev_episode_ids, len(dev_x), "dev episode")
        test_groups = list(test_groups)
        test_groups = _ids(test_groups, len(test_groups), "test group")
        assert_group_disjoint(tg, dg, test_groups)
        for split, vector, rows in (("train", train_splits, len(train_x)), ("dev", dev_splits, len(dev_x))):
            if vector is not None and (len(vector) != rows or any(value != split for value in vector)):
                raise FailureRiskError(f"Test/undeclared split in {split} fitting rows")
        tw, dw = _weights(train_sample_weights, len(train_x)), _weights(dev_sample_weights, len(dev_x))
        obj.primary = primary
        obj.head_support = _head_support(_support(train_y, tg, te), _support(dev_y, dg, de))
        obj.typed_features = FailureTypeFeatures().fit(train_x, train_y, names, tg, te, tw)
        x = torch.from_numpy(obj.typed_features.transform(train_x, names))
        dx = torch.from_numpy(obj.typed_features.transform(dev_x, names))
        y, dy = torch.tensor(train_y, dtype=torch.float32), torch.tensor(dev_y, dtype=torch.float32)
        available = np.asarray([obj.head_availability[name] for name in FAILURE_TYPES])
        train_weights = np.stack([_episode_weights(tg, te, tw, np.isfinite(train_y[:, h]) & available[h]) for h in range(len(HEADS))], axis=1)
        dev_weights = np.stack([_episode_weights(dg, de, dw, np.isfinite(dev_y[:, h]) & available[h]) for h in range(len(HEADS))], axis=1)
        weights, dweights = torch.tensor(train_weights, dtype=torch.float32), torch.tensor(dev_weights, dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(obj.config.seed)
            obj.typed_model = FailureTypeMLP(x.shape[1]).cpu()
        obj.history, obj.selected_epoch = [], 0
        obj.temperatures = {name: None for name in FAILURE_TYPES}
        if available.any():
            optimizer = torch.optim.AdamW(obj.typed_model.parameters(), lr=obj.config.learning_rate, weight_decay=obj.config.weight_decay)
            generator = torch.Generator(device="cpu").manual_seed(obj.config.seed)
            best_loss, stale, best_state = math.inf, 0, None
            for epoch in range(obj.config.epochs):
                obj.typed_model.train()
                for batch in torch.randperm(len(x), generator=generator).split(obj.config.batch_size):
                    if not (weights[batch] > 0).any():
                        continue
                    loss = masked_binary_cross_entropy(obj.typed_model(x[batch]), y[batch], weights[batch], normalizers=weights.sum(0)) * (len(x) / len(batch))
                    if not torch.isfinite(loss):
                        raise FailureRiskError("Nonfinite type training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                obj.typed_model.eval()
                with torch.no_grad():
                    dev_loss = float(masked_binary_cross_entropy(obj.typed_model(dx), dy, dweights))
                if not math.isfinite(dev_loss):
                    raise FailureRiskError("Nonfinite type development loss")
                obj.history.append({"epoch": epoch + 1, "development_masked_episode_bce": dev_loss})
                if dev_loss < best_loss - 1e-8:
                    best_loss, stale = dev_loss, 0
                    best_state = {k: v.detach().clone() for k, v in obj.typed_model.state_dict().items()}
                    obj.selected_epoch = epoch + 1
                else:
                    stale += 1
                if stale >= obj.config.patience:
                    break
            obj.typed_model.load_state_dict(best_state)
            with torch.no_grad():
                logits = obj.typed_model(dx).numpy().astype(np.float64)
            for h, name in enumerate(FAILURE_TYPES):
                if not available[h]:
                    continue
                use = np.isfinite(dev_y[:, h])
                target, logit, weight = dev_y[use, h], logits[use, h], dev_weights[use, h]
                def objective(log_t):
                    scaled = logit / np.exp(log_t)
                    return float(np.sum((np.logaddexp(0, scaled) - target * scaled) * weight) / weight.sum())
                result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
                if not result.success or not math.isfinite(result.fun):
                    raise FailureRiskError(f"Development temperature calibration failed for {name}")
                obj.temperatures[name] = float(np.exp(result.x))
        obj.typed_model.eval().requires_grad_(False)
        obj.failure_provenance = {
            "schema": SCHEMA, "source_provenance": source, "source_provenance_sha256": _hash(source),
            "primary_checkpoint_sha256": expected_primary_sha256, "policy_runtime": _copy(expected_policy_runtime),
            "feature_names": names, "failure_types": list(FAILURE_TYPES),
            "training_rows": len(train_x), "development_rows": len(dev_x),
            "training_group_hash": _hash(sorted(set(tg))), "development_group_hash": _hash(sorted(set(dg))),
            "declared_test_group_hash": _hash(sorted(set(test_groups))),
            "training_arrays_sha256": _array_hash(train_x, train_y, tg, te, tw),
            "development_arrays_sha256": _array_hash(dev_x, dev_y, dg, de, dw),
            "standardization_fitted_on": "train_only_episode_weighted",
            "error_prototypes_fitted_on": "positive_observed_train_labels_only_equal_weight_per_error_episode",
            "training_loss": "independent_logits_masked_BCE_equal_weight_per_observed_episode_then_per_available_head",
            "epoch_selection_split": "dev", "temperature_calibration_split": "dev",
            "test_used_for_fit": False, "test_used_for_calibration": False,
            "unobserved_labels_are_negative": False,
            "generation_invalid_interpretation": "Observable formatting/interface risk; valid_json_action is directly visible and is not a white-box contribution claim.",
            "candidate_generation_failure_interpretation": "Successful generate_structures/create_structure RPC produced no accepted material candidate; distinct from LLM JSON/schema validity and RPC execution failure.",
            "source_code_sha256": {"failure_risk.py": file_sha256(__file__)},
        }
        # Sources/primary must still agree at the end of this independent fit.
        if _validate_primary(primary, primary_checkpoint, expected_primary_sha256, expected_policy_runtime, source) != obj.primary_binding:
            raise FailureRiskError("Primary changed during typed fitting")
        return obj

    def _payload(self):
        result = {"schema": SCHEMA, "config": asdict(self.config), "failure_types": list(FAILURE_TYPES),
                  "primary_binding": self.primary_binding, "features": self.typed_features.state(),
                  "model": self.typed_model.state_dict(), "temperatures": self.temperatures,
                  "head_support": self.head_support, "control_ready": self.control_ready,
                  "provenance": self.failure_provenance, "history": self.history, "selected_epoch": self.selected_epoch}
        result["content_sha256"] = _payload_hash(result)
        return result

    def save(self, path):
        """Atomically create only the typed checkpoint; return fields for a fit receipt.

        The caller writes ``method.failure_heads.fit.json`` with this result and
        its collection/fit evidence. Existing checkpoints are never overwritten.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload()
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {"schema": SCHEMA, "checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
                "content_sha256": payload["content_sha256"], "primary_checkpoint_sha256": self.primary_binding["checkpoint_sha256"],
                "head_availability": self.head_availability, "head_support": _copy(self.head_support),
                "control_ready": self.control_ready, "selected_epoch": self.selected_epoch,
                "temperatures": _copy(self.temperatures), "provenance": _copy(self.failure_provenance)}

    @classmethod
    def load(cls, path, *, primary, primary_checkpoint, expected_sha256,
             expected_primary_sha256, expected_policy_runtime, expected_provenance,
             require_control_ready=False):
        """No fallback: require external typed/primary hashes and source/runtime identity."""
        if file_sha256(path) != _sha(expected_sha256, "expected_sha256"):
            raise FailureRiskError("Typed checkpoint SHA256 mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state.get("schema") != SCHEMA or state.get("failure_types") != list(FAILURE_TYPES)
                or state.get("content_sha256") != _payload_hash(state)):
            raise FailureRiskError("Typed checkpoint schema/content checksum mismatch")
        source = _source(expected_provenance)
        binding = _validate_primary(primary, primary_checkpoint, expected_primary_sha256, expected_policy_runtime, source)
        if state["primary_binding"] != binding:
            raise FailureRiskError("Typed checkpoint was fitted for a different primary/schema/runtime/source")
        prov = state["provenance"]
        if (prov.get("source_provenance") != source or prov.get("source_provenance_sha256") != _hash(source)
                or prov.get("policy_runtime") != expected_policy_runtime or prov.get("primary_checkpoint_sha256") != expected_primary_sha256
                or prov.get("test_used_for_fit") is not False or prov.get("test_used_for_calibration") is not False):
            raise FailureRiskError("Typed checkpoint provenance mismatch")
        obj = cls()
        obj.primary, obj.primary_binding = primary, binding
        obj.config = FailureRiskTrainingConfig(**state["config"])
        obj.typed_features = FailureTypeFeatures.from_state(state["features"])
        if obj.typed_features.names != binding["feature_names"] or prov.get("feature_names") != binding["feature_names"]:
            raise FailureRiskError("Typed checkpoint feature schema mismatch")
        obj.head_support = state["head_support"]
        if set(obj.head_support) != set(FAILURE_TYPES):
            raise FailureRiskError("Typed checkpoint head support is incomplete")
        recomputed = _head_support({n: obj.head_support[n]["train"] for n in FAILURE_TYPES}, {n: obj.head_support[n]["dev"] for n in FAILURE_TYPES})
        if recomputed != obj.head_support or state["control_ready"] != obj.control_ready:
            raise FailureRiskError("Typed head availability is inconsistent with observed support")
        obj.temperatures = state["temperatures"]
        if set(obj.temperatures) != set(FAILURE_TYPES):
            raise FailureRiskError("Typed temperature inventory is incomplete")
        for name, available in obj.head_availability.items():
            value = obj.temperatures[name]
            if (available and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0)) or (not available and value is not None):
                raise FailureRiskError("Unsupported heads must have no probability/temperature")
        with torch.random.fork_rng(devices=[]):
            obj.typed_model = FailureTypeMLP(2 * len(binding["feature_names"]) + 3 * len(HEADS))
        obj.typed_model.load_state_dict(state["model"], strict=True)
        if any(not torch.isfinite(p).all() for p in obj.typed_model.parameters()):
            raise FailureRiskError("Nonfinite typed checkpoint parameters")
        obj.typed_model.eval().requires_grad_(False)
        obj.failure_provenance, obj.history, obj.selected_epoch = prov, state["history"], state["selected_epoch"]
        if file_sha256(path) != expected_sha256:
            raise FailureRiskError("Typed checkpoint changed while loading")
        if require_control_ready:
            obj.require_control_ready()
        return obj
