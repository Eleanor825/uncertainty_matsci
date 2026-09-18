"""Independent per-MLP TopK input-to-output transcoders and streamed training.

These modules are never attached to a policy. A transcoder learns y_MLP from
x_MLP, not an SAE reconstruction of x. Checkpoints include policy/data identity,
group-disjoint train/dev provenance, and held-out output reconstruction FVU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .esopt import tensor_state_hash


@dataclass(frozen=True)
class TranscoderConfig:
    input_dim: int
    output_dim: int
    feature_dim: int
    top_k: int

    def validate(self) -> None:
        if min(self.input_dim, self.output_dim, self.feature_dim, self.top_k) < 1 or self.top_k > self.feature_dim:
            raise ValueError("Positive dimensions and 1 <= top_k <= feature_dim are required.")


class TopKTranscoder(nn.Module):
    def __init__(self, config: TranscoderConfig, *, seed: int = 0):
        super().__init__()
        config.validate()
        self.config = config
        # Isolate initialization from the policy/sampling global random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(config.input_dim, config.feature_dim)
            self.decoder = nn.Linear(config.feature_dim, config.output_dim)
            nn.init.zeros_(self.encoder.bias)
            nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.config.input_dim:
            raise ValueError("Transcoder input dimension mismatch.")
        positive = F.relu(self.encoder(x))
        values, indices = positive.topk(self.config.top_k, dim=-1)
        return torch.zeros_like(positive).scatter(-1, indices, values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    def feature_directions(self, indices: torch.Tensor) -> torch.Tensor:
        """Rows are the output-space decoder vectors for the selected features."""
        return self.decoder.weight[:, indices].T

    def checkpoint_hash(self) -> str:
        return tensor_state_hash(dict(self.state_dict()))


def write_activation_shard(
    path: str | Path,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    group_ids: Sequence[str],
    prefix_hashes: Sequence[str],
    split: str,
    policy_fingerprint: str,
    layer_path: str,
) -> dict[str, Any]:
    """Write a small layer shard; one row per token/example, groups identify tasks.

    All rows from the same material system/episode must use the same group ID.
    Callers must collect actual MLP input/output pairs under the supplied policy.
    """
    if split not in {"train", "dev"}:
        raise ValueError("Transcoder fitting accepts train/dev only, never test.")
    if inputs.ndim != 2 or outputs.ndim != 2 or inputs.shape[0] != outputs.shape[0]:
        raise ValueError("Activation shards require paired 2D input/output rows.")
    rows = inputs.shape[0]
    if rows == 0 or len(group_ids) != rows or len(prefix_hashes) != rows:
        raise ValueError("Every activation row needs group and full-prefix identities.")
    if not policy_fingerprint or not layer_path or not all(group_ids) or not all(prefix_hashes):
        raise ValueError("Nonempty policy/layer/group/prefix identities are required.")
    state = {"inputs": inputs.detach().cpu().float().contiguous(), "outputs": outputs.detach().cpu().float().contiguous()}
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Activation shards cannot contain nonfinite values.")
    metadata = {"schema_version": 1, "split": split, "policy_fingerprint": policy_fingerprint,
                "layer_path": layer_path, "group_ids": list(group_ids), "prefix_hashes": list(prefix_hashes),
                "input_dim": inputs.shape[1], "output_dim": outputs.shape[1], "rows": rows,
                "tensor_hash": tensor_state_hash(state)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**state, "metadata": metadata}, path)
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _load_shard(path: str | Path) -> dict[str, Any]:
    shard = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_state_hash({"inputs": shard["inputs"], "outputs": shard["outputs"]}) != shard["metadata"]["tensor_hash"]:
        raise ValueError(f"Activation shard content hash mismatch: {path}")
    return shard


def validate_shard_splits(train_paths: Sequence[str | Path], dev_paths: Sequence[str | Path],
                          *, policy_fingerprint: str, layer_path: str,
                          config: TranscoderConfig) -> dict[str, Any]:
    """Validate manifests/content and group isolation without concatenating shards."""
    if not train_paths or not dev_paths:
        raise ValueError("Both training and held-out dev shards are required.")
    groups: dict[str, set[str]] = {"train": set(), "dev": set()}
    prefixes: dict[str, set[str]] = {"train": set(), "dev": set()}
    identities = []
    counts = {"train": 0, "dev": 0}
    for split, paths in (("train", train_paths), ("dev", dev_paths)):
        for path in paths:
            shard = _load_shard(path)
            meta = shard["metadata"]
            if meta["split"] != split or meta["policy_fingerprint"] != policy_fingerprint or meta["layer_path"] != layer_path:
                raise ValueError("Shard split, policy fingerprint, or MLP layer mismatch.")
            if (meta["input_dim"], meta["output_dim"]) != (config.input_dim, config.output_dim):
                raise ValueError("Shard dimensions disagree with the transcoder.")
            groups[split].update(meta["group_ids"])
            prefixes[split].update(meta["prefix_hashes"])
            counts[split] += meta["rows"]
            identities.append({"split": split, "tensor_hash": meta["tensor_hash"],
                               "groups": sorted(set(meta["group_ids"])), "prefixes": sorted(set(meta["prefix_hashes"]))})
            del shard
    if groups["train"] & groups["dev"] or prefixes["train"] & prefixes["dev"]:
        raise ValueError("Train/dev leakage: group or exact prefix occurs in both splits.")
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"dataset_fingerprint": digest, "policy_fingerprint": policy_fingerprint, "layer_path": layer_path,
            "rows": counts, "groups": {split: sorted(items) for split, items in groups.items()}, "shards": identities}


def iter_activation_batches(paths: Sequence[str | Path], batch_size: int, *,
                            seed: int, shuffle: bool) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(paths), generator=generator).tolist() if shuffle else range(len(paths))
    for index in order:
        shard = _load_shard(paths[index])
        x, y = shard["inputs"], shard["outputs"]
        rows = torch.randperm(x.shape[0], generator=generator) if shuffle else torch.arange(x.shape[0])
        for start in range(0, rows.numel(), batch_size):
            selected = rows[start:start + batch_size]
            yield x[selected], y[selected]
        del shard, x, y


@torch.no_grad()
def reconstruction_metrics(transcoder: TopKTranscoder, batches: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    """Output FVU = sum||y-yhat||² / sum||y-mean_dev(y)||² over all dev rows."""
    device = transcoder.encoder.weight.device
    dtype = transcoder.encoder.weight.dtype
    count = 0
    squared_error = 0.0
    output_sum = torch.zeros(transcoder.config.output_dim, dtype=torch.float64)
    output_square_sum = 0.0
    active_count = 0.0
    for x, y in batches:
        x, y = x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)
        activation = transcoder.encode(x)
        reconstruction = transcoder.decoder(activation)
        squared_error += (y.double() - reconstruction.double()).square().sum().item()
        output_sum += y.detach().cpu().double().sum(dim=0)
        output_square_sum += y.double().square().sum().item()
        active_count += (activation > 0).sum().item()
        count += y.shape[0]
    if count == 0:
        raise ValueError("Cannot evaluate empty dev data.")
    centered = max(0.0, output_square_sum - output_sum.square().sum().item() / count)
    return {"rows": float(count), "output_mse": squared_error / (count * transcoder.config.output_dim),
            "output_fvu": squared_error / centered if centered > 1e-12 else 0.0,
            "fvu_undefined": float(centered <= 1e-12), "mean_active_features": active_count / count}


def train_layer_transcoder(
    config: TranscoderConfig,
    train_paths: Sequence[str | Path],
    dev_paths: Sequence[str | Path],
    *,
    policy_fingerprint: str,
    layer_path: str,
    output_path: str | Path,
    seed: int = 0,
    epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    max_dev_fvu: float | None = None,
) -> tuple[TopKTranscoder, dict[str, Any]]:
    """Train one layer at a time; select the epoch using DEV, never test data.

    No samples are silently capped. The caller controls shard sizes and epochs.
    max_dev_fvu marks eligibility for graph use; undefined FVU never passes.
    """
    config.validate()
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("epochs, batch_size and finite learning_rate must be positive.")
    if max_dev_fvu is not None and (not math.isfinite(max_dev_fvu) or max_dev_fvu < 0):
        raise ValueError("max_dev_fvu must be finite and nonnegative.")
    provenance = validate_shard_splits(train_paths, dev_paths, policy_fingerprint=policy_fingerprint,
                                       layer_path=layer_path, config=config)
    transcoder = TopKTranscoder(config, seed=seed).to(device)
    optimizer = torch.optim.AdamW(transcoder.parameters(), lr=learning_rate, weight_decay=0.0)
    best_loss = math.inf
    best_state = None
    history = []
    for epoch in range(epochs):
        train_loss, rows = 0.0, 0
        transcoder.train()
        for x, y in iter_activation_batches(train_paths, batch_size, seed=seed + epoch, shuffle=True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(transcoder(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite transcoder training loss.")
            loss.backward()
            optimizer.step()
            transcoder.normalize_decoder_()
            train_loss += loss.item() * x.shape[0]
            rows += x.shape[0]
        transcoder.eval()
        dev = reconstruction_metrics(transcoder, iter_activation_batches(dev_paths, batch_size, seed=seed, shuffle=False))
        history.append({"epoch": epoch, "train_rows": rows, "train_output_mse": train_loss / rows, "dev": dev})
        if dev["output_mse"] < best_loss:
            best_loss = dev["output_mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in transcoder.state_dict().items()}
            best_epoch = epoch
    assert best_state is not None
    transcoder.load_state_dict(best_state)
    transcoder.eval()
    dev = history[best_epoch]["dev"]
    metadata = {"schema_version": 1, "method": "per_mlp_topk_input_to_output_v1", "config": asdict(config),
                "seed": seed, "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
                "selected_epoch": best_epoch, "history": history, "dev": dev, "max_dev_fvu": max_dev_fvu,
                "fidelity_gate_passed": max_dev_fvu is not None and not dev["fvu_undefined"] and dev["output_fvu"] <= max_dev_fvu,
                "provenance": provenance, "transcoder_hash": transcoder.checkpoint_hash()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, output_path)
    Path(str(output_path) + ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return transcoder, metadata


def load_transcoder(path: str | Path, *, expected_policy_fingerprint: str | None = None,
                    expected_layer_path: str | None = None) -> tuple[TopKTranscoder, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if expected_policy_fingerprint is not None and metadata["provenance"]["policy_fingerprint"] != expected_policy_fingerprint:
        raise ValueError("Transcoder was fitted to a different policy fingerprint.")
    if expected_layer_path is not None and metadata["provenance"]["layer_path"] != expected_layer_path:
        raise ValueError("Transcoder layer mismatch.")
    transcoder = TopKTranscoder(TranscoderConfig(**metadata["config"]), seed=metadata["seed"])
    transcoder.load_state_dict(checkpoint["state_dict"], strict=True)
    if transcoder.checkpoint_hash() != metadata["transcoder_hash"]:
        raise ValueError("Transcoder checkpoint hash mismatch.")
    return transcoder.eval(), metadata
