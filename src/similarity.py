"""Compute client-client similarities from saved layer-wise features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch


DEFAULT_FEATURE_DIR = "outputs/features/exp1_lora_b"
DEFAULT_OUTPUT_DIR = "outputs/similarities/exp1_lora_b"
CLIENT_FILE_PATTERN = re.compile(r"^client_(\d+)\.pt$")


class ClientFeatures(NamedTuple):
    client_id: int
    task_name: str
    feature_path: Path
    layers: dict[str, torch.Tensor]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def discover_clients(feature_dir: Path) -> list[int]:
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    client_ids: list[int] = []
    for path in feature_dir.iterdir():
        if not path.is_file():
            continue
        match = CLIENT_FILE_PATTERN.match(path.name)
        if match:
            client_ids.append(int(match.group(1)))

    if not client_ids:
        raise FileNotFoundError(f"No client_*.pt files found under {feature_dir}")
    return sorted(client_ids)


def load_client_features(feature_dir: Path, client_id: int) -> ClientFeatures:
    feature_path = feature_dir / f"client_{client_id}.pt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file for client {client_id}: {feature_path}")

    feature = torch.load(feature_path, map_location="cpu")
    if not isinstance(feature, dict):
        raise ValueError(f"Feature file must contain a dictionary: {feature_path}")

    required_keys = {"client_id", "task_name", "layers"}
    missing = sorted(required_keys - set(feature))
    if missing:
        raise ValueError(f"Feature file {feature_path} is missing required keys: {missing}")

    loaded_client_id = int(feature["client_id"])
    if loaded_client_id != client_id:
        raise ValueError(
            f"Feature client_id mismatch for {feature_path}: "
            f"expected {client_id}, got {feature['client_id']}"
        )

    layers = feature["layers"]
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"Feature file {feature_path} must contain a non-empty layers dictionary.")

    tensor_layers: dict[str, torch.Tensor] = {}
    for layer_name, tensor in layers.items():
        if not isinstance(layer_name, str):
            raise ValueError(f"Feature file {feature_path} has a non-string layer key: {layer_name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Layer {layer_name!r} in {feature_path} is not a torch.Tensor.")
        tensor_layers[layer_name] = tensor.detach().cpu()

    return ClientFeatures(
        client_id=loaded_client_id,
        task_name=str(feature["task_name"]),
        feature_path=feature_path,
        layers=tensor_layers,
    )


def load_all_client_features(feature_dir: Path, client_ids: list[int]) -> list[ClientFeatures]:
    features = [load_client_features(feature_dir, client_id) for client_id in client_ids]
    return sorted(features, key=lambda item: item.client_id)


def validate_common_layer_keys(clients: list[ClientFeatures]) -> list[str]:
    if not clients:
        raise ValueError("No clients were loaded.")

    expected = set(clients[0].layers)
    for client in clients[1:]:
        current = set(client.layers)
        missing = sorted(expected - current)
        extra = sorted(current - expected)
        if missing or extra:
            message = [f"Client {client.client_id} has inconsistent layer keys."]
            if missing:
                message.append(f"Missing: {missing}")
            if extra:
                message.append(f"Extra: {extra}")
            raise ValueError(" ".join(message))

    return sorted(expected)


def maybe_normalize_columns(tensor: torch.Tensor, *, normalize_columns: bool, eps: float, layer_name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu().float()
    if not normalize_columns:
        return tensor
    if tensor.ndim != 2:
        raise ValueError(
            f"Column normalization requires 2D tensors, but layer {layer_name!r} "
            f"has shape {tuple(tensor.shape)}."
        )
    column_norms = torch.linalg.vector_norm(tensor, ord=2, dim=0, keepdim=True).clamp_min(eps)
    return tensor / column_norms


def cosine_similarity_safe(left: torch.Tensor, right: torch.Tensor, *, eps: float) -> float:
    left_flat = left.reshape(-1).float()
    right_flat = right.reshape(-1).float()
    left_norm = torch.linalg.vector_norm(left_flat, ord=2).clamp_min(eps)
    right_norm = torch.linalg.vector_norm(right_flat, ord=2).clamp_min(eps)
    similarity = torch.dot(left_flat, right_flat) / (left_norm * right_norm)
    if torch.isnan(similarity):
        return 0.0
    return float(similarity.item())


def compute_similarity_matrices(
    clients: list[ClientFeatures],
    layer_keys: list[str],
    *,
    normalize_columns: bool,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(clients)
    similarity_matrix = np.zeros((count, count), dtype=np.float64)
    distance_matrix = np.zeros((count, count), dtype=np.float64)

    normalized_layers: list[dict[str, torch.Tensor]] = []
    for client in clients:
        normalized_layers.append(
            {
                layer_name: maybe_normalize_columns(
                    client.layers[layer_name],
                    normalize_columns=normalize_columns,
                    eps=eps,
                    layer_name=layer_name,
                )
                for layer_name in layer_keys
            }
        )

    for i in range(count):
        similarity_matrix[i, i] = 1.0
        distance_matrix[i, i] = 0.0
        for j in range(i + 1, count):
            layer_similarities = [
                cosine_similarity_safe(
                    normalized_layers[i][layer_name],
                    normalized_layers[j][layer_name],
                    eps=eps,
                )
                for layer_name in layer_keys
            ]
            similarity = float(np.mean(layer_similarities))
            distance = float(np.mean([1.0 - value for value in layer_similarities]))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2.0
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    np.fill_diagonal(similarity_matrix, 1.0)
    np.fill_diagonal(distance_matrix, 0.0)
    return similarity_matrix, distance_matrix


def save_outputs(
    *,
    output_dir: Path,
    clients: list[ClientFeatures],
    similarity_matrix: np.ndarray,
    distance_matrix: np.ndarray,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    similarity_path = output_dir / "similarity_matrix.npy"
    distance_path = output_dir / "distance_matrix.npy"
    meta_path = output_dir / "client_meta.json"

    np.save(similarity_path, similarity_matrix)
    np.save(distance_path, distance_matrix)

    client_meta = [
        {
            "client_id": client.client_id,
            "task_name": client.task_name,
            "feature_path": str(client.feature_path),
        }
        for client in clients
    ]
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(client_meta, file, indent=2, sort_keys=True)

    return similarity_path, distance_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute client-client cosine similarities from feature files.")
    parser.add_argument(
        "--feature_dir",
        default=DEFAULT_FEATURE_DIR,
        help="Directory containing client_{id}.pt feature files.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where similarity outputs will be saved.",
    )
    parser.add_argument(
        "--clients",
        nargs="+",
        type=int,
        default=None,
        help="Client ids to compare. If omitted, discover all client_*.pt files.",
    )
    parser.add_argument(
        "--normalize_columns",
        type=parse_bool,
        default=False,
        help="Whether to L2-normalize columns of each 2D layer tensor. Accepts true/false.",
    )
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for safe normalization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eps <= 0:
        raise ValueError(f"--eps must be positive; got {args.eps}")

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    client_ids = sorted(args.clients) if args.clients is not None else discover_clients(feature_dir)
    clients = load_all_client_features(feature_dir, client_ids)
    layer_keys = validate_common_layer_keys(clients)

    similarity_matrix, distance_matrix = compute_similarity_matrices(
        clients,
        layer_keys,
        normalize_columns=args.normalize_columns,
        eps=args.eps,
    )
    similarity_path, distance_path, _ = save_outputs(
        output_dir=output_dir,
        clients=clients,
        similarity_matrix=similarity_matrix,
        distance_matrix=distance_matrix,
    )

    print(f"Loaded {len(clients)} clients")
    print(f"Number of common layers: {len(layer_keys)}")
    print(f"Saved similarity matrix to {similarity_path}")
    print(f"Saved distance matrix to {distance_path}")


if __name__ == "__main__":
    main()
