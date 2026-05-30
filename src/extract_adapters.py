"""Extract LoRA B matrices from Experiment 1 client adapters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


DEFAULT_ADAPTER_DIR = "outputs/adapters/exp1_lora_b"
DEFAULT_OUTPUT_DIR = "outputs/features/exp1_lora_b"
CLIENT_DIR_PATTERN = re.compile(r"^client_(\d+)$")


def discover_clients(adapter_dir: Path) -> list[int]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    client_ids: list[int] = []
    for path in adapter_dir.iterdir():
        if not path.is_dir():
            continue
        match = CLIENT_DIR_PATTERN.match(path.name)
        if match:
            client_ids.append(int(match.group(1)))

    if not client_ids:
        raise FileNotFoundError(f"No client_* directories found under {adapter_dir}")
    return sorted(client_ids)


def normalize_lora_b_layer_name(parameter_name: str) -> str:
    """Normalize a PEFT LoRA-B parameter name to its original module path."""
    normalized = parameter_name
    for prefix in ("base_model.model.", "model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]

    normalized = re.sub(r"\.lora_B\.[^.]+\.weight$", "", normalized)
    normalized = re.sub(r"\.lora_B\.weight$", "", normalized)
    return normalized


def is_lora_b_matrix(name: str, value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and "lora_B" in name
        and name.endswith(".weight")
        and value.ndim == 2
    )


def extract_lora_b_layers(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    layers: dict[str, torch.Tensor] = {}
    source_names: dict[str, str] = {}

    for name, value in state_dict.items():
        if not is_lora_b_matrix(name, value):
            continue

        layer_name = normalize_lora_b_layer_name(name)
        if layer_name in layers:
            raise ValueError(
                "Duplicate normalized LoRA B layer name "
                f"{layer_name!r} from {name!r} and {source_names[layer_name]!r}"
            )
        layers[layer_name] = value.detach().cpu()
        source_names[layer_name] = name

    if not layers:
        raise ValueError("No LoRA B matrices found in adapter state_dict.")
    return layers


def load_metadata(meta_path: Path, client_id: int) -> dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file for client {client_id}: {meta_path}")

    try:
        with meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata for client {client_id}: {meta_path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata for client {client_id} must be a JSON object: {meta_path}")
    if "client_id" not in metadata:
        raise ValueError(f"Metadata for client {client_id} is missing 'client_id': {meta_path}")
    if int(metadata["client_id"]) != client_id:
        raise ValueError(
            f"Metadata client_id mismatch for {meta_path}: "
            f"expected {client_id}, got {metadata['client_id']}"
        )
    if "task_name" not in metadata:
        raise ValueError(f"Metadata for client {client_id} is missing 'task_name': {meta_path}")

    return metadata


def load_adapter_state(adapter_path: Path, client_id: int) -> dict[str, Any]:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter state for client {client_id}: {adapter_path}")

    state_dict = torch.load(adapter_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Adapter state for client {client_id} must be a dictionary: {adapter_path}")
    return state_dict


def validate_layer_keys(
    *,
    client_id: int,
    layer_keys: set[str],
    expected_keys: set[str] | None,
) -> set[str]:
    if expected_keys is None:
        return set(layer_keys)

    missing = sorted(expected_keys - layer_keys)
    extra = sorted(layer_keys - expected_keys)
    if missing or extra:
        message = [f"Client {client_id} has inconsistent LoRA B layer keys."]
        if missing:
            message.append(f"Missing: {missing}")
        if extra:
            message.append(f"Extra: {extra}")
        raise ValueError(" ".join(message))
    return expected_keys


def extract_client_features(
    *,
    adapter_dir: Path,
    output_dir: Path,
    client_id: int,
) -> tuple[dict[str, Any], set[str]]:
    client_dir = adapter_dir / f"client_{client_id}"
    adapter_path = client_dir / "adapter.pt"
    meta_path = client_dir / "meta.json"

    metadata = load_metadata(meta_path, client_id)
    state_dict = load_adapter_state(adapter_path, client_id)
    layers = extract_lora_b_layers(state_dict)

    feature = {
        "client_id": client_id,
        "task_name": metadata["task_name"],
        "source_adapter_path": str(adapter_path),
        "layers": layers,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(feature, output_dir / f"client_{client_id}.pt")

    print(f"client {client_id} task={metadata['task_name']} extracted {len(layers)} LoRA B matrices")
    return feature, set(layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LoRA B matrices from Experiment 1 adapters.")
    parser.add_argument(
        "--adapter_dir",
        default=DEFAULT_ADAPTER_DIR,
        help="Directory containing client_{id}/adapter.pt and meta.json files.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted client feature files will be saved.",
    )
    parser.add_argument(
        "--clients",
        nargs="+",
        type=int,
        default=None,
        help="Client ids to extract. If omitted, discover all client_* directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    client_ids = sorted(args.clients) if args.clients is not None else discover_clients(adapter_dir)

    expected_keys: set[str] | None = None
    for client_id in client_ids:
        _, layer_keys = extract_client_features(
            adapter_dir=adapter_dir,
            output_dir=output_dir,
            client_id=client_id,
        )
        expected_keys = validate_layer_keys(
            client_id=client_id,
            layer_keys=layer_keys,
            expected_keys=expected_keys,
        )

    print("All clients have consistent LoRA B layer keys.")


if __name__ == "__main__":
    main()
