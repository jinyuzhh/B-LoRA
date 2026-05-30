"""Plot distance heatmaps and same/different-task distance boxplots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SIMILARITY_DIR = "outputs/similarities/exp1_lora_b"
DEFAULT_OUTPUT_DIR = "outputs/figures/exp1_lora_b"


def load_client_meta(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing client metadata: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata: {path}") from exc

    if not isinstance(meta, list):
        raise ValueError(f"client_meta.json must contain a list: {path}")
    for index, item in enumerate(meta):
        if not isinstance(item, dict):
            raise ValueError(f"client_meta[{index}] must be an object.")
        missing = sorted({"client_id", "task_name", "feature_path"} - set(item))
        if missing:
            raise ValueError(f"client_meta[{index}] is missing required keys: {missing}")
    return meta


def validate_distance_matrix(distance_matrix: np.ndarray, client_meta: list[dict[str, Any]]) -> None:
    if distance_matrix.ndim != 2:
        raise ValueError(f"distance_matrix must be 2D, got shape {distance_matrix.shape}")
    rows, columns = distance_matrix.shape
    if rows != columns:
        raise ValueError(f"distance_matrix must be square, got shape {distance_matrix.shape}")
    if rows != len(client_meta):
        raise ValueError(
            f"distance_matrix size {rows} does not match client_meta length {len(client_meta)}"
        )
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("distance_matrix must be symmetric.")
    if not np.allclose(np.diag(distance_matrix), 0.0):
        raise ValueError("distance_matrix diagonal must be close to 0.")


def load_inputs(similarity_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    distance_path = similarity_dir / "distance_matrix.npy"
    meta_path = similarity_dir / "client_meta.json"
    if not distance_path.exists():
        raise FileNotFoundError(f"Missing distance matrix: {distance_path}")

    distance_matrix = np.load(distance_path)
    client_meta = load_client_meta(meta_path)
    validate_distance_matrix(distance_matrix, client_meta)
    return distance_matrix.astype(np.float64), client_meta


def sorted_client_order(client_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_meta = [
        {
            "original_index": index,
            "client_id": int(item["client_id"]),
            "task_name": str(item["task_name"]),
        }
        for index, item in enumerate(client_meta)
    ]
    sorted_meta = sorted(indexed_meta, key=lambda item: (item["task_name"], item["client_id"]))
    return [
        {
            "sorted_index": sorted_index,
            "client_id": item["client_id"],
            "task_name": item["task_name"],
            "original_index": item["original_index"],
        }
        for sorted_index, item in enumerate(sorted_meta)
    ]


def save_sorted_clients(sorted_clients: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sorted_clients.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(sorted_clients, file, indent=2, sort_keys=True)
    return path


def task_boundary_positions(sorted_clients: list[dict[str, Any]]) -> list[float]:
    boundaries: list[float] = []
    for index in range(1, len(sorted_clients)):
        if sorted_clients[index]["task_name"] != sorted_clients[index - 1]["task_name"]:
            boundaries.append(index - 0.5)
    return boundaries


def plot_heatmap(
    *,
    distance_matrix: np.ndarray,
    sorted_clients: list[dict[str, Any]],
    output_dir: Path,
    title: str,
    dpi: int,
) -> tuple[Path, Path]:
    order = [item["original_index"] for item in sorted_clients]
    reordered = distance_matrix[np.ix_(order, order)]
    labels = [f"{item['client_id']}-{item['task_name']}" for item in sorted_clients]

    size = max(6.0, min(12.0, 0.55 * len(sorted_clients) + 3.0))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(reordered, cmap="viridis", interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Distance")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)

    for boundary in task_boundary_positions(sorted_clients):
        ax.axhline(boundary, color="white", linewidth=1.5)
        ax.axvline(boundary, color="white", linewidth=1.5)

    fig.tight_layout()
    png_path = output_dir / "distance_heatmap.png"
    pdf_path = output_dir / "distance_heatmap.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def collect_pair_distances(
    distance_matrix: np.ndarray,
    client_meta: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    same_task: list[float] = []
    different_task: list[float] = []
    task_names = [str(item["task_name"]) for item in client_meta]

    for i in range(len(task_names)):
        for j in range(i + 1, len(task_names)):
            distance = float(distance_matrix[i, j])
            if task_names[i] == task_names[j]:
                same_task.append(distance)
            else:
                different_task.append(distance)

    if not same_task:
        raise ValueError("No same-task unordered client pairs are available for the boxplot.")
    if not different_task:
        raise ValueError("No different-task unordered client pairs are available for the boxplot.")
    return same_task, different_task


def plot_boxplot(
    *,
    distance_matrix: np.ndarray,
    client_meta: list[dict[str, Any]],
    output_dir: Path,
    title: str,
    dpi: int,
) -> tuple[Path, Path]:
    same_task, different_task = collect_pair_distances(distance_matrix, client_meta)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.boxplot(
        [same_task, different_task],
        labels=["Same task", "Different task"],
        showmeans=True,
    )
    ax.set_ylabel("Distance")
    ax.set_title(f"{title}: Same vs Different Task")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    png_path = output_dir / "same_vs_different_boxplot.png"
    pdf_path = output_dir / "same_vs_different_boxplot.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot client distance matrix results.")
    parser.add_argument(
        "--similarity_dir",
        default=DEFAULT_SIMILARITY_DIR,
        help="Directory containing distance_matrix.npy and client_meta.json.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figure files will be saved.",
    )
    parser.add_argument("--title", default="Client Distance Matrix", help="Plot title.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive; got {args.dpi}")

    similarity_dir = Path(args.similarity_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distance_matrix, client_meta = load_inputs(similarity_dir)
    sorted_clients = sorted_client_order(client_meta)
    sorted_clients_path = save_sorted_clients(sorted_clients, output_dir)
    heatmap_png, _ = plot_heatmap(
        distance_matrix=distance_matrix,
        sorted_clients=sorted_clients,
        output_dir=output_dir,
        title=args.title,
        dpi=args.dpi,
    )
    boxplot_png, _ = plot_boxplot(
        distance_matrix=distance_matrix,
        client_meta=client_meta,
        output_dir=output_dir,
        title=args.title,
        dpi=args.dpi,
    )

    print(f"Loaded {len(client_meta)} clients")
    print(f"Saved heatmap to {heatmap_png}")
    print(f"Saved boxplot to {boxplot_png}")
    print(f"Saved sorted client order to {sorted_clients_path}")


if __name__ == "__main__":
    main()
