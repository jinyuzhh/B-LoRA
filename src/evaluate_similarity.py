"""Evaluate whether client distance matrices recover task groups."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)


DEFAULT_SIMILARITY_DIR = "outputs/similarities/exp1_lora_b"
DEFAULT_OUTPUT_DIR = "outputs/metrics/exp1_lora_b"


def clustering_purity(y_true, y_pred):
    """Compute clustering purity for aligned true and predicted labels."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred lengths differ: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("Cannot compute purity for empty labels.")

    total_correct = 0
    for cluster in sorted(set(y_pred)):
        true_labels = [true for true, pred in zip(y_true, y_pred) if pred == cluster]
        if true_labels:
            total_correct += Counter(true_labels).most_common(1)[0][1]
    return total_correct / len(y_true)


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


def load_inputs(similarity_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    distance_path = similarity_dir / "distance_matrix.npy"
    meta_path = similarity_dir / "client_meta.json"
    if not distance_path.exists():
        raise FileNotFoundError(f"Missing distance matrix: {distance_path}")

    distance_matrix = np.load(distance_path)
    client_meta = load_client_meta(meta_path)
    validate_distance_matrix(distance_matrix, client_meta)
    return distance_matrix.astype(np.float64), client_meta


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


def pairwise_statistics(distance_matrix: np.ndarray, task_names: list[str]) -> dict[str, Any]:
    same_distances: list[float] = []
    different_distances: list[float] = []
    auc_labels: list[int] = []
    auc_scores: list[float] = []

    for i in range(len(task_names)):
        for j in range(i + 1, len(task_names)):
            distance = float(distance_matrix[i, j])
            is_same_task = task_names[i] == task_names[j]
            if is_same_task:
                same_distances.append(distance)
            else:
                different_distances.append(distance)
            auc_labels.append(1 if is_same_task else 0)
            auc_scores.append(-distance)

    mean_same = float(np.mean(same_distances)) if same_distances else None
    std_same = float(np.std(same_distances)) if same_distances else None
    mean_different = float(np.mean(different_distances)) if different_distances else None
    std_different = float(np.std(different_distances)) if different_distances else None
    separation_gap = (
        mean_different - mean_same
        if mean_same is not None and mean_different is not None
        else None
    )

    auc: float | None
    if len(set(auc_labels)) < 2:
        warnings.warn("AUC cannot be computed because pair labels contain only one class.", RuntimeWarning)
        auc = None
    else:
        auc = float(roc_auc_score(auc_labels, auc_scores))

    return {
        "num_same_task_pairs": len(same_distances),
        "num_different_task_pairs": len(different_distances),
        "mean_same_task_distance": mean_same,
        "std_same_task_distance": std_same,
        "mean_different_task_distance": mean_different,
        "std_different_task_distance": std_different,
        "separation_gap": separation_gap,
        "auc": auc,
    }


def agglomerative_precomputed(distance_matrix: np.ndarray, *, num_clusters: int, linkage: str) -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="precomputed",
            linkage=linkage,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            linkage=linkage,
        )
    return model.fit_predict(distance_matrix)


def validate_cluster_range(
    *,
    num_clients: int,
    min_clusters: int,
    max_clusters: int | None,
    force_num_clusters: int | None,
) -> tuple[int, int]:
    if num_clients < 3:
        raise ValueError("At least 3 clients are required for silhouette-based cluster selection.")

    if force_num_clusters is not None:
        if force_num_clusters < 2 or force_num_clusters >= num_clients:
            raise ValueError(
                "--force_num_clusters must be between 2 and num_clients - 1 "
                f"for silhouette scoring; got {force_num_clusters} for {num_clients} clients."
            )
        return force_num_clusters, force_num_clusters

    if min_clusters < 2:
        raise ValueError(f"--min_clusters must be at least 2; got {min_clusters}")

    resolved_max = num_clients - 1 if max_clusters is None else max_clusters
    if resolved_max >= num_clients:
        raise ValueError(
            f"--max_clusters must be at most num_clients - 1 ({num_clients - 1}); got {resolved_max}"
        )
    if resolved_max < min_clusters:
        raise ValueError(
            f"Invalid cluster search range: min_clusters={min_clusters}, max_clusters={resolved_max}"
        )
    return min_clusters, resolved_max


def select_clusters_by_silhouette(
    distance_matrix: np.ndarray,
    *,
    linkage: str,
    min_clusters: int,
    max_clusters: int | None,
    force_num_clusters: int | None,
) -> tuple[np.ndarray, int, str, int, int, list[dict[str, float | int]], float | None]:
    num_clients = distance_matrix.shape[0]
    search_min, search_max = validate_cluster_range(
        num_clients=num_clients,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        force_num_clusters=force_num_clusters,
    )

    if force_num_clusters is not None:
        labels = agglomerative_precomputed(
            distance_matrix,
            num_clusters=force_num_clusters,
            linkage=linkage,
        )
        silhouette = compute_silhouette(distance_matrix, labels)
        score = None if silhouette is None else float(silhouette)
        scores = [{"num_clusters": force_num_clusters, "silhouette": score}]
        return labels, force_num_clusters, "forced", search_min, search_max, scores, silhouette

    best_labels: np.ndarray | None = None
    best_k: int | None = None
    best_score: float | None = None
    scores: list[dict[str, float | int]] = []

    for num_clusters in range(search_min, search_max + 1):
        labels = agglomerative_precomputed(
            distance_matrix,
            num_clusters=num_clusters,
            linkage=linkage,
        )
        silhouette = compute_silhouette(distance_matrix, labels)
        if silhouette is None:
            continue
        score = float(silhouette)
        scores.append({"num_clusters": num_clusters, "silhouette": score})
        if best_score is None or score > best_score:
            best_score = score
            best_k = num_clusters
            best_labels = labels

    if best_labels is None or best_k is None or best_score is None:
        raise ValueError(
            f"Could not select a cluster count with silhouette over range {search_min}..{search_max}."
        )

    return best_labels, best_k, "silhouette", search_min, search_max, scores, best_score


def compute_silhouette(distance_matrix: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        return float(silhouette_score(distance_matrix, labels, metric="precomputed"))
    except ValueError as exc:
        warnings.warn(f"Silhouette score cannot be computed: {exc}", RuntimeWarning)
        return None


def compute_metrics(
    *,
    distance_matrix: np.ndarray,
    client_meta: list[dict[str, Any]],
    linkage: str,
    min_clusters: int = 2,
    max_clusters: int | None = None,
    force_num_clusters: int | None = None,
) -> dict[str, Any]:
    task_names = [str(item["task_name"]) for item in client_meta]
    tasks = sorted(set(task_names))
    true_labels = [tasks.index(task_name) for task_name in task_names]

    pair_stats = pairwise_statistics(distance_matrix, task_names)
    (
        cluster_labels,
        selected_num_clusters,
        cluster_selection_method,
        cluster_search_min,
        cluster_search_max,
        cluster_search_scores,
        silhouette,
    ) = select_clusters_by_silhouette(
        distance_matrix,
        linkage=linkage,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        force_num_clusters=force_num_clusters,
    )

    metrics = {
        "num_clients": len(client_meta),
        "num_tasks": len(tasks),
        "tasks": tasks,
        "selected_num_clusters": selected_num_clusters,
        "cluster_selection_method": cluster_selection_method,
        "cluster_search_min": cluster_search_min,
        "cluster_search_max": cluster_search_max,
        "cluster_search_scores": cluster_search_scores,
        **pair_stats,
        "ari": float(adjusted_rand_score(true_labels, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, cluster_labels)),
        "purity": float(clustering_purity(true_labels, cluster_labels.tolist())),
        "silhouette": silhouette,
        "clustering_labels": [int(label) for label in cluster_labels.tolist()],
    }
    return metrics


def save_metrics(metrics: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True, allow_nan=False)
    return metrics_path


def format_optional(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "null"
    return f"{mean:.4f} +/- {std:.4f}"


def print_metrics_table(metrics: dict[str, Any]) -> None:
    same = format_mean_std(
        metrics["mean_same_task_distance"],
        metrics["std_same_task_distance"],
    )
    different = format_mean_std(
        metrics["mean_different_task_distance"],
        metrics["std_different_task_distance"],
    )

    rows = [
        ("# Clients", str(metrics["num_clients"])),
        ("# True Tasks", str(metrics["num_tasks"])),
        ("# Selected Clusters", str(metrics["selected_num_clusters"])),
        ("Cluster selection", str(metrics["cluster_selection_method"])),
        ("Best silhouette", format_optional(metrics["silhouette"])),
        ("Same-task distance", same),
        ("Different-task distance", different),
        ("Separation gap", format_optional(metrics["separation_gap"])),
        ("AUC", format_optional(metrics["auc"])),
        ("ARI", f"{metrics['ari']:.4f}"),
        ("NMI", f"{metrics['nmi']:.4f}"),
        ("Purity", f"{metrics['purity']:.4f}"),
        ("Silhouette", format_optional(metrics["silhouette"])),
    ]

    print(f"{'Metric':<30} Value")
    print("-" * 36)
    for name, value in rows:
        print(f"{name:<30} {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task recovery from client distance matrices.")
    parser.add_argument(
        "--similarity_dir",
        default=DEFAULT_SIMILARITY_DIR,
        help="Directory containing distance_matrix.npy and client_meta.json.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where metrics.json will be saved.",
    )
    parser.add_argument(
        "--linkage",
        choices=("average", "complete"),
        default="average",
        help="Agglomerative clustering linkage to use with precomputed distances.",
    )
    parser.add_argument("--min_clusters", type=int, default=2, help="Minimum clusters for automatic search.")
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=None,
        help="Maximum clusters for automatic search. Defaults to num_clients - 1.",
    )
    parser.add_argument(
        "--force_num_clusters",
        type=int,
        default=None,
        help="Force a fixed cluster count instead of automatic silhouette selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distance_matrix, client_meta = load_inputs(Path(args.similarity_dir))
    metrics = compute_metrics(
        distance_matrix=distance_matrix,
        client_meta=client_meta,
        linkage=args.linkage,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        force_num_clusters=args.force_num_clusters,
    )
    metrics_path = save_metrics(metrics, Path(args.output_dir))
    print_metrics_table(metrics)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
