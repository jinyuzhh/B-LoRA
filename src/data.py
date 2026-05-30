"""GLUE client data loading utilities.

This module defines the fixed 16-client task layout used by the experiments:

- clients 0-3: SST-2
- clients 4-7: QNLI
- clients 8-11: MRPC
- clients 12-15: QQP
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase
else:
    Dataset = Any
    PreTrainedTokenizerBase = Any


SUPPORTED_TASKS = ("sst2", "qnli", "mrpc", "qqp")
CLIENT_TASKS = {
    0: "sst2",
    1: "sst2",
    2: "sst2",
    3: "sst2",
    4: "qnli",
    5: "qnli",
    6: "qnli",
    7: "qnli",
    8: "mrpc",
    9: "mrpc",
    10: "mrpc",
    11: "mrpc",
    12: "qqp",
    13: "qqp",
    14: "qqp",
    15: "qqp",
}
TASK_TEXT_FIELDS = {
    "sst2": ("sentence",),
    "qnli": ("question", "sentence"),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
}


class ClientDataset(NamedTuple):
    train_dataset: Dataset
    eval_dataset: Dataset
    num_labels: int
    task_name: str
    client_id: int


def get_client_task(client_id: int) -> str:
    """Return the GLUE task assigned to a client id."""
    if client_id not in CLIENT_TASKS:
        valid_ids = f"{min(CLIENT_TASKS)}-{max(CLIENT_TASKS)}"
        raise ValueError(f"client_id must be in {valid_ids}; got {client_id}")
    return CLIENT_TASKS[client_id]


def _sample_split(
    dataset: Dataset,
    size: int,
    *,
    seed: int,
    split_name: str,
    task_name: str,
) -> Dataset:
    if size > len(dataset):
        raise ValueError(
            f"Requested {size} examples from {task_name} {split_name}, "
            f"but only {len(dataset)} are available."
        )
    return dataset.shuffle(seed=seed).select(range(size))


def _tokenize_dataset(
    dataset: Dataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    task_name: str,
    max_length: int,
) -> Dataset:
    text_fields = TASK_TEXT_FIELDS[task_name]

    def tokenize_batch(examples: dict) -> dict:
        texts = [examples[field] for field in text_fields]
        tokenized = tokenizer(
            *texts,
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {task_name}",
    )


def load_client_dataset(
    client_id: int,
    model_name_or_path: str,
    *,
    train_size: int = 1000,
    eval_size: int = 200,
    seed: int = 42,
    max_length: int = 128,
) -> ClientDataset:
    """Load and tokenize one client's GLUE train/eval datasets.

    Clients assigned to the same task use the same base seed plus their
    client id, which keeps sampling deterministic while giving each client a
    different subset.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "load_client_dataset requires Hugging Face datasets and transformers. "
            "Install them before loading GLUE clients."
        ) from exc

    task_name = get_client_task(client_id)
    raw = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    client_seed = seed + client_id
    train_raw = _sample_split(
        raw["train"],
        train_size,
        seed=client_seed,
        split_name="train",
        task_name=task_name,
    )
    eval_raw = _sample_split(
        raw["validation"],
        eval_size,
        seed=client_seed + 10_000,
        split_name="validation",
        task_name=task_name,
    )

    train_dataset = _tokenize_dataset(
        train_raw,
        tokenizer=tokenizer,
        task_name=task_name,
        max_length=max_length,
    )
    eval_dataset = _tokenize_dataset(
        eval_raw,
        tokenizer=tokenizer,
        task_name=task_name,
        max_length=max_length,
    )

    label_feature = raw["train"].features["label"]
    num_labels = getattr(label_feature, "num_classes", None)
    if num_labels is None:
        num_labels = len(set(raw["train"]["label"]))

    return ClientDataset(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_labels=num_labels,
        task_name=task_name,
        client_id=client_id,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and inspect GLUE client datasets.")
    parser.add_argument(
        "--model-name-or-path",
        default="bert-base-uncased",
        help="Tokenizer checkpoint name or local path.",
    )
    parser.add_argument("--client-id", type=int, default=None, help="Client id to inspect.")
    parser.add_argument("--train-size", type=int, default=1000, help="Training examples per client.")
    parser.add_argument("--eval-size", type=int, default=200, help="Evaluation examples per client.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max sequence length.")
    parser.add_argument(
        "--all-clients",
        action="store_true",
        help="Print task name and dataset sizes for all 16 clients.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client_ids = sorted(CLIENT_TASKS) if args.all_clients else [args.client_id if args.client_id is not None else 0]

    for client_id in client_ids:
        data = load_client_dataset(
            client_id=client_id,
            model_name_or_path=args.model_name_or_path,
            train_size=args.train_size,
            eval_size=args.eval_size,
            seed=args.seed,
            max_length=args.max_length,
        )
        print(
            f"client_id={data.client_id} "
            f"task={data.task_name} "
            f"train_size={len(data.train_dataset)} "
            f"eval_size={len(data.eval_dataset)} "
            f"num_labels={data.num_labels}"
        )


if __name__ == "__main__":
    main()
