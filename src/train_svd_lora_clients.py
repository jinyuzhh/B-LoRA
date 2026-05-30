"""Train independent SVD-LoRA adapters for Experiment 2."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data import CLIENT_TASKS, load_client_dataset
from src.svd_lora import (
    apply_svd_lora,
    get_svd_lora_trainable_state_dict,
    load_svd_lora_trainable_state_dict,
    mark_only_svd_lora_as_trainable,
    print_trainable_parameters,
)


TARGET_MODULES = ("query", "value")
DEFAULT_OUTPUT_DIR = "outputs/adapters/exp2_svd_lora_be"
NUM_LABELS = 2


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def verify_trainable_state_matches(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    current = get_svd_lora_trainable_state_dict(model)
    if set(current) != set(state_dict):
        missing = sorted(set(current) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(current))
        raise ValueError(
            "SVD-LoRA trainable state verification failed due to key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    mismatched = [
        name
        for name, value in current.items()
        if not torch.equal(value, state_dict[name].detach().cpu())
    ]
    if mismatched:
        raise ValueError(f"SVD-LoRA trainable state verification failed for keys: {mismatched}")


def build_svd_lora_model(
    *,
    model_name: str,
    r: int,
    alpha: float,
    b_init: str,
    e_init: str,
    init_std: float,
    train_classifier: bool,
    seed: int,
) -> torch.nn.Module:
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "Training SVD-LoRA clients requires transformers. "
            "Install the project training dependencies before running this script."
        ) from exc

    set_global_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
    )
    apply_svd_lora(
        model,
        target_modules=TARGET_MODULES,
        r=r,
        alpha=alpha,
        b_init=b_init,
        e_init=e_init,
        init_std=init_std,
    )
    mark_only_svd_lora_as_trainable(model, train_classifier=train_classifier)
    return model


def create_training_arguments(
    *,
    output_dir: Path,
    epochs: float,
    seed: int,
) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 2e-5,
        "seed": seed,
        "data_seed": seed,
        "save_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    signature = inspect.signature(TrainingArguments)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


class SaveEpochStateCallback:
    """Save trainable SVD-LoRA state at the end of each epoch."""

    def __init__(self, client_dir: Path) -> None:
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                if model is None:
                    return control
                epoch = int(round(state.epoch or 0))
                epoch_dir = client_dir / f"epoch_{epoch}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                torch.save(get_svd_lora_trainable_state_dict(model), epoch_dir / "adapter.pt")
                return control

        self.callback = _Callback()


def save_initial_trainable_state(
    *,
    output_dir: Path,
    model_name: str,
    r: int,
    alpha: float,
    b_init: str,
    e_init: str,
    init_std: float,
    train_classifier: bool,
    seed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    init_state_path = output_dir / "initial_trainable_state.pt"

    initial_model = build_svd_lora_model(
        model_name=model_name,
        r=r,
        alpha=alpha,
        b_init=b_init,
        e_init=e_init,
        init_std=init_std,
        train_classifier=train_classifier,
        seed=seed,
    )
    torch.save(get_svd_lora_trainable_state_dict(initial_model), init_state_path)
    del initial_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return init_state_path


def load_saved_trainable_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def train_one_client(
    *,
    client_id: int,
    model_name: str,
    epochs: float,
    train_size: int,
    eval_size: int,
    seed: int,
    output_dir: Path,
    r: int,
    alpha: float,
    b_init: str,
    e_init: str,
    init_std: float,
    train_classifier: bool,
    save_epoch_states: bool,
    init_state_path: Path,
) -> None:
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer

    client_dir = output_dir / f"client_{client_id}"
    trainer_dir = client_dir / "trainer"
    client_dir.mkdir(parents=True, exist_ok=True)

    client_data = load_client_dataset(
        client_id=client_id,
        model_name_or_path=model_name,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    if client_data.num_labels != NUM_LABELS:
        raise ValueError(
            f"Experiment 2 expects {NUM_LABELS} labels, but client {client_id} "
            f"task {client_data.task_name} has {client_data.num_labels}."
        )

    model = build_svd_lora_model(
        model_name=model_name,
        r=r,
        alpha=alpha,
        b_init=b_init,
        e_init=e_init,
        init_std=init_std,
        train_classifier=train_classifier,
        seed=seed,
    )
    init_state = load_saved_trainable_state(init_state_path)
    load_svd_lora_trainable_state_dict(model, init_state, strict=True)
    verify_trainable_state_matches(model, init_state)

    print(f"\n=== Client {client_id} ({client_data.task_name}) ===")
    print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    training_args = create_training_arguments(
        output_dir=trainer_dir,
        epochs=epochs,
        seed=seed,
    )
    callbacks = []
    if save_epoch_states:
        callbacks.append(SaveEpochStateCallback(client_dir).callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=client_data.train_dataset,
        eval_dataset=client_data.eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=callbacks,
    )
    trainer.train()

    adapter_path = client_dir / "adapter.pt"
    torch.save(get_svd_lora_trainable_state_dict(model), adapter_path)

    meta = {
        "client_id": client_id,
        "task_name": client_data.task_name,
        "model_name": model_name,
        "r": r,
        "alpha": alpha,
        "b_init": b_init,
        "e_init": e_init,
        "init_std": init_std,
        "seed": seed,
        "train_size": train_size,
        "eval_size": eval_size,
        "train_classifier": train_classifier,
        "init_state_path": str(init_state_path),
    }
    with (client_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, sort_keys=True)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent SVD-LoRA adapters for Experiment 2.")
    parser.add_argument("--model_name", default="roberta-base", help="Pretrained model checkpoint.")
    parser.add_argument(
        "--clients",
        nargs="+",
        type=int,
        default=sorted(CLIENT_TASKS),
        help="Client ids to train, e.g. --clients 0 1 4 5.",
    )
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--train_size", type=int, default=1000, help="Training examples per client.")
    parser.add_argument("--eval_size", type=int, default=200, help="Evaluation examples per client.")
    parser.add_argument("--seed", type=int, default=42, help="Shared initialization and sampling seed.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Experiment adapter output directory.")
    parser.add_argument("--r", type=int, default=8, help="SVD-LoRA rank.")
    parser.add_argument("--alpha", type=float, default=16, help="SVD-LoRA alpha.")
    parser.add_argument("--b_init", choices=("normal", "zero"), default="normal", help="SVD-LoRA B init.")
    parser.add_argument("--e_init", choices=("ones", "normal"), default="ones", help="SVD-LoRA E init.")
    parser.add_argument("--init_std", type=float, default=0.02, help="SVD-LoRA normal init std.")
    parser.add_argument(
        "--train_classifier",
        type=parse_bool,
        default=True,
        help="Whether classifier-head parameters are trainable. Accepts true/false.",
    )
    parser.add_argument(
        "--save_epoch_states",
        type=parse_bool,
        default=False,
        help="Whether to save trainable adapter states after each epoch. Accepts true/false.",
    )
    return parser.parse_args()


def validate_clients(client_ids: list[int]) -> None:
    invalid = [client_id for client_id in client_ids if client_id not in CLIENT_TASKS]
    if invalid:
        valid_range = f"{min(CLIENT_TASKS)}-{max(CLIENT_TASKS)}"
        raise ValueError(f"Invalid client ids {invalid}; expected ids in {valid_range}.")


def main() -> None:
    args = parse_args()
    validate_clients(args.clients)

    output_dir = Path(args.output_dir)
    init_state_path = save_initial_trainable_state(
        output_dir=output_dir,
        model_name=args.model_name,
        r=args.r,
        alpha=args.alpha,
        b_init=args.b_init,
        e_init=args.e_init,
        init_std=args.init_std,
        train_classifier=args.train_classifier,
        seed=args.seed,
    )
    print(f"Saved shared initial trainable state to {init_state_path}")

    for client_id in args.clients:
        train_one_client(
            client_id=client_id,
            model_name=args.model_name,
            epochs=args.epochs,
            train_size=args.train_size,
            eval_size=args.eval_size,
            seed=args.seed,
            output_dir=output_dir,
            r=args.r,
            alpha=args.alpha,
            b_init=args.b_init,
            e_init=args.e_init,
            init_std=args.init_std,
            train_classifier=args.train_classifier,
            save_epoch_states=args.save_epoch_states,
            init_state_path=init_state_path,
        )


if __name__ == "__main__":
    main()
