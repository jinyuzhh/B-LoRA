"""Train independent standard LoRA adapters for Experiment 1.

Each selected client starts from the same pretrained model, classifier head,
and LoRA adapter initialization. Only the saved shared trainable state is
loaded into each independently recreated client model before training.
"""

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


TARGET_MODULES = ["query", "value"]
DEFAULT_OUTPUT_DIR = "outputs/adapters/exp1_lora_b"
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


def get_trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return cloned CPU tensors for parameters currently marked trainable."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load saved trainable tensors into matching trainable model parameters."""
    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    saved_keys = set(state_dict)
    model_keys = set(trainable_parameters)
    missing = sorted(model_keys - saved_keys)
    unexpected = sorted(saved_keys - model_keys)
    if missing or unexpected:
        message = []
        if missing:
            message.append(f"missing keys: {missing}")
        if unexpected:
            message.append(f"unexpected keys: {unexpected}")
        raise ValueError("Trainable state keys do not match model trainable parameters: " + "; ".join(message))

    with torch.no_grad():
        for name, parameter in trainable_parameters.items():
            saved = state_dict[name]
            if tuple(parameter.shape) != tuple(saved.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: model has {tuple(parameter.shape)}, "
                    f"state has {tuple(saved.shape)}"
                )
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def verify_trainable_state_matches(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Verify that all current trainable parameters exactly match a saved state."""
    current = get_trainable_state_dict(model)
    if set(current) != set(state_dict):
        missing = sorted(set(current) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(current))
        raise ValueError(
            "Trainable state verification failed due to key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    mismatched = [
        name
        for name, value in current.items()
        if not torch.equal(value, state_dict[name].detach().cpu())
    ]
    if mismatched:
        raise ValueError(f"Trainable state verification failed for keys: {mismatched}")


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    print("Trainable parameters:")
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            print(f"  {name}: shape={tuple(parameter.shape)} count={count}")
    percent = 100.0 * trainable / total if total else 0.0
    print(f"Total trainable parameters: {trainable} / {total} ({percent:.4f}%)")


def enforce_classifier_trainability(model: torch.nn.Module, train_classifier: bool) -> None:
    """Ensure classifier-head parameters follow the experiment setting."""
    for name, parameter in model.named_parameters():
        if "classifier" not in name:
            continue
        if not train_classifier:
            parameter.requires_grad = False
        elif "original_module" in name:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True


def build_lora_model(
    *,
    model_name: str,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    train_classifier: bool,
    seed: int,
) -> torch.nn.Module:
    """Create a seeded sequence-classification model with PEFT LoRA attached."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "Training LoRA clients requires peft and transformers. "
            "Install the project training dependencies before running this script."
        ) from exc

    set_global_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
    )
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=TARGET_MODULES,
        modules_to_save=["classifier"] if train_classifier else None,
    )
    model = get_peft_model(model, config)
    enforce_classifier_trainability(model, train_classifier)
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


def save_initial_trainable_state(
    *,
    output_dir: Path,
    model_name: str,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    train_classifier: bool,
    seed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    init_state_path = output_dir / "initial_trainable_state.pt"

    initial_model = build_lora_model(
        model_name=model_name,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_classifier=train_classifier,
        seed=seed,
    )
    torch.save(get_trainable_state_dict(initial_model), init_state_path)
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
    lora_alpha: int,
    lora_dropout: float,
    train_classifier: bool,
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
            f"Experiment 1 expects {NUM_LABELS} labels, but client {client_id} "
            f"task {client_data.task_name} has {client_data.num_labels}."
        )

    model = build_lora_model(
        model_name=model_name,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_classifier=train_classifier,
        seed=seed,
    )
    init_state = load_saved_trainable_state(init_state_path)
    load_trainable_state_dict(model, init_state)
    verify_trainable_state_matches(model, init_state)

    print(f"\n=== Client {client_id} ({client_data.task_name}) ===")
    print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    training_args = create_training_arguments(
        output_dir=trainer_dir,
        epochs=epochs,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=client_data.train_dataset,
        eval_dataset=client_data.eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    adapter_path = client_dir / "adapter.pt"
    torch.save(get_trainable_state_dict(model), adapter_path)

    meta = {
        "client_id": client_id,
        "task_name": client_data.task_name,
        "model_name": model_name,
        "rank": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "seed": seed,
        "train_size": train_size,
        "eval_size": eval_size,
        "init_state_path": str(init_state_path),
        "train_classifier": train_classifier,
    }
    with (client_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, sort_keys=True)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent LoRA adapters for Experiment 1.")
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
    parser.add_argument("--r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument(
        "--train_classifier",
        type=parse_bool,
        default=True,
        help="Whether classifier-head parameters are trainable. Accepts true/false.",
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
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_classifier=args.train_classifier,
            init_state_path=init_state_path,
        )


if __name__ == "__main__":
    main()
