"""Reusable SVD-LoRA modules and helpers.

Self-test:
    pytest tests/test_svd_lora.py
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SVDLoRALinear(nn.Module):
    """Wrap an nn.Linear layer with an SVD-LoRA update.

    The update is parameterized as Delta W = B diag(E) A and applied as:
    original_linear(x) + scaling * (((x @ A.T) * E) @ B.T).
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        b_init: str = "normal",
        e_init: str = "ones",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"SVDLoRALinear can only wrap nn.Linear, got {type(linear)!r}")
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}")

        self.original_linear = linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        device = self.original_linear.weight.device
        dtype = self.original_linear.weight.dtype
        in_features = self.original_linear.in_features
        out_features = self.original_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(self.r, in_features, device=device, dtype=dtype))
        self.lora_E = nn.Parameter(torch.empty(self.r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, self.r, device=device, dtype=dtype))
        self.reset_parameters(b_init=b_init, e_init=e_init, init_std=init_std)

    def reset_parameters(self, *, b_init: str, e_init: str, init_std: float) -> None:
        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)

        if e_init == "ones":
            nn.init.ones_(self.lora_E)
        elif e_init == "normal":
            nn.init.normal_(self.lora_E, mean=0.0, std=init_std)
        else:
            raise ValueError(f"Unsupported e_init={e_init!r}; expected 'ones' or 'normal'.")

        if b_init == "normal":
            nn.init.normal_(self.lora_B, mean=0.0, std=init_std)
        elif b_init == "zero":
            nn.init.zeros_(self.lora_B)
        else:
            raise ValueError(f"Unsupported b_init={b_init!r}; expected 'normal' or 'zero'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.original_linear(x)
        hidden = torch.matmul(x, self.lora_A.transpose(0, 1))
        hidden = hidden * self.lora_E
        delta = torch.matmul(hidden, self.lora_B.transpose(0, 1))
        return base_output + self.scaling * delta


def apply_svd_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("query", "value"),
    r: int = 8,
    alpha: float = 16,
    b_init: str = "normal",
    e_init: str = "ones",
    init_std: float = 0.02,
) -> nn.Module:
    """Replace selected child nn.Linear modules with SVDLoRALinear wrappers."""
    target_module_names = set(target_modules)

    for child_name, child_module in list(model.named_children()):
        if isinstance(child_module, SVDLoRALinear):
            continue
        if child_name in target_module_names and isinstance(child_module, nn.Linear):
            setattr(
                model,
                child_name,
                SVDLoRALinear(
                    child_module,
                    r=r,
                    alpha=alpha,
                    b_init=b_init,
                    e_init=e_init,
                    init_std=init_std,
                ),
            )
        else:
            apply_svd_lora(
                child_module,
                target_modules=target_modules,
                r=r,
                alpha=alpha,
                b_init=b_init,
                e_init=e_init,
                init_std=init_std,
            )
    return model


def mark_only_svd_lora_as_trainable(model: nn.Module, train_classifier: bool = True) -> None:
    """Freeze all parameters except SVD-LoRA params and optional classifier params."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if name.endswith(("lora_A", "lora_E", "lora_B")):
            parameter.requires_grad = True
        elif train_classifier and "classifier" in name:
            parameter.requires_grad = True


def extract_svd_lora_state(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Extract CPU copies of A, E, and B from every SVDLoRALinear module."""
    state: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name, module in model.named_modules():
        if isinstance(module, SVDLoRALinear):
            state[layer_name] = {
                "A": module.lora_A.detach().cpu().clone(),
                "E": module.lora_E.detach().cpu().clone(),
                "B": module.lora_B.detach().cpu().clone(),
            }
    return state


def get_svd_lora_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return cloned CPU tensors for parameters currently marked trainable."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_svd_lora_trainable_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    """Load a trainable state dictionary into matching model parameters."""
    parameters = dict(model.named_parameters())
    trainable_names = {name for name, parameter in parameters.items() if parameter.requires_grad}
    saved_names = set(state_dict)

    missing = sorted(trainable_names - saved_names)
    unexpected = sorted(saved_names - trainable_names)
    if strict and (missing or unexpected):
        message = []
        if missing:
            message.append(f"missing keys: {missing}")
        if unexpected:
            message.append(f"unexpected keys: {unexpected}")
        raise ValueError("SVD-LoRA trainable state keys do not match: " + "; ".join(message))

    with torch.no_grad():
        for name, saved_value in state_dict.items():
            if name not in parameters:
                if strict:
                    raise ValueError(f"Unexpected key in SVD-LoRA state dict: {name}")
                continue
            parameter = parameters[name]
            if strict and not parameter.requires_grad:
                raise ValueError(f"State dict key is not trainable in target model: {name}")
            if tuple(parameter.shape) != tuple(saved_value.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: model has {tuple(parameter.shape)}, "
                    f"state has {tuple(saved_value.shape)}"
                )
            parameter.copy_(saved_value.to(device=parameter.device, dtype=parameter.dtype))


def print_trainable_parameters(model: nn.Module) -> None:
    """Print trainable and total parameter counts."""
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = 100.0 * trainable / total if total else 0.0
    print(f"Trainable parameters: {trainable}")
    print(f"Total parameters: {total}")
    print(f"Trainable ratio: {ratio:.4f}%")

