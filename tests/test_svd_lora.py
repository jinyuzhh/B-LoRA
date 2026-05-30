import torch
from torch import nn

from src.svd_lora import (
    SVDLoRALinear,
    apply_svd_lora,
    extract_svd_lora_state,
    get_svd_lora_trainable_state_dict,
    load_svd_lora_trainable_state_dict,
    mark_only_svd_lora_as_trainable,
)


class DummySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.key = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4)
        self.dense = nn.Linear(4, 4)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = DummySelfAttention()


class DummyRobertaLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([DummyLayer(), DummyLayer()])
        self.classifier = nn.Linear(4, 2)


def test_svd_lora_linear_preserves_2d_and_3d_shapes():
    wrapped = SVDLoRALinear(nn.Linear(4, 3), r=2, alpha=4)

    assert wrapped(torch.randn(5, 4)).shape == (5, 3)
    assert wrapped(torch.randn(5, 7, 4)).shape == (5, 7, 3)


def test_original_linear_is_frozen_and_lora_params_are_trainable():
    wrapped = SVDLoRALinear(nn.Linear(4, 3), r=2)

    assert not wrapped.original_linear.weight.requires_grad
    assert wrapped.original_linear.bias is not None
    assert not wrapped.original_linear.bias.requires_grad
    assert wrapped.lora_A.requires_grad
    assert wrapped.lora_E.requires_grad
    assert wrapped.lora_B.requires_grad


def test_apply_svd_lora_replaces_only_query_and_value_modules():
    model = apply_svd_lora(DummyRobertaLike(), target_modules=("query", "value"), r=2)

    for layer in model.encoder.layer:
        attention = layer.attention.self
        assert isinstance(attention.query, SVDLoRALinear)
        assert isinstance(attention.value, SVDLoRALinear)
        assert isinstance(attention.key, nn.Linear)
        assert isinstance(attention.dense, nn.Linear)
    assert isinstance(model.classifier, nn.Linear)


def test_mark_only_svd_lora_as_trainable_toggles_classifier():
    model = apply_svd_lora(DummyRobertaLike(), r=2)

    mark_only_svd_lora_as_trainable(model, train_classifier=True)
    trainable_true = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any(name.endswith("lora_A") for name in trainable_true)
    assert any(name.endswith("lora_E") for name in trainable_true)
    assert any(name.endswith("lora_B") for name in trainable_true)
    assert any("classifier" in name for name in trainable_true)

    mark_only_svd_lora_as_trainable(model, train_classifier=False)
    trainable_false = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any(name.endswith("lora_A") for name in trainable_false)
    assert any(name.endswith("lora_E") for name in trainable_false)
    assert any(name.endswith("lora_B") for name in trainable_false)
    assert not any("classifier" in name for name in trainable_false)


def test_extract_svd_lora_state_returns_expected_layers_and_shapes():
    model = apply_svd_lora(DummyRobertaLike(), r=2)
    state = extract_svd_lora_state(model)

    assert len(state) == 4
    assert "encoder.layer.0.attention.self.query" in state
    assert "encoder.layer.0.attention.self.value" in state
    for layer_state in state.values():
        assert set(layer_state) == {"A", "E", "B"}
        assert layer_state["A"].shape == (2, 4)
        assert layer_state["E"].shape == (2,)
        assert layer_state["B"].shape == (4, 2)
        assert layer_state["A"].device.type == "cpu"


def test_saved_and_loaded_trainable_state_dicts_match():
    torch.manual_seed(123)
    source = apply_svd_lora(DummyRobertaLike(), r=2)
    mark_only_svd_lora_as_trainable(source, train_classifier=True)
    source_state = get_svd_lora_trainable_state_dict(source)

    torch.manual_seed(456)
    target = apply_svd_lora(DummyRobertaLike(), r=2)
    mark_only_svd_lora_as_trainable(target, train_classifier=True)
    load_svd_lora_trainable_state_dict(target, source_state)
    target_state = get_svd_lora_trainable_state_dict(target)

    assert set(source_state) == set(target_state)
    for name, tensor in source_state.items():
        assert torch.equal(tensor, target_state[name])

