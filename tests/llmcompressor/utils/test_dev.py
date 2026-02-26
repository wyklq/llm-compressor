"""
Tests for llmcompressor.utils.dev(skip_weights_initialize, TORCH_INIT_FUNCTIONS import).

"""

import pytest
import torch

from llmcompressor.utils.dev import (
    TORCH_INIT_FUNCTIONS,
    skip_weights_initialize,
)


@pytest.mark.unit
def test_torch_init_functions_import():
    """TORCH_INIT_FUNCTIONS imports correctly under both transformers 4.x and 5.x."""
    assert TORCH_INIT_FUNCTIONS is not None
    assert isinstance(TORCH_INIT_FUNCTIONS, dict)
    assert len(TORCH_INIT_FUNCTIONS) > 0
    assert "uniform_" in TORCH_INIT_FUNCTIONS
    assert "normal_" in TORCH_INIT_FUNCTIONS
    assert "xavier_uniform_" in TORCH_INIT_FUNCTIONS


@pytest.mark.unit
def test_skip_weights_initialize_patch_and_restore():
    """skip_weights_initialize patches torch.nn.init and restores on exit."""
    layer = torch.nn.Linear(10, 5)
    orig_uniform = torch.nn.init.uniform_

    with skip_weights_initialize():
        assert torch.nn.init.uniform_ is not orig_uniform
        torch.nn.init.uniform_(layer.weight)

    assert torch.nn.init.uniform_ is orig_uniform


@pytest.mark.unit
def test_skip_weights_initialize_use_zeros():
    """skip_weights_initialize(use_zeros=True) zero-fills tensors."""
    layer = torch.nn.Linear(10, 5)
    with skip_weights_initialize(use_zeros=True):
        torch.nn.init.uniform_(layer.weight)
    assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
