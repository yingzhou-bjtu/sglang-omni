# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import torch

from sglang_omni.platforms import torchada_compat


def test_torchada_compatibility_patch_is_noop_off_musa() -> None:
    original = torch.Tensor.log_

    with patch.object(
        torchada_compat,
        "torchada_is_musa_platform",
        return_value=False,
    ):
        torchada_compat.apply_torchada_compatibility_patches()

    assert torch.Tensor.log_ is original


def test_torchada_compatibility_patch_is_idempotent() -> None:
    original = torch.Tensor.log_

    with patch.object(
        torchada_compat,
        "torchada_is_musa_platform",
        return_value=True,
    ):
        torchada_compat.apply_torchada_compatibility_patches()
        patched = torch.Tensor.log_
        torchada_compat.apply_torchada_compatibility_patches()

    try:
        assert patched is torch.Tensor.log_
        assert getattr(patched, torchada_compat._PATCHED_FLAG, False)
    finally:
        torch.Tensor.log_ = original
