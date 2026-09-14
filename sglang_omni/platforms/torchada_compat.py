# SPDX-License-Identifier: Apache-2.0
"""Temporary downstream patches for gaps in the installed torchada release."""

from __future__ import annotations

import functools

import torch

_PATCHED_FLAG = "_sglang_omni_torchada_log_inplace_patched"


def apply_torchada_compatibility_patches() -> None:
    """Patch only the MUSA behavior missing from the installed torchada build.

    torchada currently translates the CUDA-style seeded sampler path to MUSA,
    but its float64 ``Tensor.log_`` operation is unsupported by MUDNN in the
    validated runtime. Keep the in-place contract by computing through the
    supported float32 kernel. This is a temporary downstream patch until the
    behavior is available in torchada or torch_musa.
    """
    try:
        is_musa_platform = torchada_is_musa_platform()
    except ImportError:
        return
    if not is_musa_platform:
        return

    original = torch.Tensor.log_
    if getattr(original, _PATCHED_FLAG, False):
        return

    @functools.wraps(original)
    def log_inplace_compat(self: torch.Tensor) -> torch.Tensor:
        if self.device.type == "musa" and self.dtype == torch.float64:
            result = torch.log(self.to(dtype=torch.float32))
            return self.copy_(result.to(dtype=torch.float64))
        return original(self)

    setattr(log_inplace_compat, _PATCHED_FLAG, True)
    torch.Tensor.log_ = log_inplace_compat


def torchada_is_musa_platform() -> bool:
    """Return torchada's platform decision without duplicating detection."""
    import torchada

    return bool(torchada.is_musa_platform())
