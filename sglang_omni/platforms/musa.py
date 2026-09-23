# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.device_graph import (
    CudaDeviceGraphBackend,
    DeviceGraphBackend,
)
from sglang_omni.platforms.interface import JointRopeInplaceKernel, OmniPlatform

logger = logging.getLogger(__name__)


def apply_rope_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
) -> None:
    """Rotate query and key in place with MUSA's ordinary tensor ops.

    cos_sin_cache holds cosines first and sines second; positions selects its
    rows for the token-major [T, H, D] views, and is_neox picks the pairing.
    """
    half = cos_sin_cache.shape[-1] // 2
    rows = cos_sin_cache.index_select(0, positions)
    cos = rows[:, :half].unsqueeze(1)
    sin = rows[:, half:].unsqueeze(1)

    for tensor in (query, key):
        if is_neox:
            leading = tensor[..., :half]
            trailing = tensor[..., half:]
            rotated_leading = leading * cos - trailing * sin
            rotated_trailing = leading * sin + trailing * cos
            leading.copy_(rotated_leading)
            trailing.copy_(rotated_trailing)
        else:
            even = tensor[..., 0::2]
            odd = tensor[..., 1::2]
            rotated_even = even * cos - odd * sin
            rotated_odd = even * sin + odd * cos
            even.copy_(rotated_even)
            odd.copy_(rotated_odd)


try:
    import torchada  # noqa: F401
except ImportError as exc:
    logger.warning(
        f"Failed to import torchada: {exc}. MUSA platform compatibility will not work."
    )


# note (yingzhou): the runtime platform class is rebuilt from the SRT MUSA
# class, so these answers need a mixin that can sit ahead of OmniPlatform.
class MUSAOmniCapabilities:
    """MUSA answers for the optional Omni platform capabilities."""

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        # note (yingzhou): sgl-kernel's AOT op is CUDA-only; MUSA keeps the
        # capability by rotating with ordinary tensor ops.
        return apply_rope_inplace

    def _get_device_graph_backend(self) -> DeviceGraphBackend:
        # note (yingzhou): the private hook CUDA/ROCm/NPU/XPU override too;
        # CudaDeviceGraphBackend covers the backends behind torch.cuda.
        return CudaDeviceGraphBackend()

    def enable_breakable_prefill_graph(self) -> bool:
        # note (yingzhou): that path reads the stream capture status through
        # cuda-python, which MUSA does not provide.
        return False


class MUSAOmniPlatform(MUSAOmniCapabilities, CUDAOmniPlatform):
    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"

    def get_fused_qk_norm_rope(self):
        # sgl-kernel's AOT fused_qk_norm_rope op is CUDA-only today.
        # Use the native QK-norm + RoPE path on MUSA.
        return None

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        return OmniPlatform.apply_model_worker_backend_policy(
            self, server_args, model_config, model_arch_override
        )
