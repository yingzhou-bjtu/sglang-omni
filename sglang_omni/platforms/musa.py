# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform

if TYPE_CHECKING:
    from sglang_omni.platforms.interface import JointRopeInplaceKernel

logger = logging.getLogger(__name__)

try:
    import torchada  # noqa: F401
except ImportError as exc:
    logger.warning(
        f"Failed to import torchada: {exc}. MUSA platform compatibility will not work."
    )


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
) -> None:
    """Apply SGLang native rotary embedding to query and key in place."""
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

    rows = cos_sin_cache.index_select(0, positions)
    half = rows.shape[-1] // 2
    cos = rows[:, :half]
    sin = rows[:, half:]
    q.copy_(apply_rotary_emb(q, cos, sin, is_neox))
    k.copy_(apply_rotary_emb(k, cos, sin, is_neox))


class MUSAOmniPlatform(CUDAOmniPlatform):
    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"

    def get_fused_qk_norm_rope(self):
        # sgl-kernel's AOT fused_qk_norm_rope op is CUDA-only today.
        # Use the native QK-norm + RoPE path on MUSA.
        return None

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        # note (yzxiao): CUDA fused joint RoPE is a JIT kernel. MUSA answers
        # with SGLang native rotary embedding until that compiler path exists.
        return apply_rope_inplace

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        return OmniPlatform.apply_model_worker_backend_policy(
            self, server_args, model_config, model_arch_override
        )
