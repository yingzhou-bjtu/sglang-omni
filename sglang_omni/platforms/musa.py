# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.srt.platforms.device_mixin import PlatformEnum

from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import JointRopeInplaceKernel, OmniPlatform


class MUSAOmniPlatform(CUDAOmniPlatform):
    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"

    def __init__(self) -> None:
        super().__init__()
        self.joint_rope_inplace_kernel: JointRopeInplaceKernel | None = None
        try:
            from torchada import install_sglang_rope_compat
        except ImportError:
            pass
        else:
            install_sglang_rope_compat()
            from sglang.kernels.ops.attention.rope import apply_rope_inplace

            self.joint_rope_inplace_kernel = apply_rope_inplace

    def get_joint_rope_inplace_kernel(self) -> JointRopeInplaceKernel:
        if self.joint_rope_inplace_kernel is None:
            raise RuntimeError(
                "MUSA joint RoPE requires torchada>=0.1.101 with "
                "install_sglang_rope_compat()"
            )
        else:
            pass
        return self.joint_rope_inplace_kernel

    def get_fused_qk_norm_rope(self):
        # sgl-kernel's AOT fused_qk_norm_rope op is CUDA-only today.
        # Use the native QK-norm + RoPE path on MUSA.
        return None

    def enable_codec_decode_graph(self) -> bool:
        return False

    def apply_model_worker_backend_policy(
        self,
        server_args: ServerArgs,
        model_config: ModelConfig,
        model_arch_override: str | None,
    ) -> str | None:
        return OmniPlatform.apply_model_worker_backend_policy(
            self, server_args, model_config, model_arch_override
        )
