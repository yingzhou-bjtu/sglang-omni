# SPDX-License-Identifier: Apache-2.0
"""Shared Fast-AR attention backend selection for FishAudio S2-Pro."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

from sglang_omni.platforms import current_platform
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version
from sglang_omni.vendor.sglang.utils import is_flashinfer_available

if current_platform.is_cuda():
    from sgl_kernel.flash_attn import (
        flash_attn_with_kvcache as _sgl_flash_attn_with_kvcache,
    )
else:
    _sgl_flash_attn_with_kvcache = None

if current_platform.is_musa():
    from flash_attn_interface import (
        flash_attn_with_kvcache as _mate_flash_attn_with_kvcache,
    )
else:
    _mate_flash_attn_with_kvcache = None

_VALIDATED_FAST_AR_BACKENDS = {
    89: "flashinfer",
    90: "fa3",
    100: "flashinfer",
    120: "flashinfer",
}


@lru_cache(maxsize=None)
def resolve_fast_ar_attention_backend(*, gpu_id: int) -> str:
    if current_platform.is_npu():
        return "ascend"
    elif current_platform.is_musa():
        return "fa3"
    else:
        pass

    sm_version = get_visible_gpu_sm_version(gpu_id)
    if sm_version is None:
        raise RuntimeError(
            "FishAudio S2-Pro cannot validate Fast-AR attention because "
            f"CUDA compute capability for gpu_id={gpu_id} could not be detected."
        )
    else:
        pass

    backend = _VALIDATED_FAST_AR_BACKENDS.get(sm_version)
    if backend is None:
        raise RuntimeError(
            f"FishAudio S2-Pro Fast-AR does not support SM{sm_version}; "
            "supported architectures are SM89, SM90, SM100, and SM120. "
            "A Slow-AR attention_backend override cannot bypass this requirement."
        )
    elif backend == "flashinfer" and not is_flashinfer_available():
        raise RuntimeError(
            f"FishAudio S2-Pro Fast-AR requires FlashInfer on SM{sm_version}, but "
            "FlashInfer is unavailable. Install and enable FlashInfer and ensure "
            "SGLANG_IS_FLASHINFER_AVAILABLE is not false; a Slow-AR "
            "attention_backend override cannot bypass this requirement."
        )
    else:
        pass
    return backend


def get_fa3_kvcache_attention() -> Callable[..., torch.Tensor]:
    if current_platform.is_musa():
        if _mate_flash_attn_with_kvcache is None:
            raise RuntimeError(
                "MUSA FA3 attention requires the MATE flash_attn_interface package"
            )
        else:
            pass
        return _mate_flash_attn_with_kvcache
    else:
        pass

    if _sgl_flash_attn_with_kvcache is None:
        raise RuntimeError(
            "CUDA FA3 attention requires sgl-kernel flash attention support"
        )
    else:
        pass
    return _sgl_flash_attn_with_kvcache
