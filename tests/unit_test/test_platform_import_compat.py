# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

pytest.importorskip("sglang")


def test_platform_module_imports_without_optional_rocm(monkeypatch) -> None:
    platforms = importlib.import_module("sglang_omni.platforms")
    monkeypatch.setattr(platforms, "ROCMOmniPlatform", None)

    class FakePlatform:
        def is_cuda(self) -> bool:
            return False

        def is_rocm(self) -> bool:
            return True

        def is_cpu(self) -> bool:
            return False

        def is_xpu(self) -> bool:
            return False

    with pytest.raises(RuntimeError, match="requires sglang.srt.platforms.rocm"):
        platforms._as_omni_platform(FakePlatform())


def test_vendor_attention_tp_aliases_follow_parallel_state() -> None:
    vendor_layers = importlib.import_module("sglang_omni.vendor.sglang.layers")
    parallel_state = importlib.import_module("sglang.srt.distributed.parallel_state")

    assert (
        vendor_layers.get_attention_tp_rank
        is parallel_state.get_attn_tensor_model_parallel_rank
    )
    assert (
        vendor_layers.get_attention_tp_size
        is parallel_state.get_attn_tensor_model_parallel_world_size
    )


def test_musa_backend_policy_bypasses_cuda_policy(monkeypatch) -> None:
    cuda_platforms = importlib.import_module("sglang_omni.platforms.cuda")
    musa_platforms = importlib.import_module("sglang_omni.platforms.musa")
    omni_interface = importlib.import_module("sglang_omni.platforms.interface")

    calls = []

    def omni_policy(self, server_args, model_config, model_arch_override):
        del self, server_args, model_config, model_arch_override
        calls.append("omni")
        return "omni-policy"

    def cuda_policy(self, server_args, model_config, model_arch_override):
        del self, server_args, model_config, model_arch_override
        calls.append("cuda")
        return "cuda-policy"

    monkeypatch.setattr(
        omni_interface.OmniPlatform,
        "apply_model_worker_backend_policy",
        omni_policy,
    )
    monkeypatch.setattr(
        cuda_platforms.CUDAOmniPlatform,
        "apply_model_worker_backend_policy",
        cuda_policy,
    )

    platform = object.__new__(musa_platforms.MUSAOmniPlatform)
    result = platform.apply_model_worker_backend_policy(None, None, None)

    assert result == "omni-policy"
    assert calls == ["omni"]
