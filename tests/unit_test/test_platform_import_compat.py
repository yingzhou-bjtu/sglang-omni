# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

pytest.importorskip("sglang")


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
