# SPDX-License-Identifier: Apache-2.0
"""Thinker TP factory configuration with CFG safeguards."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.models.llada2_uni import stages


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("algorithm", ["LowConfidence", "LowConfidenceCFG"])
def test_thinker_factory_preserves_tp_and_cfg_configuration(
    monkeypatch: pytest.MonkeyPatch, tp_size: int, tp_rank: int, algorithm: str
) -> None:
    if algorithm == "LowConfidenceCFG":
        pytest.importorskip("flashinfer.cascade")

    server_args = SimpleNamespace(dllm_algorithm=algorithm, mem_fraction_static=0.5)
    build_args = Mock(return_value=server_args)
    create_scheduler = Mock()
    monkeypatch.setattr(stages, "build_sglang_server_args", build_args)
    monkeypatch.setattr(stages, "create_dllm_thinker_scheduler", create_scheduler)
    monkeypatch.setattr(stages, "register_llada2_uni_cfg", Mock())
    override_args = Mock()
    monkeypatch.setattr(stages, "override_server_args", override_args)
    monkeypatch.setattr(stages, "resolved_view", lambda args: args)
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda *_args, **_kwargs: SimpleNamespace(type="cuda", index=tp_rank + 2),
    )
    if algorithm == "LowConfidenceCFG":
        monkeypatch.setattr(stages, "has_llada2_flashinfer_attention", lambda: True)
    else:
        pass

    scheduler = stages.create_sglang_dllm_thinker_executor_from_config(
        "model",
        device="cuda",
        gpu_id=tp_rank + 2,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=29500,
        dllm_algorithm=algorithm,
        server_args_overrides={"tp_size": 8, "mem_fraction_static": 0.5},
    )

    assert scheduler is create_scheduler.return_value
    create_scheduler.assert_called_once_with(
        server_args, tp_rank + 2, tp_rank=tp_rank, nccl_port=29500
    )
    settings = build_args.call_args.kwargs
    assert settings["tp_size"] == tp_size
    assert settings["disable_cuda_graph"] is True
    assert settings["mem_fraction_static"] == 0.5
    assert settings["dllm_algorithm"] == algorithm
    if algorithm == "LowConfidenceCFG":
        from sglang_omni.models.llada2_uni.cfg_attention_backend import (
            CFG_ATTENTION_BACKEND,
        )

        assert settings["attention_backend"] == CFG_ATTENTION_BACKEND
        assert settings["dllm_fdfo"] is False
        override_args.assert_called_once_with(
            server_args,
            "sglang_omni.llada2_uni.cfg_attention",
            attention_backend=CFG_ATTENTION_BACKEND,
        )
    else:
        assert settings["attention_backend"] == "flashinfer"


@pytest.mark.parametrize(
    "has_dense_flashinfer,expected_backend",
    [(False, "triton"), (True, "flashinfer")],
)
def test_thinker_factory_selects_musa_attention_by_capability(
    monkeypatch: pytest.MonkeyPatch,
    has_dense_flashinfer: bool,
    expected_backend: str,
) -> None:
    server_args = SimpleNamespace(
        dllm_algorithm="LowConfidence", mem_fraction_static=0.5
    )
    build_args = Mock(return_value=server_args)
    create_scheduler = Mock()
    monkeypatch.setattr(stages, "build_sglang_server_args", build_args)
    monkeypatch.setattr(stages, "create_dllm_thinker_scheduler", create_scheduler)
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda *_args, **_kwargs: SimpleNamespace(type="musa", index=0),
    )
    override_args = Mock()
    monkeypatch.setattr(stages, "override_server_args", override_args)
    monkeypatch.setattr(stages, "pin_resolved_device_type", lambda *_args: None)
    monkeypatch.setattr(stages, "resolved_view", lambda args: args)
    monkeypatch.setattr(
        stages,
        "has_llada2_flashinfer_attention",
        lambda: has_dense_flashinfer,
    )

    stages.create_sglang_dllm_thinker_executor_from_config(
        "model",
        device="musa",
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        nccl_port=29500,
        dllm_algorithm="LowConfidence",
    )

    assert build_args.call_args.kwargs["attention_backend"] == expected_backend
    override_args.assert_not_called()


def test_thinker_factory_rejects_cfg_without_dense_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stages,
        "has_llada2_flashinfer_attention",
        lambda: False,
    )
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda *_args, **_kwargs: SimpleNamespace(type="musa", index=0),
    )

    with pytest.raises(RuntimeError, match="requires FlashInfer dense attention"):
        stages.create_sglang_dllm_thinker_executor_from_config(
            "model",
            device="musa",
            gpu_id=0,
            dllm_algorithm="LowConfidenceCFG",
        )
