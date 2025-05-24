# SPDX-License-Identifier: Apache-2.0
"""Tests for QTIP quantization configuration.

Run `pytest tests/quantization/test_qtip_config.py`.
"""

import torch

from vllm.model_executor.layers.quantization.qtip import QTIPConfig


def test_qtip_config_creation():
    """Test QTIP configuration creation and validation."""

    config = QTIPConfig(td_x=8,
                        td_y=8,
                        L=16,
                        K=2,
                        V=2,
                        tlut_bits=16,
                        decode_mode="1mad",
                        scale=32.0)

    assert config.td_x == 8
    assert config.td_y == 8
    assert config.L == 16
    assert config.K == 2
    assert config.V == 2
    assert config.tlut_bits == 16
    assert config.decode_mode == "1mad"
    assert config.scale == 32.0
    assert config.pack_factor == 64  # td_x * td_y

    config_dict = {
        "td_x": 8,
        "td_y": 8,
        "L": 16,
        "K": 2,
        "V": 2,
        "tlut_bits": 16,
        "decode_mode": "1mad",
        "scale": 32.0
    }
    config_from_dict = QTIPConfig.from_config(config_dict)
    assert config_from_dict.td_x == config.td_x
    assert config_from_dict.td_y == config.td_y
    assert config_from_dict.L == config.L
    assert config_from_dict.K == config.K
    assert config_from_dict.V == config.V
    assert config_from_dict.tlut_bits == config.tlut_bits
    assert config_from_dict.decode_mode == config.decode_mode
    assert config_from_dict.scale == config.scale


def test_qtip_config_methods():
    """Test QTIP configuration methods."""
    config = QTIPConfig(td_x=8,
                        td_y=8,
                        L=16,
                        K=2,
                        V=2,
                        tlut_bits=16,
                        decode_mode="1mad",
                        scale=32.0)

    assert config.get_name() == "qtip"

    assert torch.half in config.get_supported_act_dtypes()

    assert config.get_min_capability() == 60

    assert "quantize_config.json" in config.get_config_filenames()


# needs cuda, currently using cpu
# @pytest.mark.parametrize(
#     "model",
#     [
#         "meta-llama/Llama-2-7b-hf",
#     ])
# def test_qtip_inference(vllm_runner, model, monkeypatch):
#     """Test inference with QTIP quantization."""

#     monkeypatch.setenv("VLLM_USE_V1", "0")

#     qtip_config = {
#         "quant_method": "qtip",
#         "td_x": 8,
#         "td_y": 8,
#         "L": 16,
#         "K": 2,
#         "V": 2,
#         "tlut_bits": 16,
#         "decode_mode": "1mad",
#         "scale": 32.0
#     }

#     with vllm_runner(model_name=model,
#                      quantization="qtip",
#                      enforce_eager=True,
#                      ) as llm:

#         model = llm.model.llm_engine.model_executor.driver_worker.
# model_runner.model
#         layer = model.model.layers[0]

#         assert isinstance(layer.self_attn.qkv_proj.quant_method,
#                           QTIPLinearMethod)

#         output = llm.generate_greedy("Hello my name is", max_tokens=20)
#         assert output
