# SPDX-License-Identifier: Apache-2.0
"""Tests for QTIP quantization configuration.

Run `pytest tests/quantization/test_qtip_config.py`.
"""
from typing import Any, Optional

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods, get_quantization_config)
from vllm.model_executor.layers.quantization.qtip import (
    QTIPConfig, QTIPLinearMethod)


def test_qtip_config_creation():
    """Test QTIP configuration creation and validation."""
    # 测试基本配置创建
    config = QTIPConfig(
        td_x=8,
        td_y=8,
        L=16,
        K=2,
        V=2,
        tlut_bits=16,
        decode_mode="1mad",
        scale=32.0
    )
    
    # 验证配置属性
    assert config.td_x == 8
    assert config.td_y == 8
    assert config.L == 16
    assert config.K == 2
    assert config.V == 2
    assert config.tlut_bits == 16
    assert config.decode_mode == "1mad"
    assert config.scale == 32.0
    assert config.pack_factor == 64  # td_x * td_y

    # 测试从字典创建配置
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
    config = QTIPConfig(
        td_x=8,
        td_y=8,
        L=16,
        K=2,
        V=2,
        tlut_bits=16,
        decode_mode="1mad",
        scale=32.0
    )
    
    # 测试get_name方法
    assert config.get_name() == "qtip"
    
    # 测试get_supported_act_dtypes方法
    assert torch.half in config.get_supported_act_dtypes()
    
    # 测试get_min_capability方法
    assert config.get_min_capability() == 60
    
    # 测试get_config_filenames方法
    assert "quantize_config.json" in config.get_config_filenames()


def test_qtip_linear_method():
    """Test QTIP linear method creation and application."""
    config = QTIPConfig(
        td_x=8,
        td_y=8,
        L=16,
        K=2,
        V=2,
        tlut_bits=16,
        decode_mode="1mad",
        scale=32.0
    )
    
    # 测试量化方法创建
    quant_method = config.get_quant_method(None, "")
    assert isinstance(quant_method, QTIPLinearMethod)
    
    # 测试非LinearBase层返回None
    non_linear_layer = torch.nn.Linear(10, 10)
    assert config.get_quant_method(non_linear_layer, "") is None


@pytest.mark.parametrize("model", [
    "meta-llama/Llama-2-7b-hf",  # 使用一个较小的模型进行测试
])
def test_qtip_inference(vllm_runner, model, monkeypatch):
    """Test inference with QTIP quantization."""
    # 设置环境变量
    monkeypatch.setenv("VLLM_USE_V1", "0")
    
    # 创建QTIP配置
    qtip_config = {
        "quant_method": "qtip",
        "td_x": 8,
        "td_y": 8,
        "L": 16,
        "K": 2,
        "V": 2,
        "tlut_bits": 16,
        "decode_mode": "1mad",
        "scale": 32.0
    }
    
    # 运行推理测试
    with vllm_runner(
        model_name=model,
        quantization="qtip",
        enforce_eager=True,
        quantize_config=qtip_config
    ) as llm:
        # 获取模型
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model
        layer = model.model.layers[0]
        
        # 检查量化方法
        assert isinstance(layer.self_attn.qkv_proj.quant_method, QTIPLinearMethod)
        
        # 测试生成
        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output
