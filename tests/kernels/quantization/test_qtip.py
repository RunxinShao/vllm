# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm import _custom_ops as ops

# 测试参数
HIDDEN_SIZES = [1024, 2048, 4096]
OUT_SIZES = [4096, 8192, 16384]
L_VALUES = [32, 64]
K_VALUES = [4, 8]
V_VALUES = [16, 32]

@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("out_size", OUT_SIZES)
@pytest.mark.parametrize("L", L_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("V", V_VALUES)
def test_bitshift_codebook(hidden_size, out_size, L, K, V):
    # 创建 codebook
    codebook = ops.bitshift_codebook(
        L=L,
        K=K,
        V=V,
        tlut_bits=L,  # 使用 L 作为 tlut_bits
        decode_mode="lut"  # 使用 lut 模式
    )
    
    # 准备输入
    weight = torch.rand((hidden_size, out_size), device='cuda', dtype=torch.float16)
    
    # 测试 codebook 功能
    # 构造一个简单的 encoded 索引
    encoded = torch.randint(0, 2**L, (hidden_size, out_size), device='cuda', dtype=torch.long)
    decoded = codebook.recons(encoded)
    
    # 检查输出形状
    assert decoded.shape == (V, hidden_size, out_size)
    # 检查数值范围
    assert torch.all(decoded >= -1.0) and torch.all(decoded <= 1.0)

@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("out_size", OUT_SIZES)
@pytest.mark.parametrize("L", L_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("V", V_VALUES)
def test_bitshift_gemm(hidden_size, out_size, L, K, V):
    # 创建 codebook
    codebook = ops.bitshift_codebook(
        L=L,
        K=K,
        V=V,
        tlut_bits=L,
        decode_mode="lut"
    )
    
    # 准备输入
    x = torch.rand((1, hidden_size), device='cuda', dtype=torch.float16)
    # 构造 trellis
    trellis = torch.randint(0, 2**L, (hidden_size, out_size), device='cuda', dtype=torch.long)
    
    # 测试矩阵乘法
    output = ops.bitshift_gemm(
        input=x,
        trellis=trellis,
        codebook=codebook,
        td_x=8,
        td_y=8,
        scale=32.0,
        SU=torch.ones(hidden_size, device='cuda', dtype=torch.float16),
        SV=torch.ones(out_size, device='cuda', dtype=torch.float32)
    )
    
    # 检查输出形状
    assert output.shape == (1, out_size)
    # 检查数值范围
    assert torch.all(torch.isfinite(output))

# 性能测试可以保留，但建议放在单独的文件中
def benchmark_bitshift():
    # 你的性能测试代码
    pass
