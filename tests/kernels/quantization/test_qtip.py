# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm import _custom_ops as ops

HIDDEN_SIZES = [4096]
OUT_SIZES = [4096]
L_VALUES = [16]
K_VALUES = [3]
V_VALUES = [1]


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("out_size", OUT_SIZES)
@pytest.mark.parametrize("L", L_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("V", V_VALUES)
def test_bitshift_codebook(hidden_size, out_size, L, K, V):
    td_x, td_y = 8, 8
    block_size = td_x * td_y
    row_blocks = hidden_size // td_x
    col_blocks = out_size // td_y
    num_blocks = row_blocks * col_blocks

    codebook = ops.bitshift_codebook(L=L,
                                     K=K,
                                     V=V,
                                     tlut_bits=L,
                                     decode_mode="lut")

    encoded = torch.randint(0,
                            2**L, (num_blocks, block_size),
                            device='cpu',
                            dtype=torch.long)

    decoded = codebook.recons(encoded)

    assert decoded.shape == (V, num_blocks, block_size)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("out_size", OUT_SIZES)
@pytest.mark.parametrize("L", L_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("V", V_VALUES)
def test_bitshift_gemm(hidden_size, out_size, L, K, V):
    td_x = 8
    td_y = 8
    block_size = td_x * td_y

    m = hidden_size
    n = out_size
    row_blocks = m // td_x
    col_blocks = n // td_y
    num_blocks = row_blocks * col_blocks

    codebook = ops.bitshift_codebook(L=L,
                                     K=K,
                                     V=V,
                                     tlut_bits=L,
                                     decode_mode="lut")

    x = torch.rand((1, m), device='cpu', dtype=torch.float16)

    trellis = torch.randint(0,
                            2**L, (num_blocks, block_size),
                            device='cpu',
                            dtype=torch.long)

    SU = torch.ones(m, device='cpu', dtype=torch.float16)
    SV = torch.ones(n, device='cpu', dtype=torch.float16)

    output = ops.bitshift_gemm(input=x,
                               trellis=trellis,
                               codebook=codebook,
                               td_x=td_x,
                               td_y=td_y,
                               scale=32.0,
                               SU=SU,
                               SV=SV)

    assert output.shape == (1, n)
