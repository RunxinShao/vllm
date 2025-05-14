import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing import Optional
import numpy as np

from vllm._custom_ops import bitshift_codebook, bitshift_gemm
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import PackedvLLMParameter


class QTIPConfig(QuantizationConfig):
    """
    QTIP (Quantization with Trellis Index Packing) static quantization configuration class
    """
    def __init__(
        self,
        td_x: int,
        td_y: int,
        L: int,
        K: int,
        V: int,
        tlut_bits: int,
        decode_mode: str,
        scale: float = 32.0
    ):
        self.td_x = td_x
        self.td_y = td_y
        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode
        self.scale = scale
        # Number of elements in each block
        self.pack_factor = td_x * td_y

    def __repr__(self) -> str:
        return (
            f"QTIPConfig(td_x={self.td_x}, td_y={self.td_y},"
            f" L={self.L}, K={self.K}, V={self.V},"
            f" tlut_bits={self.tlut_bits}, decode_mode='{self.decode_mode}',"
            f" scale={self.scale})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "qtip"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict) -> "QTIPConfig":
        td_x = cls.get_from_keys(config, ["td_x"])
        td_y = cls.get_from_keys(config, ["td_y"])
        L = cls.get_from_keys(config, ["L"])
        K = cls.get_from_keys(config, ["K"])
        V = cls.get_from_keys(config, ["V"])
        tlut_bits = cls.get_from_keys(config, ["tlut_bits"])
        decode_mode = cls.get_from_keys(config, ["decode_mode"])
        scale = cls.get_from_keys_or(config, ["scale"], default=32.0)
        return cls(td_x, td_y, L, K, V, tlut_bits, decode_mode, scale)

    def get_quant_method(self, layer: nn.Module, prefix: str) -> "QTIPLinearMethod":
        return QTIPLinearMethod(self)


class QTIPLinearMethod(LinearMethodBase):
    """
    QTIP linear layer quantization method
    """
    def __init__(self, quant_config: QTIPConfig):
        self.cfg = quant_config
        # Build lookup table (codebook)
        self.cb = bitshift_codebook(
            L=self.cfg.L,
            K=self.cfg.K,
            V=self.cfg.V,
            tlut_bits=self.cfg.tlut_bits,
            decode_mode=self.cfg.decode_mode
        )
        self.scale = self.cfg.scale

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):
        output_size_per_partition = sum(output_partition_sizes)
        pack_factor = self.cfg.pack_factor
        assert input_size_per_partition % pack_factor == 0, \
            "input size must be multiple of pack_factor"
        rows = input_size_per_partition // pack_factor

        qweight = PackedvLLMParameter(
            data=torch.empty(rows, output_size_per_partition, dtype=torch.int32),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=pack_factor,
            weight_loader=lambda p, w: p.data.copy_(w)
        )
        layer.register_parameter("qweight", qweight)

        # Register SU and SV
        layer.register_buffer("SU", torch.ones(input_size_per_partition, dtype=params_dtype))
        layer.register_buffer("SV", torch.ones(output_size_per_partition, dtype=torch.float32))

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """
        Unpack loaded quantized indices and restore to int32 index matrix
        """
        packed = layer.qweight.data
        unpacked = self.cb.unpack_trellis(packed, self.cfg.pack_factor)
        layer.qweight.data = unpacked.to(torch.int32)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = bitshift_gemm(
            input=x,
            trellis=layer.qweight,
            codebook=self.cb,
            td_x=self.cfg.td_x,
            td_y=self.cfg.td_y,
            scale=self.scale,
            SU=layer.SU,
            SV=layer.SV
        )
        if bias is not None:
            output = output + bias
        return output
