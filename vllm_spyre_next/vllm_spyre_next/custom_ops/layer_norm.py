# SPDX-License-Identifier: Apache-2.0

import torch
from functools import lru_cache

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)

_SPYRE_MIN_BATCH_SIZE = 64


class SpyreLayerNorm:
    """
    Spyre implementation of LayerNorm.

    Standard LayerNorm using mean + variance normalization.
    """

    def __init__(self):
        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        self._layer_name = register_layer(self, "spyre_layernorm")

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        eps: float,
        hidden_size: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        """
        Core LayerNorm math.
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        x_norm = (x - mean) * torch.rsqrt(variance + eps)

        if weight is not None:
            x_norm = x_norm * weight

        if bias is not None:
            x_norm = x_norm + bias

        return x_norm

    def _forward_spyre_impl(
        self,
        x: torch.Tensor,
        eps: float,
        hidden_size: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        x_dtype = x.dtype
        x_device = x.device

        orig_batch_size = x.shape[0]

        # Pad small batch (Spyre requirement)
        if x.shape[0] < _SPYRE_MIN_BATCH_SIZE:
            pad = _SPYRE_MIN_BATCH_SIZE - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))

        out = self.forward_spyre(
            convert(x, self._target_device, self._target_dtype),
            eps,
            hidden_size,
            convert(weight, self._target_device, self._target_dtype),
            convert(bias, self._target_device, self._target_dtype)
            if bias is not None
            else None,
        )

        return convert(out, dtype=x_dtype, device=x_device)[:orig_batch_size]


def _op_func(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Spyre backend entry point.
    """
    layer = get_layer(layer_name)

    result = layer._forward_spyre_impl(
        x,
        layer.eps,
        layer.dim,
        layer.weight,
        layer.bias,
    )

    output.copy_(result)


@lru_cache(maxsize=1)
def register():
    """
    Register Spyre LayerNorm custom op.
    """
    direct_register_custom_op(
        op_name="spyre_layernorm",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )

    logger.info("Registered custom op: SpyreLayerNorm")
