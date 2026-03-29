"""MLP modules: SwiGLU (for Mamba2/attention blocks) and RWKV CMix (relu²)."""

import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feedforward network (Shazeer 2020).

    Used in uppercase block types ('M', 'T', 'R') alongside their mixer.

    Args:
        d_model: Input/output dimension.
        d_intermediate: Hidden dimension. If None, defaults to 8/3 * d_model
            rounded up to nearest multiple_of.
        multiple_of: Alignment for hidden dimension.
    """

    def __init__(
        self,
        d_model: int,
        d_intermediate: int = None,
        bias: bool = False,
        multiple_of: int = 128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if d_intermediate is None:
            d_intermediate = int(8 * d_model / 3)
        d_intermediate = (
            (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
        )
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = F.silu(gate) * y
        return self.fc2(y)
