import math
import torch
import torch.nn as nn

from collections.abc import Callable


class BitLinearMod(nn.Module):
    """
    BitLinearMod: modified version of BitLinear (arXiv:2310.11453).

    The BitLinearMod class implements a 1-bit linear layer, inspired by the BitNet
    approach, designed for quantized neural networks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initializer_weight: Callable[[torch.Tensor], torch.Tensor] = None,
        initializer_bias: Callable[[torch.Tensor], torch.Tensor] = None,
        eps: float = 1.0e-5,
    ) -> None:
        """
        Instantiates a `BitLinearMod` layer.

        Parameters
        ----------
        in_features : int
            Input size per sample.
        out_features : int
            Output size per sample.
        bias : bool, optional
            If `False`, no bias is added. Default: `True`.
        initializer_weight : Callable[[torch.Tensor], torch.Tensor], optional
            Function for custom weight initialization. If `None`, default are used.
        initializer_bias: Callable[[torch.Tensor], torch.Tensor], optional
            Function for custom bias initialization. If `None`, default are used.
        eps : float, optional
            Small value to prevent division by zero in normalization. Default: 1.0e-5.

        Notes
        --------
        A modified version of BitLinear. The modifications are based on my experiments
        and are intended to improve the efficiency and performance of the layer.

        Changes:
        - Removed input normalization.
        - Removed input x-x.min() scaling (proposed in the paper for layers
        prior to nonlinear ReLU-like activations).
        - Quantization base assumed as 1-bit.
        - Added bias to initial symmetry break.

        Examples
        --------
        >>> bl = BitLinearMod(20, 30)
        >>> x = torch.randn(128, 20)
        >>> output = bl(x)
        """
        super(BitLinearMod, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # Init weight.
        w = torch.empty(in_features, out_features).contiguous()
        w = (
            BitLinearMod._default_initializer_weight(w)
            if initializer_weight is None
            else initializer_weight(w)
        )
        self.weight = nn.Parameter(w)

        # Init bias.
        if bias:
            b = torch.empty(out_features).contiguous()
            b = (
                BitLinearMod._default_initializer_bias(b)
                if initializer_bias is None
                else initializer_bias(b)
            )
            self.bias = nn.Parameter(b)
        else:
            self.register_parameter("bias", None)

        pass

    @staticmethod
    def _default_initializer_weight(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # He et al. (2015): arXiv:1502.01852
        # I've used sqrt(3.0) instead of sqrt(2.0) below, since it leads
        # to a better preservation of input magnitudes.
        fan = torch.nn.init._calculate_correct_fan(tensor, mode="fan_in")
        bound = math.sqrt(3.0 / fan)
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    @staticmethod
    def _default_initializer_bias(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # Slight break of symmetry.
        with torch.no_grad():
            return tensor.normal_(mean=0.0, std=1.0e-4)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure input is contiguous.
        x = x.contiguous()

        # Centralize and quantize weights.
        w = torch.sign(self.weight - self.weight.mean())
        w = self.weight + (w - self.weight).detach()  # STE

        # Absmax quantization of the input.
        gamma = torch.max(torch.abs(x), dim=1, keepdim=True).values
        x = x / (gamma + self.eps)

        # Multiply quantized x and W, and scale the result back.
        scale = torch.max(torch.abs(self.weight)) * gamma
        x = x @ w
        x.mul_(scale)

        # Add bias if it exists.
        if self.bias is not None:
            x += self.bias

        return x
