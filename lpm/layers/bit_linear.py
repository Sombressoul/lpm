import math
import torch
import torch.nn as nn

from collections.abc import Callable


# "BitNet: Scaling 1-bit Transformers for Large Language Models"
# 2023, arXiv:2310.11453
# URL: https://arxiv.org/pdf/2310.11453.pdf
class BitLinear(nn.Module):
    """
    BitNet: Scaling 1-bit Transformers for Large Language Models, 2023 (arXiv:2310.11453)

    The BitLinear class implements a 1-bit linear layer, inspired by the BitNet
    approach, designed for quantized neural networks. It features customizable
    initializers, and precision control through quantization base.
    The pre_act flag adjusts quantization for layers followed by activations,
    while input_norm enables input normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer_weight: Callable[[torch.Tensor], torch.Tensor] = None,
        eps: float = 1.0e-5,
        quant_base: int = 1,
        pre_act: bool = False,
        input_norm: bool = False,
    ) -> None:
        """
        Instantiates a `BitLinear` layer.

        Parameters
        ----------
        in_features : int
            Input size per sample.
        out_features : int
            Output size per sample.
        initializer_weight : Callable[[torch.Tensor], torch.Tensor], optional
            Function for custom weight initialization. If `None`, default are used.
        eps : float, optional
            Small value to prevent division by zero in normalization. Default: 1.0e-5.
        quant_base : int, optional
            Determines quantization precision. Default: 1.
        pre_act : bool, optional
            If `True`, adjusts quantization for subsequent activation. Default: `False`.
        input_norm : bool, optional
            If `True`, applies normalization to inputs. Default: `False`.

        Notes
        --------
        The `quant_base` parameter determines the quantization precision. It is used
        to control the quantization level of the layer.

        - The default value differs from the one specified in the paper: 1 instead of 8.

        The `pre_act` parameter is used to adjust quantization for subsequent
        activation. This is useful when the layer is followed by an activation
        function, such as ReLU.

        - Disabled by default, since observed performance degradation (probably
        due to used weights, or used architecture of test network, or because 
        of my crooked hands).

        The `input_norm` parameter is used to apply normalization to he input.
        This is useful when the input is expected to have a larger range of values,
        such as in image processing. By default, the input is not normalized,
        but if `input_norm` is set to `True`, the input is normalized.

        - Disabled by default, since observed performance degradation (probably
        due to used weights, or used architecture of test network, or because 
        of my crooked hands).
        - Besides, intermediate normalization is the task of the architecture
        designer, not of a specific module.

        Examples
        --------
        >>> bl = BitLinear(20, 30, pre_act=True)
        >>> x = torch.randn(128, 20)
        >>> output = bl(x)
        """
        super(BitLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.quant_base = quant_base
        self.pre_act = pre_act
        self.input_norm = input_norm

        # Init weight.
        w = torch.empty(in_features, out_features)
        w = (
            BitLinear._default_initializer_weight(w)
            if initializer_weight is None
            else initializer_weight(w)
        )
        self.weight = nn.Parameter(w)

        pass

    @staticmethod
    def _default_initializer_weight(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # He et al. (2015): arXiv:1502.01852
        fan = torch.nn.init._calculate_correct_fan(tensor, mode="fan_in")
        # I've used sqrt(3.0) instead of sqrt(2.0) below, since it leads
        # to a better preservation of input magnitudes.
        bound = math.sqrt(3.0 / fan)
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Straight-Through Estimator
        ste = lambda x, y: y + (x - y).detach()

        # Centralize and quantize weights.
        a = self.weight.mean()
        w_q = torch.sign(self.weight - a)
        w_q = ste(w_q, self.weight)  # STE: bypass signum

        # Lei Ba et al. (2016): arXiv:1607.06450
        if self.input_norm:
            x_mean = x.mean(dim=1, keepdim=True)
            x_var = x.var(dim=1, keepdim=True)
            x = (x - x_mean) / torch.sqrt(x_var + self.eps)

        # Absmax quantization of the input.
        gamma = torch.max(torch.abs(x), dim=1, keepdim=True).values
        Qb = 2 ** (self.quant_base - 1)

        if self.pre_act:
            x_min = torch.min(x, dim=1, keepdim=True).values
            x_norm = (x - x_min) * (Qb / (gamma + self.eps))
            x_q = torch.clip(x_norm, self.eps, Qb - self.eps)
        else:
            x_norm = x * (Qb / (gamma + self.eps))
            x_q = torch.clip(x_norm, -Qb + self.eps, Qb - self.eps)

        # Get dequantize scale.
        beta = torch.max(torch.abs(self.weight))
        scale = (beta * gamma) / Qb

        # Multiply quantized x and W and scale the result back.
        x = (x_q @ w_q) * scale

        return x
