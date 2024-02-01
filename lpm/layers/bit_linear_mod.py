import math
import torch
import torch.nn as nn

from typing import Optional, Callable


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
        bias: Optional[bool] = True,
        initializer_weight: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        initializer_bias: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        eps: Optional[float] = 1.0e-5,
        params_dtype: Optional[torch.dtype] = torch.bfloat16,
        fp8_e4m3: Optional[bool] = False,
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
            If `False`, no bias is added.
            Default: `True`
        initializer_weight : Callable[[torch.Tensor], torch.Tensor], optional
            Function for custom weight initialization. If `None`, default are used.
            Default: `BitLinearMod._default_initializer_weight`
        initializer_bias: Callable[[torch.Tensor], torch.Tensor], optional
            Function for custom bias initialization. If `None`, default are used.
            Default: `BitLinearMod._default_initializer_bias`
        eps : float, optional
            Small value to prevent division by zero in normalization.
            Default: `1.0e-5`
        params_dtype : torch.dtype, optional
            Controls the type used to allocate the initial parameters.
            Default: `torch.bfloat16`
        fp8_e4m3 : bool, optional
            If `True`, the layer will use autocasting to the FP8 e4m3 format during calculations.
            Default: `False`

            Note: FP8 e4m3 format is not supported by all hardware. Requires `Transformer Engine`
            to be installed.

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
        self.params_dtype = params_dtype
        self.fp8_e4m3 = fp8_e4m3

        # Init weight.
        w = torch.empty(
            size=[in_features, out_features],
            dtype=self.params_dtype,
        ).contiguous()
        w = (
            BitLinearMod._default_initializer_weight(w)
            if initializer_weight is None
            else initializer_weight(w)
        )
        self.weight = nn.Parameter(w)

        # Init bias.
        if bias:
            b = torch.empty(
                size=[out_features],
                dtype=self.params_dtype,
            ).contiguous()
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
        # I've used sqrt(3.0) instead of sqrt(2.0) below, as this seems to
        # lead to a better preservation of input magnitudes.
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

    def _get_ctx(
        self,
        device_type: str,
    ) -> torch.autocast:
        # if hasattr(self, "autocast_ctx"):
        #     return self.autocast_ctx

        if self.fp8_e4m3:
            try:
                import transformer_engine as te

                fp8_e4m3 = te.common.recipe.DelayedScaling(
                    margin=0,
                    interval=1,
                    fp8_format=te.common.recipe.Format.E4M3,
                    amax_compute_algo="max",
                )
                self.autocast_ctx = te.pytorch.fp8_autocast(
                    enabled=True,
                    fp8_recipe=fp8_e4m3,
                )
            except ImportError:
                raise ImportError(
                    "fp8_e4m3 requires `transformer_engine` to be installed."
                )
        else:
            self.autocast_ctx = torch.autocast(
                dtype=self.params_dtype,
                device_type=device_type,
            )

        return self.autocast_ctx

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Cast x to params dtype.
        src_dtype = x.dtype
        x = x.to(dtype=self.params_dtype)

        with self._get_ctx(device_type=x.device.type):
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

        # Cast x from params dtype back to src dtype.
        x = x.to(dtype=src_dtype)

        return x
