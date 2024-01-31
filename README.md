# LPM - Low-Precision Modules

## Overview

The `LPM` (Low Precision Modules) is a collection of PyTorch modules that utilize low-precision calculations and/or low-precision/quantized weights.

Some of the presented modules are implementations of specific scientific publications, while others are my own developments or an extensions of ideas from the mentioned scientific works.

## Modules

1. __BitLinear__: An implementation of the "BitNet" paper ([arXiv:2310.11453](https://arxiv.org/pdf/2310.11453.pdf)).
2. __BitLinearMod__: A modified version of the BitLinear layer from the "BitNet" paper.
3. (WIP) __BitConv2D__: My extension of ideas from the "BitNet" paper, applied to two-dimensional convolutional layers.

## Contributing

Contributions to the `LPM` project are welcome. Please submit a pull request or open an issue if you have suggestions or improvements.

## License

`LPM` is released under the MIT License. See the `LICENSE` file for more details.
