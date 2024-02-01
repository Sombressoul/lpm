# LPM - Low-Precision Modules

## Overview

The `LPM` (Low Precision Modules) is a collection of PyTorch modules that utilize low-precision calculations and/or low-precision/quantized weights.

Some of the presented modules are implementations of specific scientific publications, while others are my own developments or an extensions of ideas from the mentioned scientific works.

## Modules

1. __BitLinear__: An implementation of the "BitNet" paper ([arXiv:2310.11453](https://arxiv.org/pdf/2310.11453.pdf)).
2. __BitLinearMod__: A modified version of the BitLinear layer from the "BitNet" paper. Simplified math with bias support. Also supports FP8 (E4M3) autocasting (requires [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) to be installed).
3. (WIP) __BitConv2D__: Extension of ideas from the "BitNet" paper, applied to two-dimensional convolutional layers.

## Known Issues

- Process of building a wheel for the Flash-Attention V2 during the building of the Transformer Engine (if you decided to use install_dev.bash) could require a lot of RAM (~97 Gb in my case without workaround with MAX_JOBS=1; with MAX_JOBS=1 requires ~12 Gb). It is a [well known issue](https://github.com/Dao-AILab/flash-attention/issues/358). Also, be patient, because this is quite a long process.

## Contributing

Contributions to the `LPM` project are welcome. Please submit a pull request or open an issue if you have suggestions or improvements.

## License

`LPM` is released under the MIT License. See the `LICENSE` file for more details.
