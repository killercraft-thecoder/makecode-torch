# 🔄 Torch - Version History

## **1.2.0** - Added New Activation Functions
- Added **ELU**, **Swish**, and **GELU** activation functions.
- Updated README to reflect the full set of available activation functions.

## **1.1.2** - Improved Debugging
- Added error logging in `test.ts` to verify predictions.

## **1.1.1** - Tensor Operations Fix
- Improved **Tensor.add()** to match safety checks from `sub()`.

## **1.1.0** - Full ConvLayer Integration
- Added missing **forward()**, **backward()**, and **train()** methods to `ConvLayer`.
- Updated `Sequential` to properly handle convolutional layers.

## **1.0.0** - Initial Release
- Basic neural network structure with **Linear layers, Tensors, and Training support**.
