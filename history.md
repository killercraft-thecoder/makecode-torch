📌 Looking for the README? [Click here](https://github.com/killercraft-thecoder/makecode-torch/blob/master/README.md)

# 🔄 Torch - Version History

### **ℹ️ Note**
- **`convert.py` was added via direct GitHub commit** and is available **outside the versioning system**.
- Users **do not need 1.4.0** to use PyTorch model conversion, for Full convert.py support a version of > 1.1.0 is needed

## **1.4.0** - Optimized Matrix Multiplication + PyTorch Model Conversion
- **Optimized `matmul()` in Torch.Tensor** to improve memory allocation and speed in MakeCode Arcade.

## **1.3.0** - Expanded Activation Functions
- Added new activation functions: **Hard Sigmoid, Bent Identity, and Mish**.
- Updated README to reflect the expanded activation function list.

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
