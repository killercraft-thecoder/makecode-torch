 📌 Looking for the README? [Click here](https://github.com/killercraft-thecoder/makecode-torch/blob/master/README.md)

# 🔄 Torch - Version History

## **1.7.0** - Performance & Training Enhancements  
- **Optimized `matmul()` in Torch.Tensor** → Improved efficiency and memory allocation for MakeCode Arcade.  
- **Added Silent Mode (`silent` parameter) to `train()`** → Allows training without console logs when set to `true`.  
- **Improved memory handling** → Proper array allocation fixes to prevent unintended modifications.  
- **Refined matrix operations** → Ensured independent row duplication for stable calculations.  

### 🔹 **Important Notes**  
- `matmul()` now runs **faster and more efficiently**, reducing memory overhead in matrix-heavy computations.  
- Silent Mode (`silent: true`) prevents unnecessary logging in training, making it useful for **background executions**.  

## **1.6.0** - Expanded Error Function Support  
- **Added new error functions** to enhance training flexibility:  
  - **MSE (Mean Squared Error)** → Standard squared error computation.  
  - **MCE (Mean Cubed Error)** → Amplifies larger errors for stronger adjustments.  
  - **ME (Mean Error, X = X)** → Direct propagation of error without modification.  
  - **RMSE (Root Mean Squared Error)** → Square root-based loss for balanced error scaling.  
  - **Huber Loss (Delta = 1)** → Stabilizes training by reducing sensitivity to outliers.  

### 🔹 **Important Notes**  
- Users can now choose error functions when training **Torch.Sequential** and **Torch.Linear** models.  
- Default remains **MSE**, but experiments with **Huber, RMSE, and MCE** may improve stability in different scenarios.  
- This update enhances model adaptability for various optimization strategies.  

## **1.5.0** - Custom Loss Function Support  
- **Added support for custom loss functions** in both `Torch.Sequential` and `Torch.Linear`.  
  - Users can now pass a **lossFunction** parameter to `train()` to customize error calculations.  
  - Default remains **Mean Squared Error (MSE)** if no loss function is provided.  
  - `backward()` now accepts **lossFunction** for greater flexibility in weight updates.  
- **Refactored training logic** to improve modularity and maintainability.  
- **Version bump from 1.4.2 to 1.5.0**, marking a significant feature addition.  

### 🔹 **Important Notes**  
- Recommended to experiment with different loss functions (**MAE, Huber, Cross-Entropy**) for various training scenarios.  
- Users upgrading from **1.4.x** should ensure compatibility with any custom training workflows.  

## **1.4.1** - Performance Optimization Update  
- **Softmax optimization** → Improved numerical stability by removing unsupported MakeCode Arcade features.  
- **MAE function refactor** → Reduced loop overhead for faster computation.  
- **General activation function improvements** → Optimized Sigmoid, Gelu, and other functions to avoid redundant calculations.  

### 🔹 **Important Notes**  
- Performance improvements should be noticeable in activation-heavy models.   

## **1.4.0** - Optimized Matrix Multiplication  
- **Optimized `matmul()` in Torch.Tensor** to improve memory allocation and speed in MakeCode Arcade.  

### 🔹 **Important Notes**  
- Older versions may experience slower tensor operations due to inefficient processing.  
- Recommended for users handling **large-scale matrix computations**.  

## **1.3.0** - Expanded Activation Functions  
- Added new activation functions: **Hard Sigmoid, Bent Identity, and Mish**.  
- Updated README to reflect the expanded activation function list.  

### 🔹 **Important Notes**  
- Mish and Bent Identity are useful for **non-linear transformations**.  
- This version expands activation flexibility, making it easier to fine-tune models.  

## **1.2.0** - Added New Activation Functions  
- Added **ELU**, **Swish**, and **GELU** activation functions.  
- Updated README to reflect the full set of available activation functions.  

### 🔹 **Important Notes**  
- ELU and Swish offer smoother activation behavior in deep models.  
- GELU can improve learning dynamics, especially in **transformer-based architectures**.  

## **1.1.2** - Improved Debugging  
- Added error logging in `test.ts` to verify predictions.  

### 🔹 **Important Notes**  
- Debugging logs provide deeper insights into **prediction errors**.  

## **1.1.1** - Tensor Operations Fix  
- Improved **Tensor.add()** to match safety checks from `sub()`.  

### 🔹 **Important Notes**  
- Fixed a subtle issue where tensor addition lacked error safety mechanisms.  
- Recommended for users working with **large-scale tensor arithmetic**.  

## **1.1.0** - Full ConvLayer Integration  
- Added missing **forward()**, **backward()**, and **train()** methods to `ConvLayer`.  
- Updated `Sequential` to properly handle convolutional layers.  

### 🔹 **Important Notes**  
- Torch now **fully supports convolutional layers**, allowing CNN-based models.  
- Users working with **image processing tasks** should upgrade to benefit from ConvLayer optimizations.  

## **1.0.0** - Initial Release  
- Basic neural network structure with **Linear layers, Tensors, and Training support**.  

### ℹ️ **Important Note**  
- **`convert.py` was introduced via a direct GitHub commit** and exists **outside the standard versioning system**.  
- Users **are not required to be on version 1.4.0** to utilize PyTorch model conversion. However, for full compatibility with `convert.py`, a **Torch version greater than 1.1.0** is recommended.  
- To ensure access to the latest features, performance improvements, and full compatibility, **it is highly recommended to use the most recent version of Torch**.  
