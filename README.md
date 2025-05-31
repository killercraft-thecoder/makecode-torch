# Torch - Neural Network Library for MakeCode Arcade

Torch is a lightweight **neural network library** designed for use within **MakeCode Arcade**. It provides essential **tensor operations, basic neurons, and layer-based training** to build small-scale neural networks.

## 🚀 Features
- **Tensor Operations** → Supports matrix multiplication, element-wise function applications, and tensor addition.
- **Neuron & Linear Layers** → Implements simple neuron models and fully connected layers.
- **Convolutional Layer** → Enables basic convolution operations for feature extraction.
- **Training Support** → Uses **Mean Squared Error (MSE)** loss and backpropagation for weight updates.
- **Activation Functions** → Includes **ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax, ELU, Swish, GELU, Hard Sigmoid, Bent Identity, and Mish** for non-linear transformations.

📌 [See Full Version History](https://github.com/killercraft-thecoder/makecode-torch/blob/master/history.md)
📌 [See How to Update Models](https://github.com/killercraft-thecoder/makecode-torch/blob/master/updating.md)
## 🔗 Installation & Usage

### ✅ **Use as an Extension**
This repository can be added as an **extension** in MakeCode Arcade.

1. Open [MakeCode Arcade](https://arcade.makecode.com/)
2. Click **New Project**
3. Open **Extensions** under the gearwheel menu
4. Search for **`https://github.com/killercraft-thecoder/makecode-torch`** and import

### ✏ **Edit this Project**
To edit this repository directly within MakeCode:

1. Open [MakeCode Arcade](https://arcade.makecode.com/)
2. Click **Import**, then **Import URL**
3. Paste **`https://github.com/killercraft-thecoder/makecode-torch`** and click **Import**

## 🌍 GitHub Repository
> You can explore the full project here:  
> [https://killercraft-thecoder.github.io/makecode-torch/](https://killercraft-thecoder.github.io/makecode-torch/)

## 🔄 PyTorch Model Conversion (`convert.py`)

Torch now includes a **conversion tool** that allows users to **convert saved PyTorch models** into **MakeCode Arcade-compatible TypeScript**.  

### **Usage**
1. **Run `convert.py`** and provide the path to your saved PyTorch model (`.pth` file).
2. The script will **extract weights** and **convert layers** into a structured TypeScript format.
3. The **generated file** can be imported into your MakeCode Arcade project!

Example Command:
```sh
python convert.py
```

### 🔹 GitHub Syncing Tip for MakeCode Arcade

Did you know? **MakeCode Arcade only pulls changes affecting managed project files**, but ignores externally added files like `history.md`.  
This means you can:
- Store **documentation or version history** externally without affecting the MakeCode project.
- Keep **development notes or testing scripts** only in GitHub.
- Maintain **extra resources** without cluttering the MakeCode editor.
- Also Rember makecode arcade only pulls files in the files part of pxt.json!

If you need to **make non-code updates**, doing them externally can keep the project cleaner while still benefiting from GitHub's commit tracking!

