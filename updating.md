# 🔄 Updating Torch Models Between Versions

## Overview
This guide explains how to **update models when new versions of Torch are released**, ensuring compatibility without modifying the extension itself.

---

## **Updating Steps: 1.0.0 → 1.7.0**  

### **🛠 How to Adapt Your Code for Each Version**
| **Version** | **Key Changes** | **Action Required** |
|------------|----------------|---------------------|
| **1.1.x** | Added ConvLayer support | Use `Torch.ConvLayer(...)` for CNN-based models |
| **1.2.x** | Expanded activation functions | Update `Torch.ReLU` to `Torch.relu`, `Torch.Sigmoid` to `Torch.sigmoid` |
| **1.3.x** | Optimized `matmul()` | Ensure matrix multiplication usage does not use old features/bugs |
| **1.4.x** | Improved memory handling | None Needed unless the memory allocation used is critcal |
| **1.5.x** | Introduced custom loss function support | Pass `Torch.mse`, `Torch.huber`, etc., in `train()` |
| **1.6.x** | Added expanded error functions | Update models to use `Torch.mce`, `Torch.rmse`, etc. |
| **1.7.x** | Silent Mode for training | Use `train(..., silent: true)` to disable logs |

---

## **🔹 Updating Model Training Code**
### **Example: Updating a Model from 1.5.x → 1.7.x**
**Old Code (1.5.x)**
```typescript
model.train(inputs, targets, learningRate, epochs, Torch.relu, Torch.mse);
console.log("Training complete!");
```

**Updated code (1.7.x)**

```typescript
model.train(inputs, targets, learningRate, epochs, Torch.relu, Torch.mse, true); // Silent Mode enabled
console.log("Training complete!");
```

Why?

silent:true prevents logging output during training

the API remains the same ensurng a smooth transition

## ** Example 2: Updating a model from 1.0.x → 1.7.x**

**Old code (1.0.x)**
```typescript
model.train(inputs,targets,learningRate,epochs,Torch.ReLU); // Train
console.log("Training Done!");
```

**Updated code (1.7.x)**
```typescript
model.train(inputs,targets,learningRate,epochs,Torch.relu); // Train , Small diffrence becuase of rename of ReLU -> relu
console.log("Training Done!");
```
