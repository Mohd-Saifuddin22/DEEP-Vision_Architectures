# Deep CNN Training: MiniVGG & ResNet on CIFAR-10 (CUDA Enabled)

---

## 📌 Project Objective

Build and compare three convolutional architectures on the CIFAR-10 dataset using PyTorch and CUDA:

1. ✅ **MiniVGG** – Baseline CNN architecture
2. ✅ **MiniVGG Optimized** – Enhanced with BatchNorm & Learning Rate Scheduler
3. ✅ **MiniResNet** – Lightweight ResNet-style model with residual connections

---

## 📊 Dataset Details

- **Dataset**: CIFAR-10
- **Images**: 60,000 color images (32×32)
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Classes**: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

---

## ⚙️ Model Architectures

### 🔹 MiniVGG (Baseline)

- 2 Conv layers → MaxPool  
- 2 Conv layers → MaxPool  
- Flatten → Dense → Dropout → Output

🔧 No BatchNorm or Scheduler  
🧪 Achieved: **77.36%** Test Accuracy

---

### 🔹 MiniVGG Optimized

- Same layout as baseline
- ✅ BatchNorm after each Conv layer
- ✅ StepLR Scheduler (step=10, γ=0.5)

📈 **Improved Accuracy**: **79.36%**  
📉 Lower Training Loss  
🚀 Stable over 20 epochs  
🕐 Training Time (20 epochs): **5.98 min**

---

### 🔹 MiniResNet (Custom ResNet-Style)

- Conv → ResBlock1 → MaxPool  
- ResBlock2 → MaxPool  
- AdaptiveAvgPool → Flatten → FC  
- Residual Blocks with Conv-BN-ReLU layers

✅ Early Stopping  
✅ ReduceLROnPlateau Scheduler  
✅ Data Augmentation  
📈 Best Validation Accuracy: **77.96%**  
🧪 Final Test Accuracy: **78.05%**

---

## 🧪 Performance Summary

| Model              | Epochs | Scheduler        | BatchNorm | Final Val Acc | Test Acc | Training Time |
|-------------------|--------|------------------|-----------|---------------|----------|----------------|
| MiniVGG (Baseline)| 10     | ❌ None           | ❌ No     | 77.64%        | **77.36%** | 2.84 min (pin) |
| MiniVGG Optimized | 20     | ✅ StepLR         | ✅ Yes    | 79.36%        | **79.36%** | 5.98 min       |
| MiniResNet        | 20     | ✅ ReduceLROnPlateau | ✅ Yes | 77.96%        | **78.05%** | ~6–7 min (est.) |

---

## 🔧 Hyperparameters Used

| Parameter       | MiniVGG     | MiniVGG Optimized | MiniResNet |
|----------------|-------------|-------------------|-------------|
| Batch Size     | 64          | 64                | 64          |
| Learning Rate  | 0.001       | 0.001             | 0.001       |
| Optimizer      | Adam        | Adam              | Adam        |
| Epochs         | 10          | 20                | 20          |
| Scheduler      | ❌ None     | ✅ StepLR         | ✅ ReduceLROnPlateau |
| Early Stopping | ❌ No       | ❌ No             | ✅ Yes       |
| Pin Memory     | ✅ Yes      | ✅ Yes            | ✅ Yes       |
| Augmentation   | ❌ No       | ❌ No             | ✅ RandomCrop + Flip |

---

## 🖼 Model Architectures (Layer-wise)

### 🧱 MiniVGG Optimized

- Conv(3→32) → BN → ReLU  
- Conv(32→32) → BN → ReLU → MaxPool  
- Conv(32→64) → BN → ReLU  
- Conv(64→64) → BN → ReLU → MaxPool  
- FC(64×8×8 → 512) → Dropout(0.5) → FC(10)

---

### 🧱 MiniResNet

- Conv(3→32) → BN → ReLU  
- Residual Block: [Conv→BN→ReLU→Conv→BN + shortcut] → ReLU  
- MaxPool  
- Residual Block: [Conv→BN→ReLU→Conv→BN + shortcut] → ReLU  
- MaxPool  
- AdaptiveAvgPool → Flatten → FC(64→10)

---

## 🧠 Key Improvements

- ✅ **Batch Normalization** stabilized training and reduced overfitting
- ✅ **Schedulers** helped fine-tune learning rate dynamically
- ✅ **Early Stopping** prevented overfitting in ResNet
- ✅ **pin_memory=True** enabled faster data transfer to GPU
- ✅ **Residual Blocks** improved gradient flow and boosted accuracy

---

## 💾 Files Included

| File                        | Description                               |
|-----------------------------|-------------------------------------------|
| `MINI_VGG.ipynb`            | Baseline MiniVGG architecture             |
| `Mini_VGG optimized.ipynb`  | Optimized MiniVGG with BatchNorm + LR Scheduler |
| `Mini_ResNet.ipynb`         | Custom lightweight ResNet implementation |
| `README.md`                 | Documentation and performance summary     |

---

## 🧪 Final Test Results (Best Accuracy)

| Model            | Test Accuracy |
|------------------|---------------|
| MiniVGG          | 77.36%        |
| MiniVGG Optimized| **79.36%** ✅ |
| MiniResNet       | 78.05%        |

---

## 👤 Author

**Mohd Saifuddin**  
📧 mohdsaifuddin22@gmail.com  
🐙 [GitHub: @Mohd-Saifuddin22](https://github.com/Mohd-Saifuddin22)

---

⭐ **If you found this helpful, star the repo! Contributions welcome.**
