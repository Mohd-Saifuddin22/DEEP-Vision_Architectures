## Deep CNN Training: MiniVGG on CIFAR-10 (CUDA Enabled)

---

## 📌 Task Objective

Train a deeper Convolutional Neural Network (CNN) using a custom-built MiniVGG architecture on the CIFAR-10 dataset with PyTorch and CUDA. This project also explores training efficiency using `pin_memory`, batch normalization, and a learning rate scheduler.

---

## ⚙️ Model Architecture

Two model versions are included in this repo:

### 🧪 Original MiniVGG
A basic convolutional model without BatchNorm or learning rate scheduling.

### 🔧 Optimized MiniVGG
An improved architecture with:

- ✅ **Batch Normalization** after each convolution
- ✅ **StepLR Learning Rate Scheduler**
- ✅ Better generalization and stability

#### 📐 Layers in Optimized MiniVGG:

**Convolutional Blocks:**
- Conv2D (3, 32, 3, padding=1) → BatchNorm2D → ReLU  
- Conv2D (32, 32, 3, padding=1) → BatchNorm2D → ReLU  
- MaxPool2D (2, 2)

- Conv2D (32, 64, 3, padding=1) → BatchNorm2D → ReLU  
- Conv2D (64, 64, 3, padding=1) → BatchNorm2D → ReLU  
- MaxPool2D (2, 2)

**Fully Connected:**
- Flatten  
- Linear (64×8×8, 512) → ReLU → Dropout(0.5)  
- Linear (512, 10)

---

## 📊 Dataset

- Dataset: **CIFAR-10**
- Training Samples: 50,000
- Test Samples: 10,000
- Classes: Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

---

## 🔧 Hyperparameters (Optimized Model)

| Parameter       | Value            |
|----------------|------------------|
| Epochs         | 20               |
| Batch Size     | 64               |
| Optimizer      | Adam             |
| Initial LR     | 0.001            |
| LR Scheduler   | StepLR(step=10, γ=0.5) |
| Loss Function  | CrossEntropyLoss |
| Device         | CUDA             |
| Pin Memory     | ✅ Enabled (True) |

---

## 🚀 Training Results (Optimized Model)

| Epoch | Accuracy | Training Loss |
|-------|----------|----------------|
| 1     | 79.30%   | 0.1747 |
| 5     | 79.32%   | 0.1023 |
| 10    | 79.33%   | 0.0957 |
| 15    | 79.24%   | 0.0961 |
| 20    | **79.36%** ✅ | 0.0969 |

📈 **Previous Best Accuracy**: 77.36%  
🔥 **Improved Accuracy**: **79.36%**

---

## ⏱ Training Time Comparison

| Model Version         | pin_memory | Epochs | Total Time     |
|----------------------|------------|--------|----------------|
| MINI_VGG (original)  | ❌ No       | 10     | 3.94 minutes   |
| MINI_VGG (original)  | ✅ Yes      | 10     | **2.84 minutes** ✅ |
| MINI_VGG (optimized) | ✅ Yes      | 20     | **5.98 minutes** ✅ |

✅ **pin_memory** consistently reduces data transfer time from CPU to GPU, improving training throughput.

📌 Even with double the epochs, the optimized model completes training under 6 minutes.

---

## 🧠 Key Improvements

- ✅ Added **Batch Normalization** for faster convergence  
- ✅ Introduced **Learning Rate Scheduler** for adaptive learning  
- ✅ Improved **training accuracy and stability**  
- ✅ CUDA efficiency with **pin_memory=True**

---

## 💾 Files in this Repo

| File                          | Description                                     |
|-------------------------------|-------------------------------------------------|
| `MINI_VGG.ipynb`              | Original MiniVGG training script               |
| `Mini_VGG optimized.ipynb`    | Optimized model with BatchNorm + LR Scheduler |
| `README.md`                   | This documentation                             |

---

## 🔬 CUDA Optimization Insight

- `pin_memory=True` enables faster data transfer from host (CPU) to device (GPU).
- Especially beneficial in smaller datasets like CIFAR-10 with medium batch sizes.
- Helps overlap CPU-GPU workloads using asynchronous streams.
