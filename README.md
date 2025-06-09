## Deep CNN Training: MiniVGG on CIFAR-10 (CUDA Enabled)

---

## 📌 Task Objective

Train a deeper Convolutional Neural Network (CNN) using a custom-built MiniVGG architecture on the CIFAR-10 dataset with PyTorch and CUDA, while experimenting with hyperparameters and optimizing data transfer using `pin_memory`.

---

## ⚙️ Model Architecture: MiniVGG

The model consists of:

### 🧮 Convolutional Layers:

- Conv2D (3, 32, kernel_size=3, padding=1)
- ReLU
- Conv2D (32, 32, kernel_size=3, padding=1)
- ReLU
- MaxPool2D (kernel_size=2, stride=2)

- Conv2D (32, 64, kernel_size=3, padding=1)
- ReLU
- Conv2D (64, 64, kernel_size=3, padding=1)
- ReLU
- MaxPool2D (kernel_size=2, stride=2)

### 🧮 Classifier (Fully Connected Layers):

- Flatten
- Linear (64*8*8, 512)
- ReLU
- Dropout(0.5)
- Linear (512, 10)

---

## 📊 Dataset

- Dataset: **CIFAR-10**
- Training Samples: 50,000
- Test Samples: 10,000
- Classes: Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

---

## 🔧 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Device | CUDA |
| Pin Memory | ✅ Enabled (True) |

---

## 🚀 Training Results

| Epoch | Accuracy |
|-------|----------|
| 1 | 60.99% |
| 2 | 68.83% |
| 3 | 71.66% |
| 4 | 74.85% |
| 5 | 75.90% |
| 6 | 76.95% |
| 7 | 76.72% |
| 8 | 76.93% |
| 9 | 77.64% |
| 10 | **77.36%** ✅ |

---

## ⏱ Training Time Comparison

| Setting | Total Time |
|---------|-------------|
| Without pin_memory | 3.94 minutes |
| With pin_memory | **2.84 minutes** ✅ |

✅ **pin_memory improved CUDA pipeline efficiency, reducing training time by ~28%.**

---

## 🔬 CUDA Optimization Insight

- DataLoader with `pin_memory=True` allows faster asynchronous memory copy from CPU to CUDA device.
- This optimization is particularly useful when working with small-to-medium sized datasets and large number of epochs.

---

## 💾 Model Saving

- The final trained model is saved as:
