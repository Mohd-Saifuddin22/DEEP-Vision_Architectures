# DEEP Vision Architectures: MiniVGG, ResNet & Vision Transformer on CIFAR-10 (CUDA Optimized)

---

## 📌 Project Objective

Build, train, and compare multiple deep learning architectures on the CIFAR-10 dataset using PyTorch with CUDA acceleration. This repository showcases an evolutionary journey from classical CNNs to advanced Transformer-based vision models:

1. ✅ MiniVGG — Simple convolutional model  
2. ✅ MiniVGG Optimized — BatchNorm & LR Scheduling  
3. ✅ MiniResNet — Residual learning for deeper representation  
4. ✅ Vision Transformer (ViT) — Attention-based representation learning

---

## 📊 Dataset Details

- **Dataset**: CIFAR-10  
- **Images**: 60,000 color images of 10 categories (32×32)  
- **Split**: 50,000 for training, 10,000 for testing  
- **Classes**: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

---

## 🧠 Model Architectures

### 🔹 MiniVGG (Baseline CNN)
- 4 Conv layers → MaxPooling  
- Fully Connected → Dropout → Output  
- ❌ No normalization  
- ❌ No scheduler  

### 🔹 MiniVGG Optimized
- Same layout as MiniVGG  
- ✅ Batch Normalization after each Conv  
- ✅ StepLR Scheduler (step=10, γ=0.5)  
- ✅ More stable and higher accuracy  

### 🔹 MiniResNet
- 2 Residual Blocks with Conv-BN-ReLU  
- Initial Conv → Residual Blocks → Adaptive Pool → FC  
- ✅ ReduceLROnPlateau Scheduler  
- ✅ Early Stopping  
- ✅ Better generalization & gradient flow  

### 🔹 Vision Transformer (ViT)
- Patch-based embedding using Conv2D  
- Learnable [CLS] token and positional embeddings  
- Transformer Encoder with 8 layers, 8 heads  
- MLP classification head  
- ✅ Warmup + Cosine Annealing LR Scheduler  
- ✅ RandAugment + Strong Regularization  
- ✅ Attention-based global learning

---

## 🧪 Results Comparison

| Model               | Epochs | Final Val Acc | Final Test Acc | Best Val Epoch | Total Training Time |
|--------------------|--------|----------------|----------------|----------------|----------------------|
| MiniVGG            | 10     | 77.64%         | **77.36%**     | 10             | 2.84 min (pin=True)  |
| MiniVGG Optimized  | 20     | 79.36%         | **79.36%** ✅   | 20             | 5.98 min             |
| MiniResNet         | 20     | 77.96%         | **78.05%**     | 20             | ~6.5 min             |
| Vision Transformer | 25     | 78.62%         | **78.62%**     | 25             | **1498.44 sec (~25 min)** ⏱️ |

---

## 🔧 Training Details

| Feature               | MiniVGG | MiniVGG Optimized | MiniResNet | Vision Transformer |
|-----------------------|---------|-------------------|------------|---------------------|
| BatchNorm             | ❌      | ✅                | ✅         | ✅                  |
| Scheduler             | ❌      | StepLR            | ReduceLROnPlateau | Warmup + Cosine   |
| Early Stopping        | ❌      | ❌                | ✅         | ❌                  |
| Optimizer             | Adam    | Adam              | Adam       | AdamW               |
| Augmentation          | Basic   | Basic             | RandCrop + Flip | RandAugment + Flip |
| Epochs                | 10      | 20                | 20         | 25                  |
| Pin Memory            | ✅      | ✅                | ✅         | ✅                  |

---

## 🧩 Transformer Details (ViT)

| Parameter           | Value    |
|---------------------|----------|
| Patch Size          | 4×4      |
| Embedding Dim       | 256      |
| Layers (Depth)      | 8        |
| Attention Heads     | 8        |
| MLP Hidden Dim      | 512      |
| Dropout             | 0.1      |
| LR Scheduler        | WarmupCosineAnnealingLR |
| LR                  | 3e-4     |
| RandAugment         | ✅ num_ops=2, magnitude=9 |
| Weight Decay        | 1e-4     |
| Final Accuracy      | **78.62%** 🎯 |

---

## 📦 Repository Structure

| File / Folder                | Description |
|-----------------------------|-------------|
| `MINI_VGG.ipynb`            | Baseline CNN |
| `Mini_VGG optimized.ipynb`  | CNN with BatchNorm + Scheduler |
| `Mini_ResNet.ipynb`         | Lightweight ResNet |
| `VisionTransformer.ipynb`   | Vision Transformer on CIFAR-10 |
| `README.md`                 | This full documentation |
| `./data`                    | CIFAR-10 dataset (downloaded automatically) |

---

## 🔮 Insights & Observations

- **MiniVGG Optimized** gave the best accuracy (79.36%) in shortest time.
- **ResNet** introduced better gradient flow via skip connections.
- **ViT** achieved 78.62% with attention and no convolutions after patching.
- Training ViT is **more resource-intensive** but demonstrates strong performance with enough tuning.
- ⚠️ ViT requires **more data or augmentations** to outperform CNNs on small datasets.



⭐ If this repo helped you, please **star it** and share your feedback!

