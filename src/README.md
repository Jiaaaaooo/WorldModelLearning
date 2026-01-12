# World Model Learning Roadmap 🌍

> **Current Status**: Phase 1 - VAE Backbone Implemented.  
> **Hardware**: 2x RTX 3090 (Local Debug) | 4x A100 (Cluster Training)

---

## 🛠️ Environment Setup (Reproducible)
**Strictly follow these versions to avoid CUDA errors.**

```bash
# 1. Base Environment
conda create -n lj_wml python=3.10 -y
conda activate lj_wml

# 2. PyTorch (Compatible with CUDA 11.6)
# Driver Version: 570.x | CUDA Runtime: 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# 3. Core Dependencies
pip install numpy matplotlib einops tensorboard gymnasium[box2d] opencv-python
```

## 🧠 Model Architecture: VAE (Vision)
Implemented in `src/vae.py`.
- **Input**: $ Image
- **Encoder**: 4-layer CNN -> Flatten -> Latent Distribution ($\mu, \sigma$)
- **Latent Dim**: 32 (Compressed World Representation)
- **Decoder**: Latent Vector -> 4-layer Deconv -> Reconstruction

## 📅 Progress Log
- [x] **Day 1**: Project Initialization (Git, SSH, Layout).
- [x] **Day 2**: VAE Backbone Implementation & Environment Fix.
- [ ] **Day 3**: Data Collection (Gymnasium) & Training Loop.

## 🚀 Quick Start
Test the VAE model structure:
```bash
python src/vae.py
# Expected: ✅ VAE Model check passed!
```

## 📅 Day 3: Data Pipeline & VAE Training (Vision System)
> **Date**: 2025-12-10
> **Focus**: Data Collection, Dataset Pipeline, ELBO Loss, Model Training.

### 1. 🏁 Work Summary
Today, I successfully completed the **"Vision"** component of the World Model. The VAE can now "see" the CarRacing environment and compress high-dimensional images (64x64x3) into a compact latent vector ($z=32$).

### 2. 🛠️ Implementation Details

#### **A. Data Collection (`src/generate_data.py`)**
*   **Environment**: Upgraded to `CarRacing-v3` (fixed `gymnasium` versioning issue).
*   **Process**:
    *   Random policy agent collects data.
    *   **Preprocessing**: Resized original frames (96x96) to **64x64** (standard World Model size).
    *   **Debug**: Implemented automatic image saving (`debug_images/`) to verify visual quality.
*   **Output**: `data/train_data.npy` (10,000 frames).

#### **B. Training Pipeline (`src/train_vae.py`)**
*   **Architecture**: VAE (Conv2d Stride=2 Encoder + TransposeConv Decoder).
*   **Loss Function**: **ELBO (Evidence Lower Bound)**.
    *   `Reconstruction Loss (MSE)`: Ensures image similarity.
    *   `KL Divergence`: Regularizes latent space to $N(0, I)$.
*   **Hyperparameters**:
    *   Batch Size: `128`
    *   Learning Rate: `1e-4` (Adam)
    *   Epochs: `30`

### 3. 📊 Experimental Results (RTX 3090)

| Metric | Start (Epoch 1) | End (Epoch 30) | Notes |
| :--- | :--- | :--- | :--- |
| **Total Loss** | ~561.12 | **~11.01** | Massive convergence in first 2 epochs. |
| **MSE (Recon)** | ~550.0 | ~10.0 | The car and track structure are reconstructed accurately. |
| **Time** | - | **0.8 mins** | Extremely fast training on 3090. |

**Visualization Analysis (`src/visualize_vae.py`)**:
*   The reconstructed images are slightly **blurry**.
*   **Why?** This is expected behavior for VAEs using MSE loss. The model predicts the *mean* of the pixel distribution, filtering out high-frequency noise (grass texture) while preserving key semantic information (road curve, car position). **This proves the model works.**

### 4. 💻 Reproducibility (Commands)

```bash
# 1. Generate Dataset (and view debug images)
python src/generate_data.py

# 2. Train the VAE
python src/train_vae.py

# 3. Visualize Results (Comparison Plot)
python src/visualize_vae.py
```

### 5. 📂 Git Version Control
*   **Branch**: `feat-training`
*   **Commit Message**: `feat: finish day3, trained VAE with visualization results`
*   **Status**: Merged to remote.

## 📅 Day 4: Building the Memory (MDN-RNN)
> **Date**: 2026-01-12
> **Focus**: Action Recording, Latent Space Preprocessing, MDN-RNN Architecture.

### 1. 🎯 Work Summary
Transitioned from "Vision" (VAE) to "Memory" (RNN). The goal is to predict the future latent state $z_{t+1}$ given the current state $z_t$ and action $a_t$.
*   **Solved the "Fork in the Road" problem**: Implemented Mixture Density Networks (MDN) to handle stochastic future predictions (e.g., the car might turn left OR right).
*   **Data Pipeline 2.0**: Upgraded collection scripts to include **Actions** and perform **Dimensionality Reduction** using the pre-trained VAE.

### 2. 🛠️ Implementation Details

#### **A. Data Collection Upgrade (`src/generate_data.py`)**
*   **Update**: Now records agent actions ($a \in \mathbb{R}^3$) alongside observations.
*   **Format**: `.npz` compressed format.
*   **Volume**: 500 episodes (approx. 50k frames).

#### **B. Latent Preprocessing (`src/process_data.py`)**
*   **Objective**: Speed up training by 100x. Instead of training RNN on pixels, we train on VAE latent vectors.
*   **Process**:
    *   Input: Images $(N, 64, 64, 3)$ -> VAE Encoder -> Output: Latent Vectors $\mu$ $(N, 32)$.
*   **Result**: 50,000 frames compressed into a lightweight dataset (`processed_data.npz`).

#### **C. Model Architecture (`src/rnn.py`)**
*   **Core**: LSTM (Hidden Size=256).
*   **Input**: Concatenation of Latent Vector $z$ (32) + Action $a$ (3) = 35 dim.
*   **Output (MDN Head)**:
    *   Predicts parameters for **5 Gaussian distributions**.
    *   $\pi$ (Logits), $\mu$ (Means), $\sigma$ (Scales).
*   **Loss Function**: Negative Log Likelihood (NLL) of the mixture distribution.

### 3. 💻 Reproducibility (Commands)

```bash
# 1. Collect data with actions (v2)
python src/generate_data.py

# 2. Compress images to latent vectors using VAE
python src/process_data.py
# Output: data/processed_data.npz

# 3. Verify RNN Architecture
python src/rnn.py
# Expected: ✅ MDN-RNN Architecture Check Passed!

*Maintained by @Jiaaaaooo*
