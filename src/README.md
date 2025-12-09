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

---
*Maintained by @Jiaaaaooo*
