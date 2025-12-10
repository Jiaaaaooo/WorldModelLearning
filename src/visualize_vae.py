import sys
sys.path.append(".")
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.vae import VAE
from src.dataset import CarRacingDataset
# 配置
CHECKPOINT_PATH = "runs/vae_checkpoints/vae_epoch_30.pth" # 加载刚才训练最好的模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize():
    # 1. 准备数据和模型
    dataset = CarRacingDataset()
    model = VAE().to(DEVICE)
    
    # 加载权重
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval() # 切换到评估模式 (关闭 Dropout 等)
    
    # 2. 随机抽取 5 张图
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    plt.figure(figsize=(10, 4))
    
    for i, idx in enumerate(indices):
        # 获取原图
        img_tensor = dataset[idx].unsqueeze(0).to(DEVICE) # [1, 3, 64, 64]
        
        # 让 VAE 重构
        with torch.no_grad():
            recon_img, _, _ = model(img_tensor)
            
        # --- 转换为图片格式以便显示 ---
        # Tensor [1, 3, 64, 64] -> Numpy [64, 64, 3]
        orig = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        recon = recon_img.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # 画原图
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(orig)
        plt.title("Original")
        plt.axis("off")
        
        # 画重构图
        ax = plt.subplot(2, 5, i + 6)
        plt.imshow(recon)
        plt.title("VAE Reconstructed")
        plt.axis("off")
    
    # 保存结果
    save_path = "vae_result.png"
    plt.savefig(save_path)
    print(f"🖼️ Visualization saved to {save_path}. Check it in VS Code!")

if __name__ == "__main__":
    visualize()